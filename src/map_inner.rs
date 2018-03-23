// LockFreeHashMap -- A concurrent, lock-free hash map for Rust.
// Copyright (C) 2018  rolag
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use crossbeam;
use crossbeam::epoch::{Guard, Shared};
use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::num::Wrapping;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use atomic::{AtomicBox, AtomicPtr, MaybeNull, NotNull, NotNullOwned};

#[derive(Debug)]
pub enum KeySlot<K> {
    /// Key inserted into table. Associated value may or may not have been removed.
    Key(K),
    /// This was an empty slot that is now taken. There is a newer (resized) table that should be
    /// used for a key slot.
    SeeNewTable,
}

#[derive(Debug)]
pub enum ValueSlot<'a, V: 'a> {
    /// Value insert into table.
    Value(V),
    /// Slot used to contain a `Value` but was deleted from the map.
    Tombstone,
    /// Value was inserted into a previous smaller table that needed to be resized and was copied
    /// directly into a newer table.
    ValuePrime(&'a ValueSlot<'a, V>),
    /// Value was a `Tombstone` in a previous smaller table that needed to be resized and was
    /// copied directly into a newer table.
    TombstonePrime,
}

impl<'a, V> ValueSlot<'a, V> {
    pub fn is_tombstone(&self) -> bool {
        match self {
            &ValueSlot::Tombstone => true,
            _ => false,
        }
    }

    pub fn is_tombprime(&self) -> bool {
        match self {
            &ValueSlot::TombstonePrime => true,
            _ => false,
        }
    }

    pub fn is_prime(&self) -> bool {
        match self {
            &ValueSlot::TombstonePrime | &ValueSlot::ValuePrime(_) => true,
            _ => false,
        }
    }

    pub fn as_inner(value: Option<&Self>) -> Option<&V> {
        match value {
            Some(&ValueSlot::Value(ref v)) => Some(&v),
            Some(&ValueSlot::ValuePrime(v)) => ValueSlot::as_inner(Some(v)),
            _ => None,
        }
    }
}

/// This enum represents what key/value pair to match when searching in `put_if_match()`.
#[derive(Debug)]
pub enum Match {
    /// Match if there is no key/value pair in the map
    Empty,
    /// Match if there is a key/value pair in the map
    AnyKeyValuePair,
    /// Always match
    Always,
}

pub enum KeyCompare<'k, 'q, K: 'k + Borrow<Q>, Q: 'q + ?Sized> {
    Owned(NotNullOwned<K>),
    Shared(NotNull<'k, KeySlot<K>>),
    OnlyCompare(&'q Q),
}

impl<'k, 'q, K: Borrow<Q>, Q: ?Sized> KeyCompare<'k, 'q, K, Q> {
    pub fn new(key: K) -> Self {
        KeyCompare::Owned(NotNullOwned::new(key))
    }
    fn as_qref(&self) -> QRef<K, Q> {
        match self {
            &KeyCompare::Owned(ref owned) => QRef::Owned(owned),
            &KeyCompare::Shared(ref not_null) => QRef::Shared(not_null),
            &KeyCompare::OnlyCompare(q) => QRef::Borrow(q),
        }
    }
}

enum QRef<'k, 'q, K: 'k + Borrow<Q>, Q: 'q + ?Sized> {
    Owned(&'k NotNullOwned<K>),
    Shared(&'k KeySlot<K>),
    Borrow(&'q Q),
}

impl<'k, 'q, K: Borrow<Q>, Q: ?Sized> QRef<'k, 'q, K, Q> {
    fn as_qref2(&self) -> QRef2<K, Q> {
        match self {
            &QRef::Owned(not_null) => QRef2::Shared(&**not_null),
            &QRef::Shared(&KeySlot::Key(ref k)) => QRef2::Shared(k),
            &QRef::Shared(&KeySlot::SeeNewTable) =>
                unreachable!("KeyCompare must contain a `NotNull(KeySlot::Key(K))`"),
            &QRef::Borrow(q) => QRef2::Borrow(q),
        }
    }
}

enum QRef2<'k, 'q, K: 'k + Borrow<Q>, Q: 'q + ?Sized> {
    Shared(&'k K),
    Borrow(&'q Q),
}

impl<'k, 'q, K: Borrow<Q>, Q: ?Sized> QRef2<'k, 'q, K, Q> {
    fn as_q(&self) -> &Q {
        match self {
            &QRef2::Shared(k) => k.borrow(),
            &QRef2::Borrow(q) => q,
        }
    }
}

/// This enum represents the value to insert when calling `put_if_match()`.
#[derive(Debug)]
pub enum PutValue<'a, V: 'a> {
    Owned(NotNullOwned<ValueSlot<'a, V>>),
    Shared(NotNull<'a, ValueSlot<'a, V>>),
}

impl<'a, V> PutValue<'a, V> {
    pub fn new(value: V) -> Self {
        PutValue::Owned(NotNullOwned::new(ValueSlot::Value(value)))
    }
    pub fn new_tombstone() -> Self {
        PutValue::Owned(NotNullOwned::new(ValueSlot::Tombstone))
    }
    pub fn is_tombstone(&self) -> bool {
        match self {
            &PutValue::Owned(ref owned) => if let ValueSlot::Tombstone = **owned {
                true
            } else {
                false
            },
            &PutValue::Shared(ref not_null) => {
                if let &ValueSlot::Tombstone = &**not_null {
                    true
                } else {
                    false
                }
            },
        }
    }
}

/// A map containing a unique, non-resizable array to the Key/Value pairs. If the map needs to be
/// resized, a new `MapInner` must be created and its Key/Value pairs must be copied from this one.
/// Logically, this struct owns its keys and values, and so is responsible for freeing them when
/// dropped.
pub struct MapInner<'map, K, V: 'map, S = RandomState> {
    map: Vec<(AtomicPtr<KeySlot<K>>, AtomicPtr<ValueSlot<'map, V>>)>,
    /// The amount of key/value pairs in the array, if any.
    size: AtomicUsize,
    /// Points to the newer map or null if none.
    newer_map: AtomicPtr<MapInner<'map,K,V,S>>,
    resizers_count: AtomicUsize,
    chunks_copied: AtomicUsize,
    hash_builder: S,
}


impl<'map, K: Hash + Eq, V> MapInner<'map,K,V,RandomState> {
    /// Creates a new `MapInner`. Uses the next power of two if size is not a power of two.
    pub fn with_capacity(size: usize) -> Self {
        MapInner::with_capacity_and_hasher(size, RandomState::new())
    }
}

/// `Clone` is required on `K` because of `put_if_match()`.
impl<'guard, 'map: 'guard, K, V, S> MapInner<'map, K,V,S>
    where K: Hash + Eq,
          S: BuildHasher + Clone,
{
    /// The default size of a new `LockFreeHashMap` when created by `MapInner::with_capacity()`.
    pub const DEFAULT_CAPACITY: usize = 8;

    pub fn with_capacity_and_hasher(size: usize, hasher: S) -> Self {
        let size = usize::checked_next_power_of_two(size).unwrap_or(Self::DEFAULT_CAPACITY);
        let mut map = Vec::with_capacity(size);
        for _ in 0..size {
            map.push((AtomicPtr::new(None), AtomicPtr::new(None)));
        }
        MapInner {
            map: map,
            size: AtomicUsize::new(0),
            newer_map: AtomicPtr::new(None),
            resizers_count: AtomicUsize::new(0),
            chunks_copied: AtomicUsize::new(0),
            hash_builder: hasher,
        }
    }

    /// Help copy a small chunk of the map to the `newer_map`. See `COPY_CHUNK_SIZE` for the
    /// default chunk size.
    pub fn help_copy(
        &self,
        newer_map: NotNull<Self>,
        copy_everything: bool,
        outer_map: &AtomicBox<Self>,
        guard: &'guard Guard,
    ) {
        /// Checked multiplication that gives an `upper_bound` value if the multiplication exceeds
        /// the bound (or if overflow occurs).
        fn checked_times(first: usize, second: usize, upper_bound: usize) -> usize {
            let result = first * second;
            if first != 0 && result/first != second {
                upper_bound
            } else if result > upper_bound {
                upper_bound
            } else {
                result
            }
        }
        loop {
            // `chunks_copied` is an atomic variable that keeps track of which chunk will be copied
            // into the newer table.
            let chunks_copied = self.chunks_copied.fetch_add(1, Ordering::SeqCst);
            let next_chunk = chunks_copied + 1;
            // Next find the element-wise lower and upper bounds respectively.
            let lower_bound = checked_times(chunks_copied, ::COPY_CHUNK_SIZE, self.capacity());
            let upper_bound = checked_times(next_chunk, ::COPY_CHUNK_SIZE, self.capacity());
            debug_assert!(lower_bound <= upper_bound);
            // If they're equal, then we know another thread incremented `chunks_copied` such that
            // `(chunks_copied + 1) * COPY_CHUNK_SIZE` will be equal to the size of the current
            // map. Therefore, we have nothing left to copy and can return.
            if lower_bound == upper_bound || lower_bound > upper_bound {
                // But before we do, we should decrement `chunks_copied` by 1, just to make sure it
                // won't overflow such that two threads call `self.promote()` below. It still can
                // overflow if you have `usize::MAX` amount of threads calling `help_copy`, but it
                // will be assumed that this never happens.
                self.chunks_copied.fetch_sub(1, Ordering::SeqCst);
                // TODO: In case finish copying newer_map before this map (can happen with small
                // maps?; can't guanaentee it anyway...)
                self.promote(newer_map, outer_map, guard);
                return;
            }
            // Now because `lower_bound` must be less than `upper_bound`, and since we already
            // assumed that any thread that gets some `chunks_copied` MUST then
            // copy all elements in that chunk, we MUST do so.
            // Notice that the `..upper_bound` is exclusive, so it never exceeds
            // (self.capacity() - 1), which is 1-indexed, whereas the array itself will be
            // 0-indexed. So this is fine.
            for i in lower_bound..upper_bound {
                // Now simply copy_slot() for each element.
                self.copy_slot(&*newer_map, i, outer_map, guard);
            }
            // The `outer_map`, is the `inner: AtomicBox<MapInner<_>>` field in the
            // `LockFreeHashMap` struct. Once everything has been copied, which is defined as when
            // the chunk whose last element copied is the last element, the `inner` field must be
            // promoted to the `newer_map`.
            if upper_bound == self.capacity() {
                self.promote(newer_map, outer_map, guard);
                return;
            } else if !copy_everything {
                // Otherwise, we did not copy everything and there is still more to be done,
                // or at least there was more when we last checked `chunks_copied`. If we are not
                // required to copy everything, then just return and let some other thread do it.
                return
            }
            // Otherwise, continue the loop and keep copying until everything is copied.
        }
    }

    /// Once a `MapInner` has had all its elements copied to its `newer_map` field,
    /// the LockFreeHashMap's `inner` field must be promoted so that its effects are visible
    /// globally.
    pub fn promote(
        &self,
        new_map: NotNull<Self>,
        outer_map: &AtomicBox<Self>,
        guard: &'guard Guard,
    ) -> bool
    {
        // We only have a `&self` reference to the current `MapInner`. Thus, we need to load it
        // manually from `outer_map`, which must be passed as a parameter throughout various
        // function calls...
        let current_map_shared: NotNull<_> = outer_map.load(guard);
        let current_map: &MapInner<_,_,_> = &*current_map_shared;
        // promote() is only called by help_copy(), which ensures that only the thread that copies
        // the last chunk can promote it. If any function tries to modify the newer_map, it must
        // ensure that it's copied the slot it's working with before doing any modifiction. Thus,
        // we know that 
        // the 
        if current_map as *const _ != self as *const _ {
            return false;
        }
        match outer_map.compare_and_set_shared(current_map_shared, new_map, guard) {
            Ok(_) => {
                unsafe { guard.defer(move || current_map_shared.as_shared().into_owned()); }
                return true;
            },
            Err(_) => {
                return false;
            },
        }
    }

    fn ensure_slot_copied(
        &self,
        copy_index: usize,
        outer_map: &AtomicBox<Self>,
        guard: &'guard Guard,
    ) -> NotNull<'guard, Self> {
        let newer_map_shared = self.newer_map.load(&guard);
        if let Some(new_map) = newer_map_shared.as_option() {
            self.copy_slot(&*new_map, copy_index, outer_map, guard);
            self.help_copy(new_map, false, outer_map, guard);
            new_map
        } else {
            unreachable!()
        }
    }

    // Returns true if this thread successfully copied a slot from the old to the new
    pub fn copy_slot(
        &self,
        new_map: &Self,
        old_map_index: usize,
        outer_map: &AtomicBox<Self>,
        guard: &'guard Guard
    ) {
        fn cheat_lifetime<'guard, 'map, V>(maybe: MaybeNull<'guard, V>) -> MaybeNull<'map, V> {
            MaybeNull::from_shared(Shared::from(maybe.as_shared().as_raw()))
        }
        let (ref atomic_key_slot, ref atomic_value_slot) = self.map[old_map_index];
        let old_key_shared = MaybeNull::from_shared(Shared::null());
        let old_key: NotNull<_>;
        let mut new_key = NotNullOwned::new(KeySlot::SeeNewTable);
        // Preemptively set an empty key slot to `SeeNewTable`.
        loop {
            let cas_key_result = atomic_key_slot.compare_and_set_owned_weak(
                old_key_shared, new_key, guard);
            match cas_key_result {
                Ok(_new_key) => {
                    debug_assert!(if let &KeySlot::SeeNewTable = &*_new_key {true} else {false});
                    return;
                },
                Err((current, new)) => {
                    new_key = new; // Return ownership
                    let _old_key_shared = current;
                    // Because `compare_and_set_weak()` can spuriously fail and therefore still be
                    // null. Thus, just retry with `continue` if it's still null.
                    match current.as_option() {
                        None => continue,
                        Some(k) => {
                            match k.deref() {
                                &KeySlot::SeeNewTable => return,
                                &KeySlot::Key(_) => {
                                    debug_assert!(current.as_option().is_some());
                                    old_key = k;
                                    break;
                                },
                            }
                        }
                    }
                },
            }
        }
        let mut old_value: MaybeNull<_> = cheat_lifetime(atomic_value_slot.load(guard));
        let mut original_valueslot_value = None;
        loop {
            let primed_old_value: NotNull<ValueSlot<_>> = match old_value.as_option() {
                // Swap `None` and `Tombstone` values with `TombstonePrime`.
                None => {
                    if let Err((current, _)) = atomic_value_slot.compare_and_set_owned_weak(
                        MaybeNull::from_shared(Shared::null()),
                        NotNullOwned::new(ValueSlot::TombstonePrime),
                        guard,
                    ) {
                        old_value = cheat_lifetime(current);
                        continue;
                    } else {
                        debug_assert!(atomic_value_slot.load(guard).as_option().expect("null -> T'").deref().is_tombprime());
                        return;
                    }
                },
                Some(not_null) => match not_null.deref() {
                    &ValueSlot::TombstonePrime  => return,
                    &ValueSlot::Tombstone => {
                        match atomic_value_slot.compare_and_set_owned_weak(
                            old_value,
                            NotNullOwned::new(ValueSlot::TombstonePrime),
                            guard,
                        ) {
                            Err((current, _)) => {
                                old_value = cheat_lifetime(current);
                                debug_assert!(old_value.as_option().is_some());
                                continue;
                            },
                            Ok(_new) => {
                                debug_assert!(_new.is_tombprime());
                                unsafe { guard.defer(move || not_null.drop()); }
                                return;
                            }
                        }
                    },
                    // Otherwise prime the old_value and place it back in the map.
                    &ValueSlot::Value(_) => {
                        // old_value was `Value` and not `ValuePrime`.
                        let primed_old_value_owned = NotNullOwned::new(ValueSlot::ValuePrime(not_null.deref()));
                        match atomic_value_slot.compare_and_set_owned_weak(
                            old_value, // `ValueSlot::Value(_)`
                            primed_old_value_owned,
                            guard
                        ) {
                            Err((current, _)) => {
                                old_value = cheat_lifetime(current);
                                debug_assert!(old_value.as_option().is_some());
                                continue;
                            },
                            Ok(shared_primed_value) => {
                                original_valueslot_value = Some(not_null);
                                debug_assert!(!shared_primed_value.is_tombstone());
                                shared_primed_value
                            },
                        }
                    }
                    &ValueSlot::ValuePrime(_)  => not_null,
                }
            };
            if let Some(old_value) = old_value.as_option() {
                let put_value = match old_value.deref() {
                    &ValueSlot::Value(_) => PutValue::Shared(old_value),
                    &ValueSlot::ValuePrime(v) => match v {
                        &ValueSlot::Value(_) => PutValue::Shared(MaybeNull::from_shared(Shared::from(v as *const _)).as_option().expect("was originally not null")),
                        _ => unreachable!("`ValuePrime` can only be a reference to a `Value`"),
                    }
                    &ValueSlot::Tombstone => unreachable!(),
                    &ValueSlot::TombstonePrime => unreachable!(),
                };
                // We copied the key/value pair into the new map if the previous value associated
                // with the key `is_none()`.
                let copied_into_new = new_map.put_if_match(
                    KeyCompare::Shared(old_key),
                    put_value,
                    Match::Empty,
                    outer_map,
                    guard
                ).is_none();
                if copied_into_new {
                    debug_assert!(!atomic_key_slot.is_tagged(guard));
                    debug_assert!(!atomic_value_slot.is_tagged(guard));
                    atomic_key_slot.tag(guard);
                    atomic_value_slot.tag(guard);
                    debug_assert!(atomic_key_slot.is_tagged(guard));
                    debug_assert!(atomic_value_slot.is_tagged(guard));
                }
            }
            let mut primed_old_value_maybe: MaybeNull<_> = primed_old_value.as_maybe_null();
            loop {
                match atomic_value_slot.compare_and_set_owned(
                    primed_old_value_maybe, NotNullOwned::new(ValueSlot::TombstonePrime), guard
                ) {
                    Ok(_) => {
                        unsafe { primed_old_value_maybe.try_defer_drop(guard); }
                        break;
                    },
                    Err((current, _)) => primed_old_value_maybe = current,
                }
            }
            if let Some(original_value) = original_valueslot_value {
                unsafe { guard.defer(move || {
                    if atomic_value_slot.is_tagged(guard) {
                        original_value.drop();
                    }
                })}
            }
            return;
        }
    }

    /// If `newer_map` doesn't exist, then this function tries to allocate a newer map that's
    /// double the size of `self`.
    ///
    /// Returns a `Shared` pointer to the newer map
    pub fn create_newer_map(&self, guard: &'guard Guard) -> NotNull<'guard, Self>
    {
        fn try_double(current_size: usize) -> usize {
            let doubled_size = current_size << 1;
            if doubled_size < current_size {
                current_size
            } else {
                doubled_size
            }
        }
        let newer_map: MaybeNull<Self> = self.newer_map.load(guard);
        if let Some(not_null) = newer_map.as_option() {
            return not_null;
        }
        let size = self.size.load(Ordering::SeqCst);
        let mut new_size = size;
        // Double size if map is >25% full
        if size > (self.capacity() >> 2) {
            new_size = try_double(new_size);
            // Double size if map is >50% full
            if size > (self.capacity() >> 1) {
                new_size = try_double(new_size);
            }
        }
        let array_element_byte_size: usize
            = ::std::mem::size_of::<(AtomicPtr<KeySlot<K>>, AtomicPtr<ValueSlot<'map, V>>)>();
        // This doesn't need to be accurate, so it can be wrapping to ensure it never panics.
        let Wrapping(size_in_megabytes)
            = (Wrapping(array_element_byte_size) * Wrapping(size)) >> (2^10 * 2^10);
        let current_resizers = self.resizers_count.fetch_add(1, Ordering::SeqCst);
        if current_resizers >= 2 && size_in_megabytes > 0 {
            let newer_map: MaybeNull<Self> = self.newer_map.load(guard);
            if let Some(not_null) = newer_map.as_option() {
                return not_null;
            }
            ::std::thread::sleep(Duration::from_millis(size_in_megabytes as u64));
        }
        let newer_map: MaybeNull<Self> = self.newer_map.load(guard);
        if let Some(not_null) = newer_map.as_option() {
            return not_null;
        }
        match self.newer_map.compare_null_and_set_owned(
            NotNullOwned::new(Self::with_capacity_and_hasher(new_size, self.hash_builder.clone())),
            guard
        ) {
            Ok(shared_newer_map) => {
                shared_newer_map
            },
            Err((current, _drop_our_map)) => {
                current
            },
        }
    }

    pub fn hash_key<Q: ?Sized>(&self, key: &Q) -> usize
        where K: Borrow<Q>,
              Q: Hash + Eq,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        // Assumes usize <= u64
        let hash = hasher.finish() as usize;
        // Since the len()/capacity() of the map is always a power of two, we can use a bitwise-and
        // operation
        hash & (self.capacity() - 1)
    }

    pub fn keys_are_equal<T1: ?Sized, T2: ?Sized>(&self, first: &T1, second: &T2) -> bool
        where T2: PartialEq<T1>,
    {
        second == first
    }

    /// Returns the capacity of the current map; i.e. the length of the `Vec` storing the key/value
    /// pairs.
    pub fn capacity(&self) -> usize {
        self.map.capacity()
    }

    /// Returns the size of the current map at some point in time; i.e. the number of key/value
    /// pairs in the map.
    pub fn len(&self) -> usize {
        self.size.load(Ordering::SeqCst)
    }

    pub fn get<Q: ?Sized>(
        &self,
        key: &Q,
        outer_map: &AtomicBox<Self>,
        guard: &'guard Guard
    ) -> Option<&'guard V>
        where K: Borrow<Q>,
              Q: Hash + Eq + PartialEq<K>,
    {
        let initial_index = self.hash_key(key);
        let len = self.capacity();
        for index in (initial_index..len).chain(0..initial_index) {
            let (ref atomic_key_slot, ref atomic_value_slot) = self.map[index];
            match &*atomic_key_slot.load(&guard).as_option()? {
                &KeySlot::Key(ref k) => if self.keys_are_equal(k, key) {
                    match atomic_value_slot.load(&guard).as_option()?.deref() {
                        &ValueSlot::Value(ref v) => return Some(&v),
                        &ValueSlot::Tombstone => return None,
                        &ValueSlot::ValuePrime(_) | &ValueSlot::TombstonePrime => {
                            return self.ensure_slot_copied(index, outer_map, guard)
                                .get(key, outer_map, guard)
                        }
                    }
                } else {
                    continue
                },
                &KeySlot::SeeNewTable => {
                    return self.newer_map.load(&guard)
                        .as_option()
                        .expect("Can't set `KeySlot` to `SeeNewTable` before setting `newer_map`")
                        // It is safe to `unwrap()` because a newer table must exist before any
                        // `KeySlot`s are set to `SeeNewTable`.
                        .get(key, outer_map, guard);
                },
            }
        }
        return None;
    }

    pub fn update_size(
        &'guard self,
        old_value_slot: MaybeNull<'guard, ValueSlot<V>>,
        insert_tombstone: bool,
        guard: &'guard Guard,
    ) -> Option<&'guard ValueSlot<V>>
    {
        let increment = if !insert_tombstone {
            match old_value_slot.as_option() {
                Some(ref old_value) if old_value.is_tombstone() => true,
                None => true,
                _ => false,
            }
        } else {
            false
        };
        if increment {
            self.size.fetch_add(1, Ordering::SeqCst);
        }
        let decrement = if insert_tombstone {
            match old_value_slot.as_option() {
                Some(ref old_value) if old_value.is_tombstone() => false,
                None => false,
                _ => true,
            }
        } else {
            false
        };
        if decrement {
            self.size.fetch_sub(1, Ordering::SeqCst);
        }
        match old_value_slot.as_option() {
            None => None,
            Some(value) => {
                unsafe { guard.defer(move || { value.drop(); })}
                Some(value.deref())
            }
        }
    }

    pub fn put_if_match<Q>(
        &'guard self,
        key: KeyCompare<K, Q>,
        mut put: PutValue<'map, V>,
        matcher: Match,
        outer_map: &AtomicBox<Self>,
        guard: &'guard Guard
    ) -> Option<&'guard ValueSlot<V>>
        where K: Borrow<Q>,
              Q: Hash + Eq + PartialEq<K> + ?Sized,
    {
        fn cheat_lifetime<'guard, 'map, V>(maybe: NotNull<'guard, V>) -> NotNull<'map, V> {
            MaybeNull::from_shared(Shared::from(maybe.as_shared().as_raw()))
                .as_option()
                .expect("parameter was `NotNull` to begin with")
        }
        let initial_index = self.hash_key(key.as_qref().as_qref2().as_q());
        let len = self.capacity();
        let mut key_index = None;
        let mut key = key;
        'find_key_loop:
        for index in (initial_index..len).chain(0..initial_index) {
            let atomic_key_slot: &AtomicPtr<KeySlot<K>> = &self.map[index].0;
            let option_key: Option<_> = atomic_key_slot.load(&guard)
                .as_option();
            let current_key: NotNull<KeySlot<K>> = match option_key {
                Some(existing_key) => existing_key,
                None => if put.is_tombstone() {
                    // The key is not taken, so we don't put a Tombstone value here
                    return None;
                } else {
                    match key {
                        KeyCompare::Owned(owned) => {
                            let insert_key = NotNullOwned::new(KeySlot::Key(*owned.into_owned().into_box()));
                            match atomic_key_slot.compare_null_and_set_owned(insert_key, guard) {
                                Ok(shared_key) => {
                                    // TODO: Raise keyslots-used count
                                    key = KeyCompare::Shared(shared_key);
                                    key_index = Some(index);
                                    break 'find_key_loop;
                                },
                                Err((not_null, _return)) => {
                                    let _return = match *_return.into_owned().into_box() {
                                        KeySlot::Key(owned) => owned,
                                        KeySlot::SeeNewTable => unreachable!(),
                                    };
                                    key = KeyCompare::Owned(NotNullOwned::new(_return));
                                    not_null
                                },
                            }
                        },
                        KeyCompare::Shared(not_null) => {
                            match atomic_key_slot.compare_null_and_set(not_null, guard) {
                                Ok(shared_key) => {
                                    key = KeyCompare::Shared(shared_key);
                                    key_index = Some(index);
                                    break 'find_key_loop;
                                },
                                Err((not_null, _return)) => {
                                    key = KeyCompare::Shared(_return);
                                    not_null
                                }
                            }
                        },
                        KeyCompare::OnlyCompare(_) => {
                            return None;
                        }
                    }
                },
            };
            match &*current_key {
                &KeySlot::Key(ref current_key) => if self.keys_are_equal(key.as_qref().as_qref2().as_q(), current_key.borrow()) {
                    key_index = Some(index);
                    break 'find_key_loop;
                }, // else continue
                &KeySlot::SeeNewTable => {
                    break 'find_key_loop;
                },
            }
        }

        let key_index: usize = match key_index {
            Some(k) => k,
            None => {
                // Key index is not found. So we've exhausted the entire map and need to resize.
                let new_table: NotNull<Self> = self.create_newer_map(guard);
                self.help_copy(new_table, true, outer_map, guard);
                return new_table.deref().put_if_match(key, put, matcher, outer_map, guard);
            },
        };

        // We have now found the key slot to use. This key slot will never change now so we know
        // that we may insert the value into the index `key_index`.

        let atomic_value_slot = &self.map[key_index].1;
        let mut old_value_slot: MaybeNull<_> = atomic_value_slot.load(&guard);
        let newer_map = self.newer_map.load(&guard);
        // TODO: if newer_map == None AND ((current_value is None AND table full) OR value is
        // prime) then resize
        if let Some(_) = newer_map.as_option() {
            return self.ensure_slot_copied(key_index, outer_map, guard)
                .deref()
                .put_if_match(key, put, matcher, outer_map, guard);
        }

        let insert_tombstone = put.is_tombstone();
        loop {
            let value_slot_option = old_value_slot.as_option();
            match matcher {
                Match::Empty => if let Some(v) = value_slot_option {
                    return Some(v.deref())
                },
                Match::AnyKeyValuePair => match value_slot_option.map(|v| v.deref()) {
                    Some(&ValueSlot::Tombstone)
                    | Some(&ValueSlot::TombstonePrime)
                    | None
                        => return None,
                    _ => (),
                }
                Match::Always => (),
            }
            let current_value_slot: MaybeNull<_> = match put {
                PutValue::Owned(owned) => match atomic_value_slot.compare_and_set_owned(
                    old_value_slot, owned, &guard
                ) {
                    Ok(_) => {
                        return self.update_size(old_value_slot, insert_tombstone, guard);
                    },
                    Err((current, _return_ownership)) => {
                        old_value_slot = current;
                        put = PutValue::Owned(_return_ownership);
                        current
                    },
                },
                PutValue::Shared(shared) => match atomic_value_slot.compare_and_set(
                    old_value_slot, shared, &guard
                ) {
                    Ok(_) => {
                        return self.update_size(old_value_slot, insert_tombstone, guard);
                    },
                    Err((current, _return_ownership)) => {
                        old_value_slot = current;
                        put = PutValue::Shared(cheat_lifetime(_return_ownership));
                        current
                    },
                },
            };
            if current_value_slot.as_option().map_or(false, |v| v.is_prime())
            {
                let newer_map = self.newer_map
                    .load(&guard)
                    .as_option()
                    .expect("Can't set a `ValueSlot` to `ValuePrime` before setting `newer_map`");
                self.copy_slot(&*newer_map, key_index, outer_map, guard);
                return newer_map.deref().put_if_match(key, put, matcher, outer_map, guard);
            }
        }
    }
}

impl<'map, K, V, S> MapInner<'map, K, V, S> {
    pub unsafe fn drop_newer_maps(&self, guard: &Guard) {
        if let Some(newer_map) = self.newer_map.take(guard) {
            newer_map.drop_self_and_newer_maps(guard);
        }
    }

    pub unsafe fn drop_self_and_newer_maps(self, guard: &Guard) {
        let newer_map = self.newer_map.take(guard);
        drop(self);
        if let Some(newer_map) = newer_map {
            newer_map.drop_self_and_newer_maps(guard);
        }
    }
}

impl<'map, K, V, S> Drop for MapInner<'map, K, V, S> {
    fn drop(&mut self) {
        let guard = &crossbeam::epoch::pin();
        for (mut k_ptr, mut v_ptr) in self.map.drain(..) {
            unsafe {
                guard.defer(move || {
                    if !k_ptr.is_tagged(&guard) {
                        k_ptr.try_drop(&guard);
                    }
                    v_ptr.try_drop(&guard);
                })
            }
        }
        // Don't drop the `newer_map` ptr, because `self` could have been dropped from `promote()`.
    }
}

impl<'map, K: fmt::Debug, V: fmt::Debug, S> fmt::Debug for MapInner<'map, K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let guard = &crossbeam::epoch::pin();
        write!(f, "MapInner {{ map: {:?}, size: {:?}, capacity: {}, newer_map: {:?}, resizers: {:?}, chunks_copied: {:?} }}",
            self.map.iter()
                .map(|&(ref k, ref v)| (k.load(guard).as_option(), v.load(guard).as_option()))
                .collect::<Vec<_>>(),
            self.size.load(Ordering::SeqCst),
            self.map.capacity(),
            format!("{:?}", self.newer_map),
            self.resizers_count.load(Ordering::SeqCst),
            self.chunks_copied.load(Ordering::SeqCst),
        )
    }
}

