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

//! This module implements most of the logic behind the [::LockFreeHashMap].
//!
//! The talk that Dr. Click gave is available [here](https://www.youtube.com/watch?v=HJ-719EGIts).
//! However, the information below should ideally be enough to understand all the necessary code.
//!
//! The Rust standard library has an implementation of a [HashMap](::std::collections::HashMap).
//! However, to use it concurrently (and safely), one must put a [lock](::std::sync::Mutex) on it.
//! This is very inefficient and can lead to deadlocks.
//! In order to make something lock-free, at least one thread has to make progress after some time.
//! One of the main advantages of using lock-free structures is to avoid deadlocks and livelocks.
//! Concurrent algorithms typically make use of the atomic operation "compare and swap" (CAS),
//!     which atomically swaps one value for another
//!     only if the current value is equal to an expected value.
//! (They can also equivalently use Load Linked/Store Conditional)
//! Considering that a hash map stores a key/value pair (i.e. a key with some associated value),
//!     it is important that a key is always associated with a value it's supposed to be associated
//!     to.
//! This means avoiding an inconsistent state where you have a key/value pair `(K, V)`,
//!     where `V` could never be associated with `K`.
//! Lock-free hash maps have use cases in databases, web caching and large-scale business programs.
//!
//! # Guarantees
//! - This map has the same "guarantees" as simply having some number of global variables
//!     that can be updated atomically.
//! - `O(n)` time complexity, like any hash map.
//!
//! # Valid States
//!
//! ## Key for State Diagram
//! | Value  | Meaning                                                           |
//! | ------ | ----------------------------------------------------------------- |
//! | ∅/null | Empty; No value here.                                             |
//! |   K    | A key is here.                                                    |
//! |   X    | This key slot is taken because there is a newer table available.  |
//! |   V    | A value is here.                                                  |
//! |   V'   | A value is here but the table is currently being resized.         |
//! |   T    | Value was removed.                                                |
//!
//! ## State Diagram
//! ```text
//!   (∅, ∅) ---------> (X, ∅)
//!      |
//!      |
//!     \|/
//!   (K, ∅) ---------> (K, V) <=======> (K, T)
//!      |                 |                |
//!      |                 |                |
//!      |                \|/              \|/
//!      |              (K, V') -------> (K, X)
//!      |                                 /|\
//!      |                                  |
//!      └----------------------------------┘
//! ```
//! From this diagram,
//!     you can see that once a key slot is taken,
//!     it will always point to that key and only that key.
//! Therefore,
//!     if you have two threads (thread 1, thread 2)
//!     inserting `(K1, V1)` and `(K2, V2)` respectively,
//!     where `hash(K1) == hash(K2) == 5 (for example)`,
//!     and `K1 != K2`,
//!     and the array containing the key/value pairs is `(null, null)` at index 5.
//! Then one of the threads, e.g. thread 1,
//!     will perform the transition `(null, null) -> (K1, null)` at array index 5,
//!     allowing it afterwards to insert its value `V1`,
//! The other thread needs to continue probing (at index 6, 7, ...) to find a different key slot.
//! If another thread (thread 3) already inserted a key `K2` at some index after 5,
//!     then obviously thread 2 does not need to insert its key into the map at all.
//! Thus, there are no inconsistent states where you have a value that is paired with a key
//!     that it's not associated to.
//!
//!
//! ## Resizing
//! To resize the map,
//!     a new bigger array needs to be allocated
//!     and all the key/value pairs
//!     have to be moved from their current slots in the current array
//!     into slots in the new array.
//! Because other threads can call `insert()` and `remove()` while the key/value pairs are moved,
//!     there needs to be some way of determining what order this happened in
//!     and how to copy the slot into the newer table.
//! This is done by having any calls that try to access the current slot try and help complete
//!     the copy if they find a `V'` value.
//! So if a thread calls e.g. `get()` while this is happening,
//!     it needs to copy the current slot into the new map before returning.
//! (As an implementation detail, it also helps to copy other slots while it's at it.)
//!
//! This is actually different to the exact algorithm that Click described,
//!     but it appears to achieve the same goals as his.
//!
//! The transitions necessary in both the old and new map are shown below:
//! ```text
//! Transition# |        [1]        |        [2]       |        [3]       |         [4]
//! Old Map     | (K, V) -> (K, V') |                  |                  | (K, V') -> (K, X)
//! New Map     |                   | (∅, ∅) -> (K, ∅) | (K, ∅) -> (K, V) |
//! ```
//! - Transition [1] marks the value slot as being copied.
//!   If another thread tries to access it while it's `V'`,
//!       then it must help complete copying this key/value pair into the newer map.
//!   Note that if V is actually null or `T`,
//!       we can simply set it to `X`
//!       and skip inserting a tombstone/null value into the newer map,
//!       which just wastes a key slot.
//! - Transition [2] and [3] copy the old value into the new map.
//!   They are separate states to remind you that separate threads can perform either.
//!   If inserting it fails,
//!     then we know that some thread didn't care about the current value
//!     and just `insert()`ed a new value anyway.
//!   This is fine and just means that `V` needs to be deallocated.
//!   This is because if the slot wasn't being copied,
//!     it would have overridden the current `V` with another,
//!     which would have ended up being copied into the newer table afterwards.
//! - Finally, transition [4] completes the copy by marking the valueslot as `X`,
//!     meaning there could be something here (or not)
//!     but you need to see the newer table to find out.
//!
//! If multiple threads are trying to `replace()` data at index `i` in the map,
//!     they have to either do it before transition [1] happens,
//!     or after transition [4] happens.
//! The main goal of this data structure is to be completely lock-free/non-blocking.
//! Therefore, instead of looping and waiting until all transitions have finished,
//!     which is essentially a blocking algorithm,
//!     each thread can help make progress by doing any (or all) of the 4 transitions above.

use crossbeam_epoch::{Guard, Shared};
use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::num::Wrapping;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use atomic::{AtomicBox, AtomicPtr, MaybeNull, NotNull, NotNullOwned};

#[derive(Debug)]
/// The hash map is implemented as an array of key-value pairs, where each key and value can be one
/// of several states. This enum represents the various states that a key can be in, excluding the
/// null/empty state.
pub enum KeySlot<K> {
    /// A key has been inserted into table. The key's associated value may or may not have been
    /// removed. Once a key is in this state it can't go into any other state.
    Key(K),
    /// This was an empty slot that is now taken. There is a newer (resized) table that should be
    /// used if this key slot was needed. Once a key is in this state it can't go into any other
    /// state.
    SeeNewTable,
}

#[derive(Debug, PartialEq)]
/// The hash map is implemented as an array of key-value pairs, where each key and value can be one
/// of several states. This enum represents the various states that a value can be in, excluding
/// the null/empty state.
pub enum ValueSlot<'v, V: 'v> {
    /// A value has been inserted into the table.
    Value(V),
    /// This state represents that a key has been inserted but then removed.
    Tombstone,
    /// The table is being resized currently and the value here still needs to be inserted into the
    /// newer table.
    ValuePrime(&'v ValueSlot<'v, V>),
    /// This state represents one of two things:
    ///     1) This was a `ValueSlot::Tombstone(_)` slot that is now taken. There is a newer
    ///        (resized) table that should be used if this value slot was needed.
    ///     2) This used to be a `ValueSlot::Value(_)` slot that has now been copied into the
    ///        newer table.
    /// This is the final state for any `ValueSlot`.
    SeeNewTable,
}

impl<'v, V> ValueSlot<'v, V> {
    /// Returns true if and only if the `ValueSlot` has discriminant `Tombstone`.
    pub fn is_tombstone(&self) -> bool {
        match self {
            &ValueSlot::Tombstone => true,
            _ => false,
        }
    }

    /// Returns true if and only if the `ValueSlot` has discriminant `ValuePrime`.
    pub fn is_valueprime(&self) -> bool {
        match self {
            &ValueSlot::ValuePrime(_) => true,
            _ => false,
        }
    }

    /// Returns true if and only if the `ValueSlot` has discriminant `Value`.
    pub fn is_value(&self) -> bool {
        match self {
            &ValueSlot::Value(_) => true,
            _ => false,
        }
    }

    /// Returns true if and only if the `ValueSlot` has discriminant `SeeNewTable`.
    pub fn is_seenewtable(&self) -> bool {
        match self {
            &ValueSlot::SeeNewTable => true,
            _ => false,
        }
    }

    /// Returns true if and only if the `ValueSlot` has either discriminant `ValuePrime` or
    /// `SeeNewTable`.
    pub fn is_prime(&self) -> bool {
        match self {
            &ValueSlot::SeeNewTable | &ValueSlot::ValuePrime(_) => true,
            _ => false,
        }
    }

    /// Return an `Option` reference to the inner value of generic type `V`.
    pub fn as_inner(value: Option<&Self>) -> Option<&V> {
        match value {
            Some(&ValueSlot::Value(ref v)) => Some(&v),
            Some(&ValueSlot::ValuePrime(v)) => ValueSlot::as_inner(Some(v)),
            _ => None,
        }
    }
}

/// Sometimes, when inserting a new value into the hash map, we only want to insert something if
/// the value already matches something.
///
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

/// Sometimes when calling `put_if_match()` we want to insert a key and sometimes we just want to
/// compare it with some variable of type `Q`. This enum represents which one is intended.
pub enum KeyCompare<'k, 'q, K: 'k + Borrow<Q>, Q: 'q + ?Sized> {
    Owned(NotNullOwned<KeySlot<K>>),
    Shared(NotNull<'k, KeySlot<K>>),
    OnlyCompare(&'q Q),
}

impl<'k, 'q, K: Borrow<Q>, Q: ?Sized> KeyCompare<'k, 'q, K, Q> {
    pub fn new(key: K) -> Self {
        KeyCompare::Owned(NotNullOwned::new(KeySlot::Key(key)))
    }
    /// The purpose of this function is to ultimately get a value of type `&Q`.
    /// Because we need to call `deref()` and `borrow()` a few times, we need to put the result of
    /// these functions somewhere in order to return a reference. Thus, `QRef` and `QRef2` are
    /// introduced as helper types to place these values somewhere.
    fn as_qref(&self) -> QRef<K, Q> {
        match self {
            &KeyCompare::Owned(ref owned) => QRef::Shared(owned),
            &KeyCompare::Shared(ref not_null) => QRef::Shared(not_null),
            &KeyCompare::OnlyCompare(q) => QRef::Borrow(q),
        }
    }
}

/// See `KeyCompare::as_qref()` for the motivation behind this type.
enum QRef<'k, 'q, K: 'k + Borrow<Q>, Q: 'q + ?Sized> {
    Shared(&'k KeySlot<K>),
    Borrow(&'q Q),
}

impl<'k, 'q, K: Borrow<Q>, Q: ?Sized> QRef<'k, 'q, K, Q> {
    fn as_qref2(&self) -> QRef2<K, Q> {
        match self {
            &QRef::Shared(&KeySlot::Key(ref k)) => QRef2::Shared(k),
            &QRef::Shared(&KeySlot::SeeNewTable) =>
                unreachable!("KeyCompare must contain a `NotNull(KeySlot::Key(K))`"),
            &QRef::Borrow(q) => QRef2::Borrow(q),
        }
    }
}

/// See `KeyCompare::as_qref()` for the motivation behind this type.
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

/// This enum represents the value to insert when calling `put_if_match()`, which is usually owned
/// when called from `LockFreeHashMap` and shared if its been copied from a previous, smaller map.
#[derive(Debug)]
pub enum PutValue<'v, V: 'v> {
    Owned(NotNullOwned<ValueSlot<'v, V>>),
    Shared(NotNull<'v, ValueSlot<'v, V>>),
}

impl<'v, V: PartialEq> PutValue<'v, V> {
    pub fn new(value: V) -> Self {
        PutValue::Owned(NotNullOwned::new(ValueSlot::Value(value)))
    }
    /// Returns a new `PutValue` containing an owned `ValueSlot::Tombstone` value.
    pub fn new_tombstone() -> Self {
        PutValue::Owned(NotNullOwned::new(ValueSlot::Tombstone))
    }
    /// Returns true if and only if the inner `ValueSlot` has discriminant `Tombstone`.
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

    pub fn ptr_equals(&self, value: NotNull<ValueSlot<V>>) -> bool {
        if (&*value as *const _) == self.as_raw() {
            true
        } else {
            false
        }
    }

    pub fn as_raw(&self) -> *const ValueSlot<'v, V> {
        match self {
            &PutValue::Owned(ref not_null) => &**not_null as *const _,
            &PutValue::Shared(ref not_null) => &**not_null as *const _,
        }
    }
}

pub type KVPair<'v, K, V> = (AtomicPtr<KeySlot<K>>, AtomicPtr<ValueSlot<'v, V>>);

/// A map containing a unique, non-resizable array to the Key/Value pairs. If the map needs to be
/// resized, a new `MapInner` must be created and its Key/Value pairs must be copied from this one.
/// Logically, this struct owns its keys and values, and so is responsible for freeing them when
/// dropped.
pub struct MapInner<'v, K, V: 'v, S = RandomState> {
    /// The key/value pairs in this map, allocated as an array of pairs.
    map: Vec<KVPair<'v,K,V>>,
    /// The amount of key/value pairs in the array, if any.
    size: AtomicUsize,
    /// Points to the newer map or null if none.
    newer_map: AtomicPtr<MapInner<'v,K,V,S>>,
    /// Any thread can allocate memory to resize the map and create `newer_map`. Thus, we want to
    /// try and limit the amount of allocations done. This is a monotonically increasing count of
    /// the number of threads currently trying to allocate a new map, which is used as a heuristic.
    /// See its use in the `MapInner::create_newer_map()` function.
    resizers_count: AtomicUsize,
    /// The number of `::COPY_CHUNK_SIZE = 32` element chunks that some thread has commited to
    /// copying to the newer table. Once this reaches `capacity/COPY_CHUNK_SIZE`, we know that the
    /// entire map has been copied into the large `newer_map`.
    chunks_copied: AtomicUsize,
    /// The actual number of key/value pairs that have been copied into the newer map.
    slots_copied: AtomicUsize,
    /// The hasher used to hash keys.
    hash_builder: S,
}


impl<'v, K, V, S> MapInner<'v, K, V, S> {
    /// The default size of a new `LockFreeHashMap`.
    pub const DEFAULT_CAPACITY: usize = ::LockFreeHashMap::<(), (), RandomState>::DEFAULT_CAPACITY;

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

    /// Drops `self.newer_map` and any newer maps that `self.newer_map` points to.
    pub unsafe fn drop_newer_maps(&self, guard: &Guard) {
        if let Some(newer_map) = self.newer_map.take(guard) {
            newer_map.drop_self_and_newer_maps(guard);
        }
    }

    /// Drops self, `self.newer_map` and any newer maps that `self.newer_map` points to.
    pub unsafe fn drop_self_and_newer_maps(self, guard: &Guard) {
        let newer_map = self.newer_map.take(guard);
        drop(self);
        if let Some(newer_map) = newer_map {
            newer_map.drop_self_and_newer_maps(guard);
        }
    }
}

impl<'v, K: Hash + Eq, V: PartialEq> MapInner<'v,K,V,RandomState> {
    /// Creates a new `MapInner`. Uses the next power of two if size is not a power of two.
    pub fn with_capacity(size: usize) -> Self {
        MapInner::with_capacity_and_hasher(size, RandomState::new())
    }
}

impl<'guard, 'v: 'guard, K, V, S> MapInner<'v, K,V,S>
    where K: Hash + Eq,
          V: PartialEq,
          S: BuildHasher + Clone,
{
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
            slots_copied: AtomicUsize::new(0),
            hash_builder: hasher,
        }
    }

    /// Help copy a small chunk of the map to the `newer_map`. See `::COPY_CHUNK_SIZE` for the
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
            if lower_bound >= upper_bound {
                // But before we do, we should decrement `chunks_copied` by 1, just to make sure it
                // won't overflow. It can still overflow if you have `usize::MAX` amount of threads
                // calling `help_copy`, but it will be assumed that this never happens.
                self.chunks_copied.fetch_sub(1, Ordering::SeqCst);
                // In the rare event that the `newer_map` has finished copying all its elements
                // into an even `newer_map`, it can call `promote()` and fail, because it's not the
                // current map. Thus, we call it again here, even if some other thread was
                // "supposed" to have called it.
                self.try_promote(newer_map, 0, outer_map, guard);
                return;
            }
            let mut slots_copied = 0;
            // Now because `lower_bound` must be less than `upper_bound`, and since we already
            // assumed that any thread that gets some `chunks_copied` MUST then copy all elements
            // in that chunk, we MUST do so.  Notice that the `..upper_bound` is exclusive, so it
            // never exceeds (self.capacity() - 1).
            for i in lower_bound..upper_bound {
                // Now simply copy_slot() for each element in the chunk of the array that we're
                // assigned.
                if self.copy_slot(&*newer_map, i, outer_map, guard) {
                    slots_copied += 1;
                }
            }
            self.try_promote(newer_map, slots_copied, outer_map, guard);
            if upper_bound == self.capacity() {
                return;
            } else if !copy_everything {
                // Otherwise, we did not copy everything and there is still more to be done,
                // or at least there was more when we last checked `chunks_copied`. If we are not
                // required to copy everything, then just return and let some other thread do it.
                return;
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
        // This appears to be that some other thread already promoted us, or the rare event in
        // which we called `promote()` before the previous map called `promote()`.
        // Just return here.
        if current_map as *const _ != self as *const _ {
            return false;
        }
        match outer_map.compare_and_set_shared(current_map_shared, new_map, guard) {
            Ok(_) => {
                // We successfully swapped the value of the `AtomicBox` and are therefore
                // responsible for freeing the old map's memory.
                unsafe { guard.defer(move || current_map_shared.as_shared().into_owned()); }
                return true;
            },
            Err((_current, _)) => {
                debug_assert!(&*_current as *const _ != self as *const _);
                // We know that `current_map` was `&self` at some point previously, so some other
                // thread successfully promoted the map.
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
            if self.copy_slot(&*new_map, copy_index, outer_map, guard) {
                self.slots_copied.fetch_add(1, Ordering::SeqCst);
            }
            self.help_copy(new_map, false, outer_map, guard);
            new_map
        } else {
            unreachable!("can't call ensure_slot_copied() unless found a prime value");
        }
    }

    pub fn try_promote(
        &self,
        new_map: NotNull<Self>,
        current_slots_copied: usize,
        outer_map: &AtomicBox<Self>,
        guard: &'guard Guard
    ) -> bool
    {
        let previous_slots_copied = self.slots_copied.fetch_add(current_slots_copied, Ordering::SeqCst);
        if current_slots_copied > 0 {
        debug_assert!(previous_slots_copied + current_slots_copied <= self.capacity(),
            format!("previous: {} current: {}", previous_slots_copied, current_slots_copied)
        );
        }
        if previous_slots_copied + current_slots_copied == self.capacity() {
            self.promote(new_map, outer_map, guard)
        } else {
            false
        }
    }

    /// Copies a single key/value pair from the map in `&self` to the map in `&self.newer_map`.
    ///
    /// Returns whether or not this thread was the one to copy the slot. Since copying takes
    /// several transitions that could happen from any thread, it doesn't matter which transition
    /// we pick as long as exactly one thread reports true for copying a specific slot.
    pub fn copy_slot(
        &self,
        new_map: &Self,
        old_map_index: usize,
        outer_map: &AtomicBox<Self>,
        guard: &'guard Guard
    ) -> bool
    {
        /// This is necessary because we're copying a value slot from an older map into a newer
        /// map. Thus, when we call `AtomicPtr::load(_, guard)`, we get a pointer that is only
        /// valid for the guard's lifetime. But it needs to be inserted into the newer_map and
        /// therefore must be valid for the lifetime in newer map.
        /// FIXME: Is this necessary? It seems like you would need recursive lifetimes to express
        ///        that `newer_map` has a different lifetime to `self`.
        fn cheat_lifetime<'guard, 'v, V>(maybe: MaybeNull<'guard, V>) -> MaybeNull<'v, V> {
            MaybeNull::from_shared(Shared::from(maybe.as_shared().as_raw()))
        }
        let (ref atomic_key_slot, ref atomic_value_slot) = self.map[old_map_index];
        let old_key: NotNull<_>;
        let mut new_key = NotNullOwned::new(KeySlot::SeeNewTable);

        // Preemptively set an empty key slot to `SeeNewTable`.
        loop {
            let cas_key_result = atomic_key_slot.compare_and_set_owned_weak(
                MaybeNull::from_shared(Shared::null()), new_key, guard);
            match cas_key_result {
                Ok(_new_key) => {
                    debug_assert!(if let &KeySlot::SeeNewTable = &*_new_key {true} else {false});
                    return true;
                },
                Err((current, new)) => {
                    new_key = new; // Return ownership
                    let _old_key_shared = current;
                    // Because `compare_and_set_weak()` can spuriously fail and therefore still be
                    // null. Thus, just retry with `continue` if it's still null.
                    match current.as_option() {
                        // No one updated the key slot from `empty` to something else, due to using
                        // a weak version of CAS here. Thus, we can try again.
                        None => continue,
                        Some(k) => {
                            match k.deref() {
                                &KeySlot::SeeNewTable => return false,
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

        // If we got to this point, then we know that there is an existing, non-null key. Thus, we
        // need to do the following state transitions:
        //
        //         |     [1]           |         [2]               |        [3]
        // --------+-------------------+---------------------------+-------------------
        // Old Map | (K, V) -> (K, V') |                           | (K, V') -> (K, X)
        // --------+-------------------+---------------------------+-------------------
        // New Map |                   | (null, null) -> (K, null) |
        //         |                   | (K, null) -> (K, V)       |
        //
        // However, if, in transition [1]:
        //      1) V is null here, then just set it to X and let the thread that's trying to
        //         copy in its (K,V) pair insert it into the newer map.
        //      2) V is a tombstone (T) here, then just set it to X and don't copy a key
        //         without a value into the newer map.
        // Note that also both operations in transition [2] can happen on two different threads.
        // In addition, care needs to be taken for the rest of this function to ensure that we
        // (defer) drop destructors that we need to, but only once.
        let mut old_value: MaybeNull<_> = cheat_lifetime(atomic_value_slot.load(guard));
        let not_null_old_value: NotNull<_>;
        let primed_old_value: NotNull<ValueSlot<_>>;
        let mut original_valueslot_value = None;

        loop {
            match old_value.as_option() {
                // Swap `None`/`Null` values with `SeeNewTable`.
                None => {
                    match atomic_value_slot.compare_and_set_owned(
                        MaybeNull::from_shared(Shared::null()),
                        NotNullOwned::new(ValueSlot::SeeNewTable),
                        guard,
                    ) {
                        Err((current, _)) => {
                            debug_assert!(current.as_option().is_some());
                            old_value = cheat_lifetime(current);
                            continue;
                        },
                        Ok(current) => {
                            // Successfully did (K,null) -> (K, X). Thus there's nothing more to do
                            // here, and we obviously don't need to free the null pointer.
                            debug_assert!(current.deref().is_seenewtable());
                            return true;
                        },
                    }
                },
                // Otherwise we have a `ValueSlot` here. Let's take a little peek inside.
                Some(not_null) => match not_null.deref() {
                    // Some other thread copied the slot already. Nothing to do or free here.
                    &ValueSlot::SeeNewTable  => return false,
                    &ValueSlot::Tombstone => {
                        match atomic_value_slot.compare_and_set_owned(
                            old_value,
                            NotNullOwned::new(ValueSlot::SeeNewTable),
                            guard,
                        ) {
                            Err((current, _)) => {
                                // Assert that `Tombstone` can't turn into `Null`. But it can still
                                // be V/V'/T/X
                                debug_assert!(current.as_option().is_some());
                                old_value = cheat_lifetime(current);
                                continue;
                            },
                            Ok(_new) => {
                                // Successfully did (K, T) -> (K, X). Remember that `old_value`
                                // here is just the atomic pointer to the tombstone. However, all
                                // `ValueSlot`s are behind pointers and therefore need to be freed.
                                debug_assert!(_new.is_seenewtable());
                                unsafe { guard.defer(move || not_null.drop()); }
                                return true;
                            }
                        }
                    },
                    // There's a value here. So (K, V) -> (K, V') needs to happen.
                    &ValueSlot::Value(_) => {
                        // old_value was `Value` and not `ValuePrime`.
                        let primed_old_value_owned = NotNullOwned::new(ValueSlot::ValuePrime(not_null.deref()));
                        match atomic_value_slot.compare_and_set_owned(
                            old_value, // `ValueSlot::Value(_)`
                            primed_old_value_owned,
                            guard
                        ) {
                            Err((current, _dropped_because_owned)) => {
                                debug_assert!(current.as_option().is_some());
                                old_value = cheat_lifetime(current);
                                continue;
                            },
                            Ok(shared_primed_value) => {
                                // We are the ones that successfully did (K, V) -> (K, V').
                                // We are the only ones who performed this exact transition, so
                                // we will be the ones who will free V if V has not been
                                // successfully inserted into the newer map. We are the only one's
                                // who will do this so that we can avoid a double free. So let's
                                // store the `ValueSlot` that we need to free in this variable.
                                original_valueslot_value = Some(not_null);
                                debug_assert!(shared_primed_value.is_valueprime());
                                not_null_old_value = not_null;
                                primed_old_value = shared_primed_value;
                                break;
                            },
                        }
                    }
                    &ValueSlot::ValuePrime(_)  => {
                        not_null_old_value = not_null;
                        primed_old_value = not_null;
                        break;
                    }
                }
            }
        }

        // If we have gotten this far, then we know that at least the first transition has
        // occurred, i.e. (K, V) -> (K, V').
        // Also, `not_null_old_value` and `primed_old_value` are assigned, with
        // `not_null_old_value` being either `Value` or `ValuePrime`.
        debug_assert!(not_null_old_value.is_value() || not_null_old_value.is_valueprime());
        debug_assert!(primed_old_value.is_valueprime());

        // Now, we know that `old_value` must be either V or V', depending on whether
        // `not_null` was a `ValueSlot::Value(_)` or `ValueSlot::ValuePrime(_)`.
        let put_value = match not_null_old_value.deref() {
            &ValueSlot::Value(_) => PutValue::Shared(not_null_old_value),
            &ValueSlot::ValuePrime(v) => match v {
                &ValueSlot::Value(_) =>
                    PutValue::Shared(MaybeNull::from_shared(Shared::from(v as *const _))
                        .as_option().expect("v is a reference and can't be null")
                    ),
                _ => unreachable!("`ValuePrime` can only be a reference to a `Value`"),
            }
            &ValueSlot::Tombstone => unreachable!(),
            &ValueSlot::SeeNewTable => unreachable!(),
        };
        // Now we try to copy the original value into the newer map, but only if there is
        // no value in there already. If this fails, then it was copied and/or updated in
        // the newer map.
        //
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
            debug_assert!(atomic_key_slot.is_tagged(guard));
        }

        // Now we simply need to just do (K, V') -> (K, X).
        let primed_old_value_maybe: MaybeNull<_> = primed_old_value.as_maybe_null();
        match atomic_value_slot.compare_and_set_owned(
            primed_old_value_maybe, NotNullOwned::new(ValueSlot::SeeNewTable), guard
        ) {
            Ok(_current) => {
                debug_assert!(_current.is_seenewtable());
                unsafe { primed_old_value_maybe.try_defer_drop(guard); }
            },
            Err((current, _)) => {
                debug_assert!(current.as_option()
                    .map(|v| v.is_seenewtable())
                    .unwrap_or(false),
                    "can't be null again"
                );
            },
        }
        // This is only `Some` if we are the thread that did (K, V) -> (K, V').
        if let Some(original_value) = original_valueslot_value {
            unsafe { guard.defer(move || {
                // We only want to drop this value if it was never copied to the new map.
                if !atomic_key_slot.is_tagged(guard) {
                    original_value.drop();
                }
            })}
        }
        return copied_into_new;
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
        let size = self.capacity();
        let mut new_size = size;
        // Double size if map is >25% full
        if size > (self.capacity() >> 2) {
            new_size = try_double(new_size);
            // Double size if map is >50% full
            if size > (self.capacity() >> 1) {
                new_size = try_double(new_size);
            }
        }
        let array_element_byte_size: usize = ::std::mem::size_of::<KVPair<K,V>>();
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
        debug_assert!(new_size >= self.capacity());
        match self.newer_map.compare_null_and_set_owned(
            NotNullOwned::new(Self::with_capacity_and_hasher(new_size, self.hash_builder.clone())),
            guard
        ) {
            Ok(shared_newer_map) => {
                shared_newer_map
            },
            Err((current, _drop_our_map)) => {
                debug_assert!((&*current as *const _) != (self as *const _));
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

    /// Returns the current value associated with some key, if any.
    pub fn get<Q: ?Sized>(
        &self,
        key: &Q,
        outer_map: &AtomicBox<Self>,
        guard: &'guard Guard
    ) -> Option<&'guard V>
        where K: Borrow<Q>,
              Q: Hash + Eq + PartialEq<K>,
    {
        // First we need to find/probe the index of the key.
        let initial_index = self.hash_key(key);
        let len = self.capacity();
        for index in (initial_index..len).chain(0..initial_index) {
            let (ref atomic_key_slot, ref atomic_value_slot) = self.map[index];
            // Early exit if neither the key nor value are fully inserted.
            if !atomic_key_slot.relaxed_exists(&guard) || !atomic_value_slot.relaxed_exists(&guard)
            {
                return None;
            }
            match &*atomic_key_slot.load(&guard).as_option()? {
                &KeySlot::Key(ref k) => if self.keys_are_equal(k, key) {
                    match atomic_value_slot.load(&guard).as_option()?.deref() {
                        &ValueSlot::Value(ref v) => return Some(&v),
                        &ValueSlot::Tombstone => return None,
                        // We call ensure_slot_copied() even on `SeeNewTable` because it calls
                        // try_promote().
                        &ValueSlot::ValuePrime(_) | &ValueSlot::SeeNewTable => {
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
                        // It is safe to `unwrap()` because a newer table must exist before any
                        // `KeySlot`s are set to `SeeNewTable`.
                        .expect("Can't set `KeySlot` to `SeeNewTable` before setting `newer_map`")
                        .get(key, outer_map, guard);
                },
            }
        }
        // We exhausted the entire map, so the value could still be inserted into the newer map
        return self.newer_map.load(&guard)
            .as_option()
            .map(|newer_map| newer_map.get(key, outer_map, guard))
            .unwrap_or(None)
    }

    /// Increments or decrements the current size of the map, returning the previous value in the
    /// map.
    pub fn update_size_and_defer(
        &'guard self,
        old_value_slot: MaybeNull<'guard, ValueSlot<V>>,
        insert_tombstone: bool,
        guard: &'guard Guard,
    ) -> Option<&'guard ValueSlot<V>>
    {
        // If we did not insert a tombstone, then we incremented if the old value was null or
        // tombstone.
        let increment = if !insert_tombstone {
            match old_value_slot.as_option() {
                None => true,
                Some(ref old_value) if old_value.is_tombstone() => true,
                Some(ref old_value) if old_value.is_seenewtable() => unreachable!(),
                Some(ref old_value) if old_value.is_valueprime() => unreachable!(),
                Some(_) => false,
            }
        } else {
            false
        };
        if increment {
            self.size.fetch_add(1, Ordering::SeqCst);
        }
        // If we did insert a tombstone, then we decremented if the old value was V
        let decrement = if insert_tombstone {
            match old_value_slot.as_option() {
                None => false,
                Some(ref old_value) if old_value.is_tombstone() => false,
                Some(ref old_value) if old_value.is_seenewtable() => unreachable!(),
                Some(ref old_value) if old_value.is_valueprime() => unreachable!(),
                Some(_) => true,
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
        mut put: PutValue<'v, V>,
        matcher: Match,
        outer_map: &AtomicBox<Self>,
        guard: &'guard Guard
    ) -> Option<&'guard ValueSlot<V>>
        where K: Borrow<Q>,
              Q: Hash + Eq + PartialEq<K> + ?Sized,
    {
        /// FIXME: See other cheat_lifetime() FIXME note above
        fn cheat_lifetime<'guard, 'v, V>(maybe: NotNull<'guard, V>) -> NotNull<'v, V> {
            MaybeNull::from_shared(Shared::from(maybe.as_shared().as_raw()))
                .as_option()
                .expect("parameter was `NotNull` to begin with")
        }
        let initial_index = self.hash_key(key.as_qref().as_qref2().as_q());
        let len = self.capacity();
        let mut key_index = None;
        let mut key = key;
        // First we need to find the key slot for the key.
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
                } else if let Match::AnyKeyValuePair = matcher {
                    // If key is not taken, return None if we weren't going to insert something
                    // anyway
                    return None;
                } else {
                    match key {
                        KeyCompare::Owned(owned) => {
                            match atomic_key_slot.compare_null_and_set_owned(owned, guard) {
                                Ok(shared_key) => {
                                    // TODO: Raise keyslots-used count
                                    key = KeyCompare::Shared(shared_key);
                                    key_index = Some(index);
                                    break 'find_key_loop;
                                },
                                Err((not_null, _return)) => {
                                    key = KeyCompare::Owned(_return);
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
                            // We are only comparing the keys and don't want to insert it if there
                            // is no key slot taken.
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
                // We have exhausted the entire probing range, so there are no key slots available
                // and need to resize.
                let new_table: NotNull<Self> = self.create_newer_map(guard);
                self.help_copy(new_table, true, outer_map, guard);
                return new_table.deref().put_if_match(key, put, matcher, outer_map, guard);
            },
        };

        // We have now found the key slot to use. This key slot will never change now so we know
        // that we may insert the value into the index `key_index`.

        let atomic_value_slot = &self.map[key_index].1;
        let mut old_value_slot: MaybeNull<_> = atomic_value_slot.load(&guard);

        // Now try to put the value into the map.
        let insert_tombstone = put.is_tombstone();
        loop {
            let value_slot_option = old_value_slot.as_option();
            // If the value we're trying to insert equals the current value, pretend we replaced it
            // with CAS and just return the current value.
            if let Some(v) = value_slot_option {
                if put.ptr_equals(v) {
                    return Some(v.deref());
                }
            }
            // Early return if the expected value in `matcher` doesn't equal the current value.
            match matcher {
                Match::Empty => if let Some(v) = value_slot_option {
                    return Some(v.deref())
                },
                Match::AnyKeyValuePair => match value_slot_option.map(|v| v.deref()) {
                    Some(&ValueSlot::Tombstone) | None => return None,
                    _ => (),
                }
                Match::Always => (),
            }
            // If it's prime then we need to copy the slot and try again in the new map.
            if value_slot_option.map_or(false, |v| v.is_prime()) {
                let newer_map = self.newer_map
                    .load(&guard)
                    .as_option()
                    .expect("Can't set a `ValueSlot` to `ValuePrime` before setting `newer_map`");
                self.copy_slot(&*newer_map, key_index, outer_map, guard);
                return newer_map.deref().put_if_match(key, put, matcher, outer_map, guard);
            }
            // If the new map exists, help copy the current slot and some others and try again.
            if self.newer_map.relaxed_exists(guard) {
                // TODO: if newer_map == None AND ((current_value is None AND table full) OR value
                // is prime) then resize
                return self.ensure_slot_copied(key_index, outer_map, guard)
                    .deref()
                    .put_if_match(key, put, matcher, outer_map, guard);
            }
            debug_assert!(value_slot_option.map_or(true, |v| !v.is_prime()));
            // Otherwise, try to CAS the value.
            match put {
                PutValue::Owned(owned) => match atomic_value_slot.compare_and_set_owned(
                    old_value_slot, owned, &guard
                ) {
                    Ok(_) => {
                        return self.update_size_and_defer(old_value_slot, insert_tombstone, guard);
                    },
                    Err((current, _return_ownership)) => {
                        debug_assert!(current.as_option().is_some());
                        old_value_slot = current;
                        put = PutValue::Owned(_return_ownership);
                    },
                },
                PutValue::Shared(shared) => match atomic_value_slot.compare_and_set(
                    old_value_slot, shared, &guard
                ) {
                    Ok(_) => {
                        return self.update_size_and_defer(old_value_slot, insert_tombstone, guard);
                    },
                    Err((current, _return_ownership)) => {
                        debug_assert!(current.as_option().is_some());
                        old_value_slot = current;
                        put = PutValue::Shared(cheat_lifetime(_return_ownership));
                    },
                },
            }
        }
    }

    pub fn clone_hasher(&self) -> S {
        self.hash_builder.clone()
    }
}

impl<'v, K, V, S> Drop for MapInner<'v, K, V, S> {
    fn drop(&mut self) {
        let guard = &::pin();
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

impl<'v, K: fmt::Debug, V: fmt::Debug, S> fmt::Debug for MapInner<'v, K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let guard = &::pin();
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

