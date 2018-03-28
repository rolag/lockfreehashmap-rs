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

//! LockFreeHashMap
//!
//! This is an implementation of the lock-free hash map created by Dr. Cliff Click.
//!
//! Originally, this implementation
//! [here](https://github.com/boundary/high-scale-lib/blob/master/src/main/java/org/cliffc/high_scale_lib/NonBlockingHashMap.java)
//! and
//! [recently here](https://github.com/JCTools/JCTools/blob/master/jctools-core/src/main/java/org/jctools/maps/NonBlockingHashMap.java#L770-L770)
//! was created for Java, using garbage collection where necessary.
//! This library is a Rust implementation, using epoch-based memory management to compensate for
//! the lack of garbage collection.
//! The `crossbeam` crate is used for epoch-based memory management.
//!
//! For details on the hash map's design and implementation, see the (private) [map_inner] module.
//!
//! At the time of writing, other concurrent hash maps available don't appear to allow reading and
//! writing at the same time. This map does.
//! Effectively, this map has the same guarantees as having a certain amount of global variables
//! that can be changed atomically.


// NOTE: To use valgrind, uncomment the lines below and recompile.
// NOTE: Then `valgrind --leak-check=full --show-leak-kinds=all ./target/debug/deps/lockfreehashmap-***`
//#![feature(alloc_system, global_allocator, allocator_api)]
//extern crate alloc_system;
//use alloc_system::System;
//#[global_allocator]
//static A: System = System;

extern crate crossbeam_epoch;
extern crate crossbeam_utils as crossbeam;

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::Hash;

mod atomic;
mod map_inner;

/// Re-export `crossbeam::epoch::pin()` and its return type for convenience.
pub use crossbeam_epoch::{pin, Guard};
/// Re-export `crossbeam::scope()` and its return type for convenience.
pub use crossbeam::scoped::{scope, Scope};

use atomic::AtomicBox;
use map_inner::{KeyCompare, MapInner, Match, PutValue, ValueSlot};

pub const COPY_CHUNK_SIZE: usize = 32;

pub struct LockFreeHashMap<'v, K, V: 'v, S = RandomState> {
    /// Points to the newest map (after it's been fully resized). Always non-null.
    inner: AtomicBox<MapInner<'v,K,V,S>>,
}

impl<'v, K, V, S> LockFreeHashMap<'v,K,V,S>
    where K: Hash + Eq,
          V: PartialEq,
          S: ::std::hash::BuildHasher + Clone,
{
    /// The default size of a new `LockFreeHashMap` when created by `LockFreeHashMap::new()`.
    pub const DEFAULT_CAPACITY: usize = 8;

    /// Creates an empty `LockFreeHashMap` with the specified capacity, using `hasher` to hash the
    /// keys.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash map will use the next power of 2 (i.e. 1).
    ///
    /// # Examples
    ///
    /// ```
    /// use lockfreehashmap::LockFreeHashMap;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let mut map = LockFreeHashMap::with_capacity_and_hasher(10, s);
    /// let guard = lockfreehashmap::pin();
    /// map.insert(1, 2, &guard);
    /// ```
    pub fn with_capacity_and_hasher(capacity: usize, hasher: S) -> Self {
        LockFreeHashMap {
            inner: AtomicBox::new(MapInner::with_capacity_and_hasher(capacity, hasher))
        }
    }
}

impl<'guard, 'v: 'guard, K: Hash + Eq + 'guard, V: PartialEq> LockFreeHashMap<'v,K,V> {

    /// Creates a new `LockFreeHashMap`.
    ///
    /// # Examples
    /// ```
    /// # #![allow(unused_variables)]
    /// # use lockfreehashmap::LockFreeHashMap;
    /// let map = LockFreeHashMap::<u32, String>::new();
    /// ```
    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    /// Creates a new `LockFreeHashMap` of a given size. Uses the next power of two if size is not
    /// a power of two.
    ///
    /// # Examples
    /// ```
    /// # use lockfreehashmap::LockFreeHashMap;
    /// let map = LockFreeHashMap::<u32, String>::with_capacity(12);
    /// assert_eq!(map.capacity(), 12usize.next_power_of_two());
    /// assert_eq!(map.capacity(), 16);
    /// ```
    pub fn with_capacity(size: usize) -> Self {
        LockFreeHashMap { inner: AtomicBox::new(MapInner::with_capacity(size)) }
    }

    /// Returns the number of elements the map can hold without reallocating.
    ///
    /// # Examples
    /// ```
    /// # use lockfreehashmap::LockFreeHashMap;
    /// let map = LockFreeHashMap::<u32, String>::with_capacity(8);
    /// assert_eq!(map.capacity(), 8);
    /// ```
    pub fn capacity(&self) -> usize {
        let guard = pin();
        self.load_inner(&guard).capacity()
    }

    /// Returns the number of elements in the map.
    ///
    /// # Examples
    /// ```
    /// # use lockfreehashmap::*;
    /// let map = LockFreeHashMap::<u32, String>::with_capacity(8);
    /// assert_eq!(map.capacity(), 8);
    /// assert_eq!(map.len(), 0);
    /// let guard = lockfreehashmap::pin();
    /// map.insert(5, String::from("five"), &guard);
    /// assert_eq!(map.capacity(), 8);
    /// assert_eq!(map.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        let guard = pin();
        self.load_inner(&guard).len()
    }

    /// Clears the map, removing all key-value pairs. Keeps the allocated memory for reuse.
    pub fn clear(&self) {
        unimplemented!()
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but Hash and Eq on the borrowed
    /// form must match those for the key type.
    ///
    /// # Examples
    /// ```
    /// # use lockfreehashmap::*;
    /// let map = LockFreeHashMap::<i32, i32>::new();
    /// assert!(!map.contains_key(&3));
    /// let guard = lockfreehashmap::pin();
    /// map.insert(3, 8934, &guard);
    /// assert!(map.contains_key(&3));
    /// map.remove(&3, &guard);
    /// assert!(!map.contains_key(&3));
    /// ```
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
        where K: Borrow<Q>,
              Q: Hash + Eq + PartialEq<K>,
    {
        let guard = pin();
        self.get(key, &guard).is_some()
    }

    /// Private helper method to load the `inner` field as a &[MapInner].
    fn load_inner(&self, guard: &'guard Guard) -> &'guard MapInner<'v,K,V> {
        self.inner.load(&guard).deref()
    }

    /// Returns a reference to the value corresponding to the key. The key may be any borrowed
    /// form of the map's key type, but Hash and Eq on the borrowed form must match those for the
    /// key type.
    ///
    /// # Examples
    /// ```
    /// # use lockfreehashmap::*;
    /// let map = LockFreeHashMap::<i32, i32>::new();
    /// let guard = lockfreehashmap::pin();
    /// assert_eq!(map.get(&1, &guard), None);
    /// map.insert(1, 15, &guard);
    /// assert_eq!(map.get(&1, &guard), Some(&15));
    /// ```
    pub fn get<Q: ?Sized>(&self, key: &Q, guard: &'guard Guard) -> Option<&'guard V>
        where K: Borrow<Q>,
              Q: Hash + Eq + PartialEq<K>,
    {
        return self.load_inner(guard).get(key, &self.inner, guard);
    }

    /// Inserts a key-value pair into the map. If the map did not have this key present, None is
    /// returned. If the map did have this key present, the value is updated, and the old value is
    /// returned. The key is not updated, though; this matters for types that can be `==` without
    /// being identical.
    ///
    /// # Examples
    /// ```
    /// # use lockfreehashmap::*;
    /// let map = LockFreeHashMap::<String, String>::new();
    /// let guard = lockfreehashmap::pin();
    /// let key = "key".to_string();
    /// let equal_key = "key".to_string();
    /// assert_eq!(key, equal_key); // The keys are equal
    /// assert_ne!(&key as *const _, &equal_key as *const _); // But not identical
    /// assert_eq!(map.insert(key, "value".to_string(), &guard), None);
    /// assert_eq!(map.insert(equal_key, "other".to_string(), &guard), Some(&"value".to_string()));
    /// // `map` now contains `key` as its key, rather than `equal_key`.
    /// ```
    pub fn insert(&self, key: K, value: V, guard: &'guard Guard) -> Option<&'guard V>
    {
        let value_slot: Option<&ValueSlot<V>> = self.load_inner(guard).put_if_match(
            KeyCompare::new(key),
            PutValue::new(value),
            Match::Always,
            &self.inner,
            &guard
        );
        return ValueSlot::as_inner(value_slot);
    }

    /// Inserts a key-value pair into the map, but only if there is already an existing value that
    /// corresponds to the key in the map. If the map did not have this key present, None is
    /// returned. If the map did have this key present, the value is updated, and the old value is
    /// returned. The key is not updated, though; this matters for types that can be `==` without
    /// being identical.
    ///
    /// # Examples
    /// ```
    /// # use lockfreehashmap::*;
    /// let map = LockFreeHashMap::<i32, i32>::new();
    /// let guard = lockfreehashmap::pin();
    /// assert_eq!(map.replace(&1, 1, &guard), None);
    /// assert_eq!(map.replace(&1, 1, &guard), None);
    /// assert_eq!(map.insert(1, 1, &guard), None);
    /// assert_eq!(map.replace(&1, 3, &guard), Some(&1));
    /// ```
    pub fn replace<Q: ?Sized>(&self, key: &Q, value: V, guard: &'guard Guard) -> Option<&'guard V>
        where K: Borrow<Q>,
              Q: Hash + Eq + PartialEq<K>,
    {
        let value_slot: Option<&ValueSlot<V>> = self.load_inner(guard).put_if_match(
            KeyCompare::OnlyCompare(key),
            PutValue::new(value),
            Match::AnyKeyValuePair,
            &self.inner,
            &guard
        );
        return ValueSlot::as_inner(value_slot);
    }

    /// Removes a key from the map, returning the value at the key if the key was previously in the
    /// map. The key may be any borrowed form of the map's key type, but Hash and Eq on the
    /// borrowed form must match those for the key type.
    ///
    /// # Examples
    /// ```
    /// # use lockfreehashmap::*;
    /// let map = LockFreeHashMap::<i32, i32>::new();
    /// let guard = lockfreehashmap::pin();
    /// assert_eq!(map.remove(&1, &guard), None);
    /// map.insert(1, 1, &guard);
    /// assert_eq!(map.remove(&1, &guard), Some(&1));
    /// ```
    pub fn remove<Q: ?Sized>(&self, key: &Q, guard: &'guard Guard) -> Option<&'guard V>
        where K: Borrow<Q>,
              Q: Hash + Eq + PartialEq<K>,
    {
        let value_slot: Option<&ValueSlot<V>> = self.load_inner(guard).put_if_match(
            KeyCompare::OnlyCompare(key),
            PutValue::new_tombstone(),
            Match::Always,
            &self.inner,
            &guard
        );
        return ValueSlot::as_inner(value_slot);
    }
}


impl<'v, K, V, S> Drop for LockFreeHashMap<'v, K, V, S> {
    fn drop(&mut self) {
        let guard = pin();
        // self.inner will be dropped because Drop is implemented on `AtomicBox`
        // But if self.inner has pointers to newer maps, then those need to be explicitely dropped.
        unsafe {
            self.inner.load(&guard).deref().drop_newer_maps(&guard);
        }
    }
}

impl<'guard, 'v: 'guard, K: Hash + Eq + fmt::Debug, V: fmt::Debug + PartialEq>
    fmt::Debug for LockFreeHashMap<'v,K,V>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let guard = pin();
        write!(f, "LockFreeHashMap {{ {:?} }}", self.load_inner(&guard))
    }
}

#[cfg(test)]
mod test {
    extern crate rand;
    use super::*;
    use self::rand::Rng;

    #[test]
    fn test_basic() {
        let map = LockFreeHashMap::<u8, u8>::new();
        let test_guard = pin();
        for i in 1..4 {
            map.insert(i, i, &test_guard);
        }
        let map = &map;
        scope(|scope| {
            scope.spawn(|| {
                let test_guard = pin();
                assert_eq!(map.get(&1, &test_guard), Some(&1));
            });
            scope.spawn(|| {
                let test_guard = pin();
                assert_eq!(map.insert(100, 101, &test_guard), None);
            });
            scope.spawn(|| {
                let test_guard = pin();
                assert_eq!(map.insert(5, 4, &test_guard), None);
            });
            scope.spawn(|| {
                let test_guard = pin();
                assert_eq!(map.get(&4, &test_guard), None);
                assert_eq!(map.insert(3, 4, &test_guard), Some(&3));
                assert_eq!(map.get(&3, &test_guard), Some(&4));
            });
        });
    }

    #[test]
    fn test_single_thread() {
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 31, 32, 33].iter() {
            test_single_thread_insert(*i);
        }
        test_single_thread_insert(256);
    }

    fn test_single_thread_insert(size: usize) {
        let map = &LockFreeHashMap::<usize, String>::new();
        let test_guard = pin();
        for i in 0..size {
            map.insert(i, i.to_string(), &test_guard);
            assert_eq!(i + 1, map.len());
            for j in 0..(i+1) {
                assert_eq!(Some(&j.to_string()), map.get(&j, &test_guard));
            }
        }
        let mut one = "";
        if size > 1 {
            one = map.get(&1, &test_guard).expect("map should have at least one");
        }
        for i in 0..size {
            assert_eq!(Some(&i.to_string()), map.get(&i, &test_guard));
        }
        if size > 1 {
            assert_eq!(one, "1");
        }
    }


    use std::sync::{Arc, Mutex};
    #[derive(Clone)]
    pub struct NumberWithDrop {
        number: u64,
        arcmutex: Arc<Mutex<u64>>
    }
    impl ::std::hash::Hash for NumberWithDrop {
        fn hash<H: ::std::hash::Hasher>(&self, state: &mut H) {
            self.number.hash(state);
        }
    }
    impl PartialEq for NumberWithDrop {
        fn eq(&self, other: &Self) -> bool {
            self.number == other.number
        }
    }
    impl Eq for NumberWithDrop { }
    impl Drop for NumberWithDrop {
        fn drop(&mut self) {
            *self.arcmutex
                .lock()
                .expect(&format!("NumberWithDrop failed: number={}", self.number))
                += self.number;
        }
    }

    #[test]
    fn test_resize() {
        let map = &LockFreeHashMap::<u32, Vec<u64>>::with_capacity(4);
        scope(|scope| {
            for i in 1..256 {
                scope.spawn(move || {
                    let guard = pin();
                    let mut vec = vec![];
                    for i in 1..100 {
                        vec.push(i);
                    }
                    map.insert(i, vec, &guard);
                    map.remove(&i, &guard);
                });
            }
        });
       drop(map);
    }

    #[test]
    fn test_heavy_usage() {
        const NUMBER_OF_KEYS: usize = 100;
        const NUMBER_OF_VALUES_PER_KEY: usize = 5;
        const NUMBER_OF_THREADS: usize = 30;
        const NUMBER_OF_OPERATIONS_PER_THREAD: usize = 1000;
        let mut valid_states: Vec<(Box<u32>, Vec<Box<u32>>)> = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..NUMBER_OF_KEYS {
            let key = Box::new(rng.gen());
            let mut valid_values = Vec::new();
            for _ in 0..NUMBER_OF_VALUES_PER_KEY {
                valid_values.push(Box::new(rng.gen()));
            }
            valid_values.sort();
            valid_states.push((key, valid_values));
        }
        let valid_states = &valid_states;
        let map = &LockFreeHashMap::<Box<u32>, Box<u32>>::new();
        scope(|scope| {
            for _ in 0..NUMBER_OF_THREADS {
                scope.spawn(move || {
                    let mut rng = rand::thread_rng();
                    let guard = pin();
                    for _ in 0..NUMBER_OF_OPERATIONS_PER_THREAD {
                        match (rng.gen_range(0, 3),
                               rng.gen_range::<usize>(0, NUMBER_OF_KEYS),
                               rng.gen_range::<usize>(0, NUMBER_OF_VALUES_PER_KEY))
                        {
                            (0, key, value) => if let Some(previous) = map.insert(
                                valid_states[key].0.clone(),
                                valid_states[key].1[value].clone(),
                                &guard)
                            {
                                assert!(valid_states[key].1.binary_search(previous).is_ok());
                            },
                            (1, key, _) => if let Some(previous) = map.remove(
                                &valid_states[key].0, &guard)
                            {
                                assert!(valid_states[key].1.binary_search(previous).is_ok());
                            },
                            (2, key, _) => if let Some(get) = map.get(
                                &valid_states[key].0, &guard)
                            {
                                assert!(valid_states[key].1.binary_search(get).is_ok());
                            },
                            _ => unreachable!(),
                        }
                    }
                });
            }
        });
    }

}
