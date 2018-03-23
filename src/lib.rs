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

// NOTE: To use valgrind, uncomment the lines below and recompile.
// NOTE: Then `valgrind --leak-check=full --show-leak-kinds=all ./target/debug/deps/lockfreehashmap-***`
//#![feature(alloc_system, global_allocator, allocator_api)]
//extern crate alloc_system;
//use alloc_system::System;
//#[global_allocator]
//static A: System = System;

extern crate crossbeam;

use crossbeam::epoch::Guard;
use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::Hash;

mod atomic;
mod map_inner;

pub use crossbeam::epoch::pin;
use atomic::AtomicBox;
use map_inner::{KeyCompare, MapInner, Match, PutValue, ValueSlot};

pub const COPY_CHUNK_SIZE: usize = 32;

pub struct LockFreeHashMap<'v, K, V: 'v, S = RandomState> {
    /// Points to the newest map (after it's been fully resized). Always non-null.
    inner: AtomicBox<MapInner<'v,K,V,S>>,
}

impl<'guard, 'v: 'guard, K: Hash + Eq + 'guard, V> LockFreeHashMap<'v,K,V> {

    /// The default size of a new `LockFreeHashMap` when created by `LockFreeHashMap::new()`.
    pub const DEFAULT_CAPACITY: usize = MapInner::<K,V>::DEFAULT_CAPACITY;

    /// Creates a new `LockFreeHashMap`.
    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    /// Creates a new `LockFreeHashMap` of a given size. Uses the next power of two if size is not
    /// a power of two.
    pub fn with_capacity(size: usize) -> Self {
        LockFreeHashMap { inner: AtomicBox::new(MapInner::with_capacity(size)) }
    }

    /// Returns the number of elements the map can hold without reallocating.
    pub fn capacity(&self) -> usize {
        let guard = crossbeam::epoch::pin();
        self.load_inner(&guard).capacity()
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        let guard = crossbeam::epoch::pin();
        self.load_inner(&guard).len()
    }

    // keys, values, entry

    /// Clears the map, removing all key-value pairs. Keeps the allocated memory for reuse.
    pub fn clear(&self) {
        unimplemented!()
    }

    /// Returns true if the map contains a value for the specified key. The key may be any
    /// borrowed form of the map's key type, but Hash and Eq on the borrowed form must match those
    /// for the key type.
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
        where K: Borrow<Q>,
              Q: Hash + Eq + PartialEq<K>,
    {
        let guard = crossbeam::epoch::pin();
        self.get(key, &guard).is_some()
    }

    fn load_inner(&self, guard: &'guard Guard) -> &'guard MapInner<'v,K,V> {
        self.inner.load(&guard).deref()
    }

    /// Returns a reference to the value corresponding to the key. The key may be any borrowed
    /// form of the map's key type, but Hash and Eq on the borrowed form must match those for the
    /// key type.
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
    pub fn replace<Q: ?Sized>(&self, key: &Q, guard: &'guard Guard) -> Option<&'guard V>
        where K: Borrow<Q>,
              Q: Hash + Eq + PartialEq<K>,
    {
        let value_slot: Option<&ValueSlot<V>> = self.load_inner(guard).put_if_match(
            KeyCompare::OnlyCompare(key),
            PutValue::new_tombstone(),
            Match::AnyKeyValuePair,
            &self.inner,
            &guard
        );
        return ValueSlot::as_inner(value_slot);
    }

    /// Removes a key from the map, returning the value at the key if the key was previously in the
    /// map. The key may be any borrowed form of the map's key type, but Hash and Eq on the
    /// borrowed form must match those for the key type.
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
        let guard = crossbeam::epoch::pin();
        unsafe {
            self.inner.load(&guard).deref().drop_newer_maps(&guard);
        }
    }
}

impl<'guard, 'v: 'guard, K: Hash + Eq + fmt::Debug, V: fmt::Debug>
    fmt::Debug for LockFreeHashMap<'v,K,V>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let guard = crossbeam::epoch::pin();
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
        let test_guard = crossbeam::epoch::pin();
        for i in 1..4 {
            map.insert(i, i, &test_guard);
        }
        let map = &map;
        crossbeam::scope(|scope| {
            scope.spawn(|| {
                let test_guard = crossbeam::epoch::pin();
                assert_eq!(map.get(&1, &test_guard), Some(&1));
            });
            scope.spawn(|| {
                let test_guard = crossbeam::epoch::pin();
                assert_eq!(map.insert(100, 101, &test_guard), None);
            });
            scope.spawn(|| {
                let test_guard = crossbeam::epoch::pin();
                assert_eq!(map.insert(5, 4, &test_guard), None);
            });
            scope.spawn(|| {
                let test_guard = crossbeam::epoch::pin();
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
        let test_guard = crossbeam::epoch::pin();
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
        crossbeam::scope(|scope| {
            for i in 1..256 {
                scope.spawn(move || {
                    let guard = crossbeam::epoch::pin();
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
        crossbeam::scope(|scope| {
            for _ in 0..NUMBER_OF_THREADS {
                scope.spawn(move || {
                    let mut rng = rand::thread_rng();
                    let guard = crossbeam::epoch::pin();
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
