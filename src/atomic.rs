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
use crossbeam::epoch::{Atomic, Guard, Owned, Shared};
use std::fmt;
use std::ops::Deref;
use std::sync::atomic::Ordering;

pub const ORDERING: Ordering = Ordering::SeqCst;

pub struct NotNull<'a, T: 'a>(Shared<'a, T>);

impl<'a, T> NotNull<'a, T> {
    pub fn as_maybe_null(&self) -> MaybeNull<'a, T> {
        MaybeNull(self.0)
    }
    pub fn as_shared(&self) -> Shared<'a, T> {
        self.0
    }
    pub fn deref(&self) -> &'a T {
        // This is safe because
        // 1) This type is only created in situations it's guaranteed not to be null
        // 2) This type only created from types in this module, which never uses
        // `Ordering::Relaxed`.
        debug_assert!(!self.0.is_null());
        unsafe { self.0.deref() }
    }
    pub unsafe fn drop(self) {
        let shared = self.0;
        debug_assert!(!shared.is_null());
        drop(shared.into_owned());
    }
}

impl<'a, T: fmt::Debug> fmt::Debug for NotNull<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.as_maybe_null())
    }
}

impl<'a, T> Clone for NotNull<'a, T> {
    fn clone<'b>(&'b self) -> Self {
        NotNull(self.0)
    }
}
impl<'a, T> Copy for NotNull<'a, T> { }

impl<'a, T> Deref for NotNull<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.deref()
    }
}

pub struct NotNullOwned<T>(Owned<T>);

impl<T> NotNullOwned<T> {
    pub fn new(value: T) -> Self {
        NotNullOwned(Owned::new(value))
    }
    pub fn into_owned(self) -> Owned<T> {
        self.0
    }
}

impl<T: fmt::Debug> fmt::Debug for NotNullOwned<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.deref())
    }
}

impl<T> Deref for NotNullOwned<T> {
    type Target = T;
    fn deref(&self) -> &T {
        // This is safe because this is only created from types in this module, which never use
        // `Ordering::Relaxed`.
        self.0.deref()
    }
}


pub struct MaybeNull<'a, T: 'a>(Shared<'a, T>);

impl<'a, T> Clone for MaybeNull<'a, T> {
    fn clone<'b>(&'b self) -> Self {
        MaybeNull(self.0)
    }
}
impl<'a, T> Copy for MaybeNull<'a, T> { }


impl<'a, T> MaybeNull<'a, T> {
    pub fn from_shared(shared: Shared<'a, T>) -> Self {
        MaybeNull(shared)
    }
    pub fn as_shared(&self) -> Shared<T> {
        self.0
    }
    pub fn as_option(&self) -> Option<NotNull<'a, T>> {
        match self.0.is_null() {
            true => None,
            false => Some(NotNull(self.0)),
        }
    }
    pub unsafe fn try_defer_drop(self, guard: &Guard) {
        guard.defer(move || {
            if !self.as_shared().is_null() {
                self.as_shared().into_owned();
            }
        })
    }
}

impl<'a, T: fmt::Debug> fmt::Debug for MaybeNull<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.as_option() {
            Some(not_null) => write!(f, "{:?}", not_null.deref()),
            None => write!(f, "(Null)"),
        }
    }
}

pub struct AtomicPtr<T>(Atomic<T>);

impl<T> AtomicPtr<T> {
    pub fn new(value: Option<T>) -> Self {
        match value {
            Some(t) => AtomicPtr(Atomic::new(t)),
            None => AtomicPtr(Atomic::null()),
        }
    }

    pub fn load<'g>(&self, guard: &'g Guard) -> MaybeNull<'g, T> {
        MaybeNull(self.0.load(ORDERING, guard))
    }

    pub unsafe fn take<'g>(&self, guard: &'g Guard) -> Option<Box<T>> {
        let shared = self.0.swap(Shared::null(), ORDERING, guard);
        match shared.is_null() {
            true => None,
            false => Some(shared.into_owned().into_box())
        }
    }

    pub fn compare_and_set<'g>(&self, compare: MaybeNull<T>, set: NotNull<'g, T>, guard: &'g Guard)
        -> Result<NotNull<'g, T>, (MaybeNull<'g, T>, NotNull<'g, T>)>
    {
        self.0.compare_and_set(compare.as_shared(), set.as_shared(), ORDERING, guard)
            .map(|set| NotNull(set))
            .map_err(|e| (MaybeNull(e.current), NotNull(e.new)))
    }

    pub fn compare_null_and_set_owned<'g>(
        &self,
        set: NotNullOwned<T>,
        guard: &'g Guard
    ) -> Result<NotNull<'g, T>, (NotNull<'g, T>, NotNullOwned<T>)>
    {
        self.0.compare_and_set(Shared::null(), set.into_owned(), ORDERING, guard)
            .map(|set| NotNull(set))
            .map_err(|e| (NotNull(e.current), NotNullOwned(e.new)))
    }

    pub fn compare_and_set_owned<'g>(
        &self,
        compare: MaybeNull<T>,
        set: NotNullOwned<T>,
        guard: &'g Guard
    ) -> Result<NotNull<'g, T>, (MaybeNull<'g, T>, NotNullOwned<T>)>
    {
        self.0.compare_and_set(compare.as_shared(), set.into_owned(), ORDERING, guard)
            .map(|set| NotNull(set))
            .map_err(|e| (MaybeNull(e.current), NotNullOwned(e.new)))
    }

    pub fn compare_and_set_owned_weak<'g>(
        &self,
        compare: MaybeNull<T>,
        set: NotNullOwned<T>,
        guard: &'g Guard
    ) -> Result<NotNull<'g, T>, (MaybeNull<'g, T>, NotNullOwned<T>)>
    {
        self.0.compare_and_set_weak(compare.as_shared(), set.into_owned(), ORDERING, guard)
            .map(|set| NotNull(set))
            .map_err(|e| (MaybeNull(e.current), NotNullOwned(e.new)))
    }

    pub fn set_should_deallocate<'g>(&self, guard: &'g Guard) {
        self.0.fetch_or(1, ORDERING, guard);
    }

    pub fn should_deallocate<'g>(&self, guard: &'g Guard) -> bool {
        self.0.fetch_or(0, ORDERING, guard).tag() == 1
    }

    pub unsafe fn try_drop(&mut self, guard: &Guard) {
        let inner = self.0.swap(Shared::null(), ORDERING, &guard);
        if !inner.is_null() {
            inner.into_owned();
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for AtomicPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let guard = crossbeam::epoch::pin();
        write!(f, "{:?}", self.load(&guard))
    }
}

pub struct AtomicBox<T>(Atomic<T>);

impl<T> AtomicBox<T> {
    pub fn new(value: T) -> Self {
        AtomicBox(Atomic::new(value))
    }

    pub fn load<'g>(&self, guard: &'g Guard) -> NotNull<'g, T> {
        NotNull(self.0.load(ORDERING, guard))
    }

    pub fn compare_and_set_shared<'g>(
        &'g self,
        compare: NotNull<T>,
        set: NotNull<'g, T>,
        guard: &'g Guard
    )
        -> Result<NotNull<'g, T>, (NotNull<'g, T>, NotNull<T>)>
    {
        self.0.compare_and_set(compare.0, set.0, ORDERING, guard)
            .map(|set| NotNull(set))
            .map_err(|e| (NotNull(e.current), NotNull(e.new)))
    }
}

impl<T: fmt::Debug> fmt::Debug for AtomicBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let guard = crossbeam::epoch::pin();
        write!(f, "{:?}", self.load(&guard).as_maybe_null())
    }
}

impl<T> Drop for AtomicBox<T> {
    fn drop(&mut self) {
        let guard = crossbeam::epoch::pin();
        let inner = self.0.swap(Shared::null(), ORDERING, &guard);
        debug_assert!(!inner.is_null());
        if !inner.is_null() {
            unsafe { inner.into_owned(); }
        }
    }
}
