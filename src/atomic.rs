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

use crossbeam_epoch::{Atomic, Guard, Owned, Shared};
use std::fmt;
use std::ops::Deref;
use std::sync::atomic::Ordering;

pub const ORDERING: Ordering = Ordering::SeqCst;

/// A wrapper around [Shared], for values that we know are not null and are safe to dereference.
pub struct NotNull<'t, T: 't>(Shared<'t, T>);

impl<'t, T> NotNull<'t, T> {
    /// This is effectively just weakening the guarantees around this type.
    pub fn as_maybe_null(&self) -> MaybeNull<'t, T> {
        MaybeNull(self.0)
    }
    pub fn as_shared(&self) -> Shared<'t, T> {
        self.0
    }
    /// Stronger version of `Deref`.  This returns `&'t T`, rather than `&'f T`.
    pub fn deref<'f>(&'f self) -> &'t T {
        // This is safe because
        // 1) This type is only created in situations it's guaranteed not to be null
        // 2) This type only created from types in this module, which never uses
        // `Ordering::Relaxed`.
        // 3) `T: 't`, i.e. T outlives 't, so we can return a reference to it
        debug_assert!(!self.0.is_null());
        unsafe { self.0.deref() }
    }
    /// Drop the underlying value behind this pointer.
    ///
    /// # Unsafe
    /// This is unsafe because only one thread can call this for any single underlying value.
    pub unsafe fn drop(self) {
        let shared = self.0;
        debug_assert!(!shared.is_null());
        drop(shared.into_owned());
    }
}

impl<'t, T: fmt::Debug> fmt::Debug for NotNull<'t, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.as_maybe_null())
    }
}

impl<'t, T> Clone for NotNull<'t, T> {
    fn clone<'b>(&'b self) -> Self {
        NotNull(self.0)
    }
}

impl<'t, T> Copy for NotNull<'t, T> { }

impl<'t, T> Deref for NotNull<'t, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.deref()
    }
}

/// A wrapper around [Owned], for values that we know are not null.
/// This is the owned version of [NotNull].
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
        // This is safe because this is only created from types in this module, which never uses
        // `Ordering::Relaxed` (or at least never returns pointers to values retrieved with
        // Relaxed).
        self.0.deref()
    }
}


/// Similarly to [NotNull], this is a wrapper around [Shared] for values where the pointer could be
/// null.
pub struct MaybeNull<'t, T: 't>(Shared<'t, T>);

impl<'t, T> Clone for MaybeNull<'t, T> {
    fn clone<'b>(&'b self) -> Self {
        MaybeNull(self.0)
    }
}

impl<'t, T> Copy for MaybeNull<'t, T> { }


impl<'t, T> MaybeNull<'t, T> {
    pub fn from_shared(shared: Shared<'t, T>) -> Self {
        MaybeNull(shared)
    }
    pub fn as_shared(&self) -> Shared<T> {
        self.0
    }
    pub fn as_option(&self) -> Option<NotNull<'t, T>> {
        match self.0.is_null() {
            true => None,
            false => Some(NotNull(self.0)),
        }
    }
    /// Drop the underlying value behind this pointer if it's not null and after enough epochs have
    /// passed.
    ///
    /// # Unsafe
    /// This is unsafe because only one thread can call this for any single underlying value.
    pub unsafe fn try_defer_drop(self, guard: &Guard) {
        if !self.as_shared().is_null() {
            guard.defer(move || {
                self.as_shared().into_owned();
            })
        }
    }
}

impl<'t, T: fmt::Debug> fmt::Debug for MaybeNull<'t, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.as_option() {
            Some(not_null) => write!(f, "{:?}", not_null.deref()),
            None => write!(f, "(Null)"),
        }
    }
}

/// Wrapper around [Atomic]. Represents a pointer that can be null at first but never again
/// afterwards.
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

    /// Check if the pointer is not null (i.e. the value exists), using `Ordering::Relaxed`.
    pub fn relaxed_exists<'g>(&self, guard: &'g Guard) -> bool {
        // It is safe to use `Ordering::Relaxed` as long as we don't dereference it
        !self.0.load(Ordering::Relaxed, guard).is_null()
    }

    /// Swaps out the current value, leaving null in its place. Equivalent to [Option::take()].
    ///
    /// # Unsafe
    /// This is unsafe because there must be no other pointers that can ever access this
    /// `AtomicPtr` after this function is called. This is intended to be used for [Drop::drop()].
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

    pub fn compare_null_and_set<'g>(
        &self,
        set: NotNull<'g, T>,
        guard: &'g Guard
    ) -> Result<NotNull<'g, T>, (NotNull<'g, T>, NotNull<'g, T>)>
    {
        self.0.compare_and_set(Shared::null(), set.as_shared(), ORDERING, guard)
            .map(|set| NotNull(set))
            .map_err(|e| (NotNull(e.current), NotNull(e.new)))
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

    pub fn tag<'g>(&self, guard: &'g Guard) {
        self.0.fetch_or(1, ORDERING, guard);
    }

    pub fn is_tagged<'g>(&self, guard: &'g Guard) -> bool {
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
        let guard = ::pin();
        write!(f, "{:?}", self.load(&guard))
    }
}

/// Wrapper around [Atomic]. Represents a pointer that can never be null.
pub struct AtomicBox<T>(Atomic<T>);

impl<T> AtomicBox<T> {
    pub fn new(value: T) -> Self {
        AtomicBox(Atomic::new(value))
    }

    pub fn load<'g>(&self, guard: &'g Guard) -> NotNull<'g, T> {
        NotNull(self.0.load(ORDERING, guard))
    }

    pub fn replace<'g>(&self, value: T) {
        let guard = &::pin();
        let contents = self.0.swap(Owned::new(value), ORDERING, &guard);
        unsafe { guard.defer(move || contents.into_owned()); }
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
        let guard = ::pin();
        write!(f, "{:?}", self.load(&guard).as_maybe_null())
    }
}

impl<T> Drop for AtomicBox<T> {
    fn drop(&mut self) {
        let guard = ::pin();
        let inner = self.0.swap(Shared::null(), ORDERING, &guard);
        debug_assert!(!inner.is_null());
        if !inner.is_null() {
            unsafe { inner.into_owned(); }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_atomic_ptr() {
        let ptr = AtomicPtr::new(None);
        let guard = ::pin();
        assert!(!ptr.relaxed_exists(&guard));
        assert!(ptr.load(&guard).as_option().is_none());
        let _ok = ptr.compare_null_and_set_owned(NotNullOwned::new(String::from("first")), &guard);
        let err = ptr.compare_null_and_set_owned(NotNullOwned::new(String::from("second")), &guard)
            .unwrap_err();
        let current = err.0;
        let tried_to_insert = err.1;
        assert_eq!(&*tried_to_insert, "second");
        assert_eq!(&*current, "first");
        let _ok = ptr.compare_and_set_owned(
            current.as_maybe_null(), NotNullOwned::new(String::from("third")), &guard
        ).unwrap();
        unsafe { guard.defer(move || current.drop()); }
        assert_eq!(*_ok, "third");
        assert!(!ptr.is_tagged(&guard));
        ptr.tag(&guard);
        assert!(ptr.is_tagged(&guard));
        let inner = unsafe { ptr.take(&guard).unwrap() };
        assert_eq!(*inner, "third");
        drop(ptr);
    }
}
