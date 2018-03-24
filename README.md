# LockFreeHashMap-rs

[![License](https://img.shields.io/badge/license-LGPL--3.0+-blue.svg)](https://github.com/rolag/lockfreehashmap)
[![Cargo](https://img.shields.io/crates/v/lockfreehashmap.svg)](https://crates.io/crates/lockfreehashmap)
[![Documentation](https://docs.rs/lockfreehashmap/badge.svg)](https://docs.rs/lockfreehashmap)
[![Continuous Integration](https://api.travis-ci.org/rolag/lockfreehashmap-rs.svg?branch=master)](https://travis-ci.org/rolag/lockfreehashmap-rs)

A concurrent, lock-free hash map for Rust.

This is an implementation of the lock-free hash map created by Dr. Cliff Click.
Click released a [talk](https://www.youtube.com/watch?v=HJ-719EGIts) about his hash map.
Additionally, "reference" Java code is available
[here](https://github.com/boundary/high-scale-lib/blob/master/src/main/java/org/cliffc/high_scale_lib/NonBlockingHashMap.java)
and more recently
[here](https://github.com/JCTools/JCTools/blob/master/jctools-core/src/main/java/org/jctools/maps/NonBlockingHashMap.java).


## Getting Started

This crate is available on [crates.io](https://crates.io/crates/lockfreehashmap).

To use this crate in your project, add the following to your `Cargo.toml` file:
```toml
[dependencies]
lockfreehashmap = "0.1"
```
and then add to your project root file:
```rust
extern crate lockfreehashmap;
```

## Example
```rust
extern crate lockfreehashmap;
use lockfreehashmap::{self, LockFreeHashMap};

let map = LockFreeHashMap::<u8, u8>::new();
let insert_guard = lockfreehashmap::pin();
for i in 1..4 {
    map.insert(i, i, &insert_guard);
}
drop(insert_guard);

let map = &map;
lockfreehashmap::scope(|scope| {
    // Spawn multiple threads, e.g. for a server that executes some actions on a loop
    for _ in 0..16 {
        scope.spawn(|| {
            loop {
                let mut line = String::new();
                ::std::io::stdin().read_line(&mut line).unwrap();
                let iter = line.split_whitespace();
                let command: &str = iter.next().unwrap();
                let key: u8 = iter.next().unwrap().parse();
                let value: u8 = iter.next().unwrap().parse();
                let guard = lockfreehashmap::pin();
                let _result = match command {
                    "insert" => map.insert(key, value, &guard),
                    _ => {/* ... */},
                };
                drop(guard);
            }
        });
    }
});
```

## License
GNU Lesser General Public License v3.0 or any later version

See [LICENSE](LICENSE) and [LICENSE.LESSER](LICENSE.LESSER) for details.
