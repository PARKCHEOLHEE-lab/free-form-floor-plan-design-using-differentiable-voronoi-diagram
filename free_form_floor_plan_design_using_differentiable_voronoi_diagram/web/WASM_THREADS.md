# In-browser WASM threads for the gradient — recipe & why it's OFF by default

This documents a **working** wasm-threads build of the demo that parallelizes the
finite-difference gradient across Web Workers (wasm-bindgen-rayon), and the
measurement that shows **it is a net loss on this workload**, so the demo ships
**serial**. Keep this as a turnkey starting point: if the gradient's per-eval
allocation is reduced (see *The real fix* below), re-applying this makes threads
worthwhile.

## TL;DR result

The threaded build runs, is cross-origin isolated, and is correct — but **more
threads make it slower** (measured in a worker, `shape_a`, 40 sites):

| threads | ms/step | vs 1 thread |
|---|---|---|
| 1 | 25.1 | 1.00× |
| 2 | 33.7 | 0.75× |
| 4 | 101.7 | 0.25× |
| 8 | 217.0 | 0.12× |

**Why:** the gradient hot path is allocation-bound — `grad_local::local_total`
clones the whole cell array + rebuilds the Voronoi + builds unions, ~4·S ≈ 160
times per step. wasm's shared linear memory has a **single globally-locked heap**
(dlmalloc); concurrent allocations serialize and contend, and the near-linear
slowdown with thread count is that lock contention. The *identical* rayon code on
**native** scales 3.5–6.8× (native malloc is thread-scalable). The algorithm is
parallel; wasm's allocator is the wall.

## The real fix (prerequisite for threads paying off)

Cut allocation in `local_total`: stop cloning all `base_cells` every call and
recompute only the affected cells/rooms in place (diff, not copy). That speeds up
the **serial** path directly, and only then do threads have a scalable hot path.

---

## Recipe (to re-enable threads)

### 1. Core crate (`rust/`)
Already in place: a default-on `parallel` feature gates rayon via
`#[cfg(feature = "parallel")]` in `grad.rs` / `grad_local.rs`
(`grad_local::gradients_par` is the parallel gradient; bitwise-identical to the
serial `gradients`, proven by the `gradients_par_matches_serial_bitwise` test).
The web crate currently sets `default-features = false` to stay serial — flip
that on for threads.

### 2. Web crate (`web/Cargo.toml`)
```toml
voronoi-floorplan = { path = "../rust" }   # drop default-features=false (parallel ON)

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen-rayon = "1.2"   # resolves to 1.3.x
rayon = "1"

# wasm-opt STRIPS the shared-memory flag (-> DataCloneError handing memory to
# workers). Disable it for the threaded build.
[package.metadata.wasm-pack.profile.release]
wasm-opt = false
```

### 3. `web/.cargo/config.toml` (the exact link-flag set — every flag earned)
```toml
[target.wasm32-unknown-unknown]
rustflags = [
  "-C", "target-feature=+atomics,+bulk-memory,+mutable-globals",
  "-C", "link-arg=--shared-memory",      # +atomics alone left memory NON-shared (flags 0x00/0x01)
  "-C", "link-arg=--import-memory",       # wasm-bindgen thread transform asserts mem.import.is_some()
  "-C", "link-arg=--max-memory=4294967296", # shared memory REQUIRES a max
  "-C", "link-arg=--export=__heap_base",  # release --gc-sections strips these; wasm-bindgen reads them
  "-C", "link-arg=--export=__wasm_init_tls",
  "-C", "link-arg=--export=__tls_size",
  "-C", "link-arg=--export=__tls_align",
  "-C", "link-arg=--export=__tls_base",
]
```
This block is wasm32-only, so native `cargo test` is unaffected. Put the flags
HERE (not in `CARGO_TARGET_*_RUSTFLAGS` env) so they also reach the std that
`build-std` rebuilds — the env-var form did not, leaving std non-atomic.

### 4. `web/src/lib.rs`
```rust
#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;   // exposes initThreadPool(n) to JS
// ...and in advance(): let grads = ctx.gradients_par(&self.sites);
```

### 5. `web/worker.js` — init the pool, from the WORKER (not the main thread)
```js
import init, { WasmOpt, initThreadPool } from './pkg/voronoi_floorplan_web.js';
// in ensureReady, after await init():
await initThreadPool(Math.max(1, navigator.hardwareConcurrency || 4));
```
Rayon must run off the main thread: a browser main thread **cannot
`Atomics.wait`**, so `par_iter` there spin-waits and anti-scales even harder. The
demo already runs the optimizer in `worker.js`, which can block — keep it there.

### 6. Build (nightly + build-std), in the `vfp-wasm-nightly` image
```bash
export CARGO_UNSTABLE_BUILD_STD="panic_abort,std"   # wasm-pack 0.13 mis-parses `-- -Z`
rustup run nightly wasm-pack build --target web --out-dir pkg
# wasm-bindgen-rayon 1.3 + --target web: the worker does import('../../..') (a
# directory) which native ESM (no bundler) 404s. Point it at the module file:
sed -i "s#import('../../..')#import('../../../voronoi_floorplan_web.js')#" \
  pkg/snippets/wasm-bindgen-rayon-*/src/workerHelpers.js
```

### 7. Serve with COOP/COEP (SharedArrayBuffer needs cross-origin isolation)
`serve.py` already sends `Cross-Origin-Opener-Policy: same-origin` +
`Cross-Origin-Embedder-Policy: require-corp`. A plain `python -m http.server`
does NOT — the threaded module then fails to instantiate. Verify
`self.crossOriginIsolated === true` in the page.

### 8. Measure
Run the timed loop **inside a worker** (see §5). Compare `?threads=1` vs
`?threads=<cores>` as separate page loads (initThreadPool is once-per-context).

## Gotchas, in the order they bite
1. `+atomics` alone ⇒ non-shared memory (flags `0x00`). Need `--shared-memory` + `--max-memory`.
2. wasm-bindgen thread transform requires **imported** memory (`--import-memory`).
3. `--gc-sections` strips `__heap_base` / `__wasm_init_tls` / `__tls_*` that wasm-bindgen reads ⇒ force-export them.
4. `wasm-opt` strips the shared flag ⇒ disable it.
5. RUSTFLAGS must reach `build-std`'s std ⇒ use `.cargo/config.toml`, not the target env var.
6. wasm-bindgen-rayon worker `import('../../..')` is a directory ⇒ 404 under native ESM ⇒ patch to the file.
7. Don't call rayon from the main thread (no `Atomics.wait`).
8. The payoff is gated by the allocation-bound hot path (the actual blocker).
