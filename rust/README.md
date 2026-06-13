# voronoi-floorplan (Rust port)

A pure-Rust port of the Python implementation in this repository:
free-form floor plan generation by optimizing Voronoi site positions with
central finite-difference gradients and AdamW.

- Geometry: [`voronoice`] (Voronoi) + [`geo`] + Martinez–Rueda boolean ops
  ([`geo-booleanop`]) — no GEOS, no C dependencies, no autograd framework.
- Strictly faithful algorithm: identical loss formulas (including the
  positional zip pairing quirk at MultiPolygon splits), `ε = 1e-6` central
  finite differences applied in f32, PyTorch-semantics AdamW, and torch's
  exact f32/f64 casting boundaries (torch's default dtype is float32; every
  tensor materialization is an f64 → f32 cast).
- Parallelism: [`rayon`] over the 2·N·2 perturbed forward evaluations
  (replacing Python's multiprocessing pool).

## Usage

```bash
cd rust
cargo run --release -- shape_a            # or shape_b | shape_c | shape_duck
cargo run --release -- shape_duck --iterations 200 --seed 42 --out-dir /tmp/duck
```

Each run writes `configs.json`, a tensorboard event file with the 7 scalar
tags (`loss`, `loss_wall`, `loss_area`, `loss_lloyd`, `loss_topo`,
`loss_bb`, `loss_cell_area`), and `optimization.gif` to
`runs/<example>/<timestamp>/` (or `--out-dir`).

## Tests

```bash
cargo test                                          # checkpoint + unit suites (hermetic, no Python)
cargo test --release --test outcome -- --ignored    # full 800-iteration outcome equivalence
```

The test suites compare against committed golden fixtures exported once from
the Python implementation (see `fixtures/README.md`):

| Suite | What it proves |
|---|---|
| `checkpoint` | shapes, forward Voronoi geometry, all 6 loss components, total loss (≤1e-6 rel), finite-difference gradients (stair rule), and the iteration 1–4 optimization trace (≤1%) match Python on frozen inputs for all 4 examples |
| `init` | self-contained seeded initialization: deterministic, in-boundary, k nonempty clusters |
| `artifacts` | tfevents records round-trip with valid masked CRC32C; GIF frames decode and are non-blank |
| `cli` | arg parsing + a 2-iteration smoke run produces all artifacts |
| `outcome` | full runs reach floor plans equivalent to Python's final result |

### Equivalence limits (measured, documented)

Bit-exact trajectory equality with Python is mechanically impossible: the
losses are float32, the backward divides f32 loss differences by 2e-6, and
~1e-9 cross-library f64 geometry noise flips last bits, shifting gradient
entries by exact integer multiples of `ulp(loss)/2e-6` ("stairs"). Adam
normalizes each step to ±lr from the gradient sign, so trajectories separate
macroscopically after ~4 iterations — confirmed intrinsic by Python-vs-Python
perturbation controls. The acceptance criteria therefore gate what is
provable: exact-tolerance losses, quantum-signature gradients, the early
trace, and end-to-end outcome equivalence.

[`voronoice`]: https://crates.io/crates/voronoice
[`geo`]: https://crates.io/crates/geo
[`geo-booleanop`]: https://github.com/21re/rust-geo-booleanop
[`rayon`]: https://crates.io/crates/rayon
