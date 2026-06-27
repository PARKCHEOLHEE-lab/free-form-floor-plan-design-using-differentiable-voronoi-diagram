# voronoi-floorplan (Rust · port)

A pure-Rust port of the Python implementation. The same algorithm, with pure-Rust geometry (`voronoice` + `geo` + Martinez–Rueda boolean ops via `geo-booleanop`), `ε = 1e-6` central finite-difference gradients, and PyTorch-semantics AdamW — no GEOS, no autograd. The Python original is in [`../python`](../python/README.md).

## Usage

```bash
cd free_form_floor_plan_design_using_differentiable_voronoi_diagram/rust
cargo run --release -- shape_a            # or shape_b | shape_c | shape_duck
cargo run --release -- shape_duck --iterations 200 --seed 42 --out-dir /tmp/duck
```

Each run writes `configs.json`, tensorboard `events.*` (7 loss scalars), and `optimization.gif` to `runs/<example>/<timestamp>/` (or `--out-dir`).

## Files

- `src/` — geometry, losses, finite-difference gradients, AdamW, CLI
- `fixtures/` — golden fixtures exported from Python (see `fixtures/README.md`)
- `tests/` — checkpoint + outcome equivalence suites
- `examples/` — additional example runners

## Tests

```bash
cargo test                                          # checkpoint + unit suites (hermetic, no Python)
cargo test --release --test outcome -- --ignored    # full 800-iteration outcome equivalence
```

Suites compare against the committed golden fixtures exported from Python. Bit-exact trajectory equality is mechanically impossible (f32 losses divided by `2e-6` amplify ~1e-9 cross-library geometry noise, and AdamW normalizes each step), so the criteria gate what is provable: exact-tolerance losses, quantum-signature gradients, the early trace, and end-to-end outcome equivalence.
