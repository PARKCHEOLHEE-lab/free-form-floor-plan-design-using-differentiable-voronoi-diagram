# Python vs Rust comparison

Benchmarks the original Python implementation against the Rust port on the
**same workload** (each loads the identical golden-fixture start and runs the
same iterations), and renders side-by-side evolution GIFs into a single HTML
report. Timing covers the optimization compute only (forward loss +
finite-difference backward + AdamW step); rendering and IO are excluded.

Generated artifacts (GIFs, HTML, timing JSON) are written to `output/` and are
**git-ignored** — regenerate them with the steps below.

## Regenerate

From the repo root, with the fixture Python environment active (see
`free_form_floor_plan_design_using_differentiable_voronoi_diagram/rust/fixtures/README.md`
for the pinned deps):

```bash
PKG=free_form_floor_plan_design_using_differentiable_voronoi_diagram

# Rust side — timing JSON + evolution GIF per example
cd "$PKG/rust"
for ex in shape_a shape_b shape_c shape_duck; do
  cargo run --release --example bench_compare -- "$ex" 50 ../comparison/output
done
cd ../..

# Python side — run each sequentially (each uses the whole machine for a fair
# multiprocessing-backward timing)
for ex in shape_a shape_b shape_c shape_duck; do
  .venv-fixtures/bin/python "$PKG/comparison/bench_python.py" "$ex" 50 "$PKG/comparison/output"
done

# Build the report
python3 "$PKG/comparison/make_html.py" "$PKG/comparison/output"
open "$PKG/comparison/output/comparison.html"
```

## Files

- `bench_python.py` — Python benchmark: loads the fixture start, times the
  optimization loop, renders the matplotlib GIF, writes `python_timing_*.json`.
- `make_html.py` — reads the timing JSONs and references the GIFs to build
  `comparison.html` (timing table, speedup bars, side-by-side GIFs).
- the Rust benchmark lives at `free_form_floor_plan_design_using_differentiable_voronoi_diagram/rust/examples/bench_compare.rs`.

## What the report shows

Per-iteration timing (Rust runs ~11× faster on the measured machine) and the
optimization evolution from the identical start. The two animations track
closely early and then separate — that is float32 finite-difference chaos, not
a porting bug (the frozen-input checkpoint tests prove 1e-6 loss equivalence
and the f32-quantum gradient signature). See the report's own notes.
