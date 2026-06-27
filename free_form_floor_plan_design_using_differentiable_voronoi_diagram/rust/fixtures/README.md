# Golden fixtures

JSON oracles exported once from the Python implementation. The Rust test
suites are hermetic: `cargo test` needs no Python.

| File | Contents |
|---|---|
| `<example>.checkpoint.json` | shape geometry, exact f32 initial sites, k-means room indices, iteration-0 cell geometry / losses / gradients, GEOS cell + split-piece orderings, 10-iteration loss trace |
| `<example>.final.json` | final loss, final sites, per-room areas / area shares / connected-component counts after the full 800-iteration run |

All float32 tensors are widened losslessly to f64 and serialized with
shortest-round-trip decimals, so every value reloads bit-identically.

`geos_cell_order` / `split_pieces` record GEOS-internal orderings that
`loss.py`'s positional zip pairing depends on but that no reimplementation
can recompute; the Rust forward accepts them as an optional hint (checkpoint
tests only — standalone runs use the direct voronoice site→cell mapping).

## Regeneration

Requires the pinned Python environment (repo root):

```bash
python3.11 -m venv .venv-fixtures
.venv-fixtures/bin/pip install matplotlib==3.7.2 numpy==1.25.1 'Pillow>=9.3,<10' \
    Shapely==2.0.2 svgpathtools==1.6.1 torch==2.1.0 pytz==2024.2 \
    tensorboard==2.18.0 torch-kmeans==0.2.0
# Pillow is relaxed from the repo's 9.2.0 pin (no python 3.11 wheels);
# Pillow contributes to no exported number — it is image I/O only.

.venv-fixtures/bin/python free_form_floor_plan_design_using_differentiable_voronoi_diagram/rust/fixtures/export_fixtures.py                     # all examples
.venv-fixtures/bin/python free_form_floor_plan_design_using_differentiable_voronoi_diagram/rust/fixtures/export_fixtures.py shape_a --checkpoint-only

# re-embed the shape constants after regenerating:
.venv-fixtures/bin/python free_form_floor_plan_design_using_differentiable_voronoi_diagram/rust/fixtures/gen_shapes_rs.py
```
