# voronoi-floorplan (Python · original)

The original implementation. Voronoi site positions are optimized with `PyTorch` autograd over **numerical differentiation**, using `Shapely` for the geometry, with rooms seeded by KMeans for faster convergence. The pure-Rust port is in [`../rust`](../rust/README.md).

## Usage

After [installation](../../README.md#installation) (the **voronoi-floorplan-python** dev container), run any example from the repository root:

```bash
python free_form_floor_plan_design_using_differentiable_voronoi_diagram/python/examples/shape_a.py   # or shape_b | shape_c | shape_duck
```

Each run writes `configs.json`, tensorboard `events.*`, and `optimization.gif` to `runs/<shape>/<datetime>/`.

## Files

- `src/generator.py` — Voronoi sites generator
- `src/loss.py` — loss functions over numerical differentiation
- `src/shape.py` — example boundary shapes
- `examples/` — `shape_a` · `shape_b` · `shape_c` · `shape_duck`

