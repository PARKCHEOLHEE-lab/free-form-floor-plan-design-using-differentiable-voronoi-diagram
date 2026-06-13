# voronoi-floorplan (Python — original implementation)

The original implementation of free-form floor plan generation. Voronoi site
positions are optimized with PyTorch's autograd used over **numerical
differentiation**, combined with Shapely's geometric operations to compute the
gradients. Room cells are seeded with KMeans so the optimization converges
faster than from random initialization.

The pure-Rust port of this code lives in [`../rust`](../rust/README.md); a
side-by-side timing and evolution-GIF comparison of the two is in
[`../comparison`](../comparison/README.md).

## Files

### examples
- `shape_a.py`: Orthogonal plan boundary a.
- `shape_b.py`: Orthogonal plan boundary b.
- `shape_c.py`: Orthogonal plan boundary c.
- `shape_duck.py`: Duck-shaped plan boundary.

### runs
- `shape_<name>/<datetime>`
    - `configs.json`: Used configs
    - `events.*`: Tensorboard logs
    - `optimization.gif`: Animation for optimizing the shape

### src
- `generator.py`: Voronoi diagram's sites generator.
- `loss.py`: Loss functions based on the numerical differentiation to optimize the floor plans.
- `shape.py`: Example shapes to test.

## How to run

After [installation](../../README.md#installation) (the **voronoi-floorplan-python**
dev container), run any example from the repository root:

```bash
python free_form_floor_plan_design_using_differentiable_voronoi_diagram/python/examples/shape_duck.py
python free_form_floor_plan_design_using_differentiable_voronoi_diagram/python/examples/shape_a.py
python free_form_floor_plan_design_using_differentiable_voronoi_diagram/python/examples/shape_b.py
python free_form_floor_plan_design_using_differentiable_voronoi_diagram/python/examples/shape_c.py
```

Each example generates a floor plan for a different boundary shape using the
Voronoi diagram approach with numerical differentiation and geometric
operations.

## Future works
- Set entrances of the plan
- Set a graph-based constraint for the connectivity between rooms
