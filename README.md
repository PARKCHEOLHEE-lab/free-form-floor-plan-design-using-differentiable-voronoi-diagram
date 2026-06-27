# free-form-floor-plan-design-using-differentiable-voronoi-diagram

Naive implementation of the paper [Free-form Floor Plan Design using Differentiable Voronoi Diagram](https://www.dropbox.com/scl/fi/culi7j1v14r9ax98rfmd6/2024_pg24_floorplan.pdf?rlkey=s5xwncuybrtsj5vyphhn61u0h&e=3&dl=0). The paper differentiates the Voronoi diagram analytically; this repository approximates the gradients <b>numerically</b> instead (central finite differences) and seeds rooms with KMeans for faster convergence. The original ([`python/`](free_form_floor_plan_design_using_differentiable_voronoi_diagram/python/README.md)) pairs `Shapely` geometry with `PyTorch` autograd over those numerical gradients; a <b>pure-Rust port</b> ([`rust/`](free_form_floor_plan_design_using_differentiable_voronoi_diagram/rust/README.md)) reproduces the same algorithm with hand-written finite differences verified equivalent and ~11× faster, and is compiled to <b>WebAssembly</b> for the live in-browser demo above ([`web/`](free_form_floor_plan_design_using_differentiable_voronoi_diagram/web/index.html)).

<br>


<p align="center">
    <img src="free_form_floor_plan_design_using_differentiable_voronoi_diagram/web/preset-e-demo.gif" width="48%">
    <img src="free_form_floor_plan_design_using_differentiable_voronoi_diagram/web/preset-b-demo.gif" width="48%">
</p>
<p align="center" color="gray">
  <i>
  Optimizing live in the browser (Rust compiled to WASM)
  </i>
</p>

# Structure

```
.
├── .devcontainer/
├── README.md
└── free_form_floor_plan_design_using_differentiable_voronoi_diagram/
    ├── python/
    │   ├── src/
    │   ├── examples/
    │   └── runs/
    ├── rust/
    │   ├── src/
    │   ├── examples/
    │   ├── fixtures/
    │   └── tests/
    ├── web/
    │   ├── src/
    │   ├── index.html
    │   └── worker.js
    └── comparison/
        ├── bench_python.py
        └── make_html.py
```

# Installation

This repository ships **two** dev containers, selectable from the same "Reopen in Container" picker:

- **voronoi-floorplan-python** — the original implementation, image `python:3.10.12-slim` ([`.devcontainer/python/Dockerfile`](/.devcontainer/python/Dockerfile)).
- **voronoi-floorplan-rust** — the pure-Rust port, image `rust:1-slim-bookworm` ([`.devcontainer/rust/Dockerfile`](/.devcontainer/rust/Dockerfile)).


1. Ensure you have Docker and Visual Studio Code with the Remote - Containers extension installed.
2. Clone the repository.

    ```
        git clone https://github.com/PARKCHEOLHEE-lab/free-form-floor-plan-design-using-differentiable-voronoi-diagram.git
    ```

3. Open the project with VSCode.
4. When prompted at the bottom left on the VSCode, click `Reopen in Container` or use the command palette (F1) and select `Dev Containers: Reopen in Container`. VS Code lists both configurations — pick **voronoi-floorplan-python** or **voronoi-floorplan-rust**.
5. VS Code will build the Docker container and set up the environment.
6. Once the container is built and running, you're ready to start working with the project.

See each implementation's own README for usage:
[`python/`](free_form_floor_plan_design_using_differentiable_voronoi_diagram/python/README.md) ·
[`rust/`](free_form_floor_plan_design_using_differentiable_voronoi_diagram/rust/README.md) ·
[`comparison/`](free_form_floor_plan_design_using_differentiable_voronoi_diagram/comparison/README.md)
