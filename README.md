# free-form-floor-plan-design-using-differentiable-voronoi-diagram

This project is a naive implementation of the paper [Free-form Floor Plan Design using Differentiable Voronoi Diagram](https://www.dropbox.com/scl/fi/culi7j1v14r9ax98rfmd6/2024_pg24_floorplan.pdf?rlkey=s5xwncuybrtsj5vyphhn61u0h&e=3&dl=0). The paper is based on the differentiable Voronoi diagram, but this repository uses `Shapely` and `Pytorch`. Specifically, PyTorch's autograd functionality for numerical differentiation is combined with Shapely's geometric operations to compute gradients.


<div style="display: flex">
    <p align="center">
        <img src="free_form_floor_plan_design_using_differentiable_voronoi_diagram/runs/shape_a/11-25-2024__00-20-42/optimization.gif" width=40%>
        <img src="free_form_floor_plan_design_using_differentiable_voronoi_diagram/runs/shape_b/11-25-2024__00-48-49/optimization.gif" width=40%>
    </p>
</div>
<p align="center">
  <i>Optimization processes</i>
</p>

<br>

# Installation

This repository uses the [image](/.devcontainer/Dockerfile) named `python:3.10.12-slim` for running devcontainer.


1. Ensure you have Docker and Visual Studio Code with the Remote - Containers extension installed.
2. Clone the repository.

    ```
        git clone https://github.com/PARKCHEOLHEE-lab/free-form-floor-plan-design-using-differentiable-voronoi-diagram.git
    ```

3. Open the project with VSCode.
4. When prompted at the bottom left on the VSCode, click `Reopen in Container` or use the command palette (F1) and select `Remote-Containers: Reopen in Container`.
5. VS Code will build the Docker container and set up the environment.
6. Once the container is built and running, you're ready to start working with the project.

<br>

# File Details
### examples
- `shape_a.py`: Orthogonal plan bounary a.
- `shape_b.py`: Orthogonal plan bounary b.
- `shape_c.py`: Orthogonal plan bounary c.
- `shape_duck.py`: Duck-shaped plan boundary.

### runs
- `shape_a/11-25-2024__00-20-42`
    - `configs.json`: Used configs for shape_a
    - `optimization.gif`: Animation for optimizing shape_a
- `shape_b/11-25-2024__00-48-49`
    - `configs.json`: Used configs for shape_b
    - `optimization.gif`: Animation for optimizing shape_b
- `shape_c/ `
    - `configs.json`: Used configs for shape_c
    - `optimization.gif`: Animation for optimizing shape_c

### src
- `generator.py`: Voronoi diagram's sites generator.
- `loss.py`: Loss functions based on the numerical differentiation to optimize the floor plans.
- `shape.py`: Example shapes to test.

<br>

# How to run
After installation, you can run any of the example files using the following commands:

```bash
    python free_form_floor_plan_design_using_differentiable_voronoi_diagram/examples/shape_duck.py
    python free_form_floor_plan_design_using_differentiable_voronoi_diagram/examples/shape_a.py
    python free_form_floor_plan_design_using_differentiable_voronoi_diagram/examples/shape_b.py
    python free_form_floor_plan_design_using_differentiable_voronoi_diagram/examples/shape_c.py
```

<br>

Each example will generate a floor plan based on different boundary shapes using the Voronoi diagram approach with the numerical differentiation and the geometric operations.
