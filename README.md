# free-form-floor-plan-design-using-differentiable-voronoi-diagram

This project is a naive implementation of the paper [Free-form Floor Plan Design using Differentiable Voronoi Diagram](https://www.dropbox.com/scl/fi/culi7j1v14r9ax98rfmd6/2024_pg24_floorplan.pdf?rlkey=s5xwncuybrtsj5vyphhn61u0h&e=3&dl=0). The paper is based on the <b>differentiable Voronoi diagram</b>, but this repository uses `Shapely` and `Pytorch`. Specifically, PyTorch's autograd functionality for <b>numerical differentiation</b> is combined with Shapely's geometric operations to compute gradients. Also, the initialization method to assign room cells is different. I used the KMeans to converge the result faster than random initialization.

<br>

<div style="display: flex">
    <p align="center">
        <img src="free_form_floor_plan_design_using_differentiable_voronoi_diagram/runs/shape_a/11-26-2024__17-58-56/optimization.gif" width=20%>　　
        <img src="free_form_floor_plan_design_using_differentiable_voronoi_diagram/runs/shape_b/11-26-2024__13-39-24/optimization.gif" width=20%>　　
        <img src="free_form_floor_plan_design_using_differentiable_voronoi_diagram/runs/shape_c/11-26-2024__17-35-24/optimization.gif" width=20%>　　
        <img src="free_form_floor_plan_design_using_differentiable_voronoi_diagram/runs/shape_duck/11-26-2024__12-34-06/optimization.gif" width=20%>
    </p>
</div>
<p align="center" color="gray">
  <i>
  Optimization processes for <br>shape_a.py · shape_b.py · shape_c.py · shape_duck
  </i> 
</p>

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
- `shape_<name>/<datetime>`
    - `configs.json`: Used configs
    - `events.*`: Tensorboard logs
    - `optimization.gif`: Animation for optimizing the shape

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


<br>

# Future works
- Set entrances of the plan
- Set a graph-based constraint for the connectivity between rooms