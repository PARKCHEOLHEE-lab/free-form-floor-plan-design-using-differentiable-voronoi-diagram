"""Head-to-head benchmark for the Python original.

Loads an example's golden-fixture start (the same initial sites + room
indices the Rust port uses), runs N optimization iterations timing ONLY the
compute (FloorPlanLoss forward + finite-difference backward + AdamW step,
excluding rendering/IO), renders one frame per iteration with the original
matplotlib plot, and emits python_<name>.gif + python_timing_<name>.json.

The backward pass spawns a multiprocessing pool every iteration exactly as
loss.py does (start method forced to "fork" so shapely geometry pickles on
macOS as it would on the author's Linux devcontainer) — that overhead is part
of the algorithm as written and is therefore timed.

Usage (from repo root, in the fixture venv):
  .venv-fixtures/bin/python comparison/bench_python.py <example> <iters> <out_dir>
"""

import io
import os
import sys
import json
import time
import multiprocessing

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

multiprocessing.set_start_method("fork", force=True)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from free_form_floor_plan_design_using_differentiable_voronoi_diagram.src import shape  # noqa: E402
from free_form_floor_plan_design_using_differentiable_voronoi_diagram.src.loss import FloorPlanLoss  # noqa: E402
from free_form_floor_plan_design_using_differentiable_voronoi_diagram.src.generator import FloorPlanGenerator  # noqa: E402

SHAPES = {
    "shape_a": shape.ShapeA,
    "shape_b": shape.ShapeB,
    "shape_c": shape.ShapeC,
    "shape_duck": shape.Duck,
}


def main():
    name = sys.argv[1]
    iters = int(sys.argv[2])
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    fx = json.load(open(os.path.join(REPO, "rust", "fixtures", f"{name}.checkpoint.json")))
    sites0 = np.array(fx["initial_sites"], dtype=np.float32)
    rooms = fx["room_indices"]
    ratio = fx["config"]["area_ratio"]

    configs = {
        "shape": SHAPES[name](),
        "num_sites": 40,
        "area_ratio": ratio,
        "w_wall": 2.5,
        "w_area": 20.0,
        "w_lloyd": 2.1,
        "w_topo": 1.5,
        "w_bb": 0.0,
        "w_cell": 0.0,
        "init_with_kmeans": True,
        "iterations": iters,
        "iteration_to_modify_lr": 300,
        "lr_initial": 1e-2,
        "lr_modified": 8e-3,
        "log_dir": os.path.join(out_dir, f"_pygen_{name}"),
    }
    gen = FloorPlanGenerator(configs=configs)
    # overwrite the generator's own random init with the shared fixture start
    with torch.no_grad():
        gen.sites.copy_(torch.tensor(sites0))
    gen.room_indices = rooms

    opt = torch.optim.AdamW(gen.parameters(), lr=1e-2)

    per_iter_ms = []
    frames = []
    last_loss = 0.0
    for it in range(1, iters + 1):
        if it == 300:
            for g in opt.param_groups:
                g["lr"] = 8e-3

        # --- timed: one optimization iteration (forward + backward + step) ---
        t = time.perf_counter()
        opt.zero_grad()
        loss, _ = FloorPlanLoss.apply(
            gen.sites,
            gen.boundary_polygon,
            gen.target_areas,
            gen.room_indices,
            2.5,
            20.0,
            2.1,
            1.5,
            0.0,
            0.0,
        )
        loss.backward()
        opt.step()
        per_iter_ms.append((time.perf_counter() - t) * 1e3)
        last_loss = float(loss.item())

        # --- untimed: render the post-step state with the original plot ---
        fig = gen.plot()
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=60)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB").copy())
        plt.close(fig)

    frames[0].save(
        os.path.join(out_dir, f"python_{name}.gif"),
        save_all=True,
        append_images=frames[1:],
        duration=20,
        loop=0,
    )

    total_ms = sum(per_iter_ms)
    timing = {
        "impl": "python",
        "example": name,
        "iterations": iters,
        "total_compute_s": total_ms / 1e3,
        "mean_iter_ms": total_ms / iters,
        "per_iter_ms": per_iter_ms,
        "last_iter_loss": last_loss,
        "threads": os.cpu_count(),
    }
    json.dump(timing, open(os.path.join(out_dir, f"python_timing_{name}.json"), "w"), indent=2)
    print(f"python {name}: {iters} iters, total {total_ms/1e3:.3f}s, mean {total_ms/iters:.3f} ms/iter", flush=True)


if __name__ == "__main__":
    main()
