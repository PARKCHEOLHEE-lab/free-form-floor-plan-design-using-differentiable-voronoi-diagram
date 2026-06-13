"""Golden-fixture exporter for the Rust port.

Runs the original Python implementation once per example and dumps, as JSON:

  <name>.checkpoint.json  shape geometry, initial sites (exact f32), room indices,
                          iteration-0 cell geometry / losses / gradients, GEOS cell
                          ordering, and a 10-iteration loss trace
  <name>.final.json       final sites / loss / per-room outcome after the full
                          configured run (default 800 iterations)

Numbers are exported exactly: float32 tensors are widened to f64 (lossless) and
Python's json emits shortest-round-trip decimals, so every value reloads
bit-identically.

GEOS cell ordering: shapely.voronoi_polygons returns cells in a GEOS-internal
order. loss.py's forward pairs clipped pieces with raw cells positionally
(zip), so whenever a clipped cell is a MultiPolygon the pairing after that
position shifts by one and the last piece is dropped. Reproducing those exact
numbers requires the raw-cell order, which is exported per compared iteration
as `geos_cell_order` (raw position -> generating site index).

Environment: python 3.11 venv with the repo's pinned requirements
(Pillow relaxed to >=9.3,<10 for python 3.11 wheels; Pillow contributes to no
exported number - it is image I/O only).

Usage:  .venv-fixtures/bin/python free_form_floor_plan_design_using_differentiable_voronoi_diagram/rust/fixtures/export_fixtures.py [example ...]
"""

import os
import sys
import json
import random
import multiprocessing

import numpy as np
import torch
import shapely
from shapely import geometry, ops

multiprocessing.set_start_method("fork", force=True)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

from free_form_floor_plan_design_using_differentiable_voronoi_diagram.python.src import shape  # noqa: E402
from free_form_floor_plan_design_using_differentiable_voronoi_diagram.python.src.loss import FloorPlanLoss  # noqa: E402
from free_form_floor_plan_design_using_differentiable_voronoi_diagram.python.src.generator import FloorPlanGenerator  # noqa: E402

assert shapely.__version__ == "2.0.2", shapely.__version__
assert torch.__version__.startswith("2.1.0"), torch.__version__

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 777
TRACE_ITERS = 10

EXAMPLES = {
    "shape_a": (shape.ShapeA, [0.5, 0.3, 0.1, 0.1]),
    "shape_b": (shape.ShapeB, [0.5, 0.2, 0.1, 0.1, 0.1]),
    "shape_c": (shape.ShapeC, [0.4, 0.3, 0.2, 0.1]),
    "shape_duck": (shape.Duck, [0.2, 0.2, 0.2, 0.2, 0.2]),
}


def make_configs(name, shape_cls, area_ratio):
    return {
        "shape": shape_cls(),
        "num_sites": 40,
        "area_ratio": area_ratio,
        "w_wall": 2.5,
        "w_area": 20.0,
        "w_lloyd": 2.1,
        "w_topo": 1.5,
        "w_bb": 0.0,
        "w_cell": 0.0,
        "init_with_kmeans": True,
        "iterations": 800,
        "iteration_to_modify_lr": 300,
        "lr_initial": 1e-2,
        "lr_modified": 8e-3,
        "log_dir": f"/tmp/fixture_runs/{name}",
    }


def f32s(tensor):
    """Exact f32 values as Python floats (lossless f32->f64 widening)."""
    return [[float(v) for v in row] for row in tensor.detach().numpy().astype(np.float32)]


def geos_cell_order(sites_mp, raw_cells):
    """Permutation: raw cell position -> generating site index (containment)."""
    order = []
    for rc in raw_cells:
        owners = [si for si, sp in enumerate(sites_mp.geoms) if rc.contains(sp)]
        assert len(owners) == 1, f"non-unique cell ownership: {owners}"
        order.append(owners[0])
    return order


def voronoi_raw(sites_t, boundary):
    sites_mp = geometry.MultiPoint([tuple(p) for p in sites_t.detach().numpy()])
    raw_cells = list(shapely.voronoi_polygons(sites_mp, extend_to=boundary).geoms)
    return sites_mp, raw_cells


def cells_sorted_literal(sites_mp, raw_cells, boundary):
    """Replicates loss.py forward's cell construction + zip-pop pairing exactly.

    split_pieces records, per splitting raw position, the piece areas in GEOS
    iteration order — the pairing downstream depends on that order, so the
    Rust side reproduces it from this data.
    """
    cells = []
    split_positions = []
    split_pieces = []
    for ci, cell in enumerate(raw_cells):
        inter = cell.intersection(boundary)
        pieces = list(inter.geoms) if isinstance(inter, geometry.MultiPolygon) else [inter]
        if len(pieces) > 1:
            split_positions.append([ci, len(pieces)])
            split_pieces.append([ci, [float(p.area) for p in pieces]])
        cells.extend(pieces)
    n_pieces = len(cells)

    cells2, raw2 = list(cells), list(raw_cells)
    cells_sorted = []
    for sp in sites_mp.geoms:
        for ci, (cell, rc) in enumerate(zip(cells2, raw2)):
            if rc.contains(sp):
                cells_sorted.append(cell)
                cells2.pop(ci)
                raw2.pop(ci)
                break
    return cells_sorted, n_pieces, split_positions, split_pieces


def forward_loss(sites_t, boundary, target_areas, room_indices, c, w_bb=None, w_cell=None):
    loss, comps = FloorPlanLoss.forward(
        None,
        sites_t,
        boundary,
        target_areas,
        room_indices,
        c["w_wall"],
        c["w_area"],
        c["w_lloyd"],
        c["w_topo"],
        c["w_bb"] if w_bb is None else w_bb,
        c["w_cell"] if w_cell is None else w_cell,
        save=False,
    )
    names = ["wall", "area", "lloyd", "topo", "bb", "cell"]
    return float(loss.item()), {n: float(t.item()) for n, t in zip(names, comps)}


def rooms_outcome(gen):
    """Final-plan outcome via generator.rooms_geom semantics (correct 1:1 pairing)."""
    rooms = gen.rooms_geom()
    areas, components = [], []
    for room in rooms:
        areas.append(float(room.area))
        if room.is_empty:
            components.append(0)
        elif isinstance(room, geometry.MultiPolygon):
            components.append(len(room.geoms))
        else:
            components.append(1)
    return areas, components


def export(name):
    shape_cls, area_ratio = EXAMPLES[name]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    configs = make_configs(name, shape_cls, area_ratio)
    gen = FloorPlanGenerator(configs=configs)
    boundary = gen.boundary_polygon

    exterior = list(boundary.exterior.coords[:-1])
    fixture = {
        "meta": {
            "example": name,
            "seed": SEED,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "shapely": shapely.__version__,
            "numpy": np.__version__,
        },
        "config": {k: v for k, v in configs.items() if k not in ("shape", "log_dir")},
        "shape": {
            "boundary_coords": [[float(x), float(y)] for x, y in exterior],
            "area": float(boundary.area),
            "perimeter": float(boundary.length),
        },
        "target_areas": [float(a) for a in gen.target_areas],
        "initial_sites": f32s(gen.sites),
        "room_indices": list(gen.room_indices),
    }

    # ---- iteration 0 checkpoint ----
    sites0 = gen.sites.detach()
    sites_mp, raw_cells = voronoi_raw(sites0, boundary)
    order0 = geos_cell_order(sites_mp, raw_cells)
    cells_sorted, n_pieces, split_positions, split_pieces = cells_sorted_literal(
        sites_mp, raw_cells, boundary
    )

    total, comps = forward_loss(sites0, boundary, gen.target_areas, gen.room_indices, configs)
    _, comps_w1 = forward_loss(
        sites0, boundary, gen.target_areas, gen.room_indices, configs, w_bb=1.0, w_cell=1.0
    )

    # gradient via the real autograd path (fork-pool backward)
    opt = torch.optim.AdamW(gen.parameters(), lr=configs["lr_initial"])
    opt.zero_grad()
    loss_t, _ = FloorPlanLoss.apply(
        gen.sites, boundary, gen.target_areas, gen.room_indices,
        configs["w_wall"], configs["w_area"], configs["w_lloyd"],
        configs["w_topo"], configs["w_bb"], configs["w_cell"],
    )
    loss_t.backward()
    grads = f32s(gen.sites.grad)

    # perturbation-order stability: does the GEOS order change under +-eps?
    eps = 1e-6
    n_diff = 0
    perturbed_orders = {}
    for i in range(sites0.size(0)):
        for j in range(2):
            for sign, delta in (("+", eps), ("-", -eps)):
                p = sites0.clone()
                p[i, j] += delta
                mp_p, raw_p = voronoi_raw(p, boundary)
                order_p = geos_cell_order(mp_p, raw_p)
                if order_p != order0:
                    n_diff += 1
                    perturbed_orders[f"{i},{j},{sign}"] = order_p

    fixture["iter0"] = {
        "geos_cell_order": order0,
        "n_raw_cells": len(raw_cells),
        "n_pieces": n_pieces,
        "split_positions": split_positions,
        "split_pieces": split_pieces,
        "cells_sorted": [
            {
                "area": float(c.area),
                "centroid": None if c.is_empty else [float(c.centroid.x), float(c.centroid.y)],
                "is_empty": bool(c.is_empty),
            }
            for c in cells_sorted
        ],
        "losses": dict(comps, total=total),
        "losses_bbcell_w1": {"bb": comps_w1["bb"], "cell": comps_w1["cell"]},
        "grads": grads,
        "grad_order_stability": {"n_checked": sites0.size(0) * 2 * 2, "n_diff": n_diff},
        "perturbed_orders": perturbed_orders or None,
    }

    # ---- optimization run: trace (1..TRACE_ITERS) then continue to the end ----
    trace = []
    for iteration in range(1, configs["iterations"] + 1):
        if iteration == configs["iteration_to_modify_lr"]:
            for group in opt.param_groups:
                group["lr"] = configs["lr_modified"]

        opt.zero_grad()
        loss_t, _ = FloorPlanLoss.apply(
            gen.sites, boundary, gen.target_areas, gen.room_indices,
            configs["w_wall"], configs["w_area"], configs["w_lloyd"],
            configs["w_topo"], configs["w_bb"], configs["w_cell"],
        )

        if iteration <= TRACE_ITERS:
            mp_k, raw_k = voronoi_raw(gen.sites.detach(), boundary)
            _, _, _, split_pieces_k = cells_sorted_literal(mp_k, raw_k, boundary)
            trace.append(
                {
                    "iteration": iteration,
                    "loss": float(loss_t.item()),
                    "geos_cell_order": geos_cell_order(mp_k, raw_k),
                    "split_pieces": split_pieces_k,
                }
            )
            if iteration == TRACE_ITERS:
                fixture["trace"] = trace
                cp_path = os.path.join(OUT_DIR, f"{name}.checkpoint.json")
                with open(cp_path, "w") as f:
                    json.dump(fixture, f)
                print(f"[{name}] checkpoint fixture written: {cp_path}", flush=True)
                if CHECKPOINT_ONLY:
                    print(f"[{name}] checkpoint-only mode: stopping", flush=True)
                    return

        loss_t.backward()
        opt.step()

        if iteration % 50 == 0:
            print(f"[{name}] iteration {iteration}, loss {loss_t.item()}", flush=True)

    final_loss, _ = forward_loss(
        gen.sites.detach(), boundary, gen.target_areas, gen.room_indices, configs
    )
    room_areas, room_components = rooms_outcome(gen)
    final = {
        "meta": fixture["meta"],
        "iterations": configs["iterations"],
        "final_loss": final_loss,
        "final_sites": f32s(gen.sites),
        "room_areas": room_areas,
        "room_area_shares": [a / float(boundary.area) for a in room_areas],
        "room_component_counts": room_components,
        "boundary_area": float(boundary.area),
    }
    fin_path = os.path.join(OUT_DIR, f"{name}.final.json")
    with open(fin_path, "w") as f:
        json.dump(final, f)
    print(f"[{name}] final fixture written: {fin_path}", flush=True)


CHECKPOINT_ONLY = False

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--checkpoint-only"]
    CHECKPOINT_ONLY = "--checkpoint-only" in sys.argv[1:]
    targets = args or list(EXAMPLES)
    for t in targets:
        export(t)
    print("ALL DONE", flush=True)
