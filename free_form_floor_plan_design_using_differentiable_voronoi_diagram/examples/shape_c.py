import os
import sys
import pytz
import torch
import random
import datetime
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from free_form_floor_plan_design_using_differentiable_voronoi_diagram.src import shape
from free_form_floor_plan_design_using_differentiable_voronoi_diagram.src.loss import FloorPlanLoss
from free_form_floor_plan_design_using_differentiable_voronoi_diagram.src.generator import FloorPlanGenerator


seed = 777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

configs = {
    "shape": shape.ShapeC(),
    "num_sites": 25,
    "area_ratio": [0.5, 0.2, 0.2, 0.1],
    "w_wall": 2.5,
    "w_area": 6.0,
    "w_lloyd": 2.3,
    "w_topo": 1.3,
    "w_bb": 0.55,
    "iterations": 700,
    "iteration_to_modify_lr": 300,
    "lr_initial": 1e-2,
    "lr_modified": 5e-3,
    "log_dir": os.path.abspath(
        os.path.join(
            __file__, "../../runs", datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%Y__%H-%M-%S")
        )
    ),
}

generator = FloorPlanGenerator(configs=configs)
optimizer = torch.optim.AdamW(generator.parameters(), lr=configs["lr_initial"])

for iteration in range(1, configs["iterations"] + 1):
    if iteration == configs["iteration_to_modify_lr"]:
        for param_group in optimizer.param_groups:
            param_group["lr"] = configs["lr_modified"]

    optimizer.zero_grad()
    loss, (loss_wall, loss_area, loss_lloyd, loss_topo, loss_bb) = FloorPlanLoss.apply(
        generator.sites,
        generator.boundary_polygon,
        generator.target_areas,
        generator.room_indices,
        generator.configs["w_wall"],
        generator.configs["w_area"],
        generator.configs["w_lloyd"],
        generator.configs["w_topo"],
        generator.configs["w_bb"],
    )

    loss.backward()
    optimizer.step()
    generator.log(iteration, loss, loss_wall, loss_area, loss_lloyd, loss_topo, loss_bb)

generator.gif()