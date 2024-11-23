import os
import sys
import time
import torch
import random
import shapely
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from typing import List, Callable
from shapely import geometry, ops
from torch.autograd.function import FunctionCtx

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from debugvisualizer.debugvisualizer import Plotter  # noqa: F401

from examples import examples


def runtime_calculator(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"The function {func.__name__} took {runtime} seconds to run.")
        return result

    return wrapper


class FloorPlanLoss(torch.autograd.Function):
    w_wall = 1.5
    w_area = 0.8
    w_lloyd = 1.2
    w_topo = 2.0

    @staticmethod
    def compute_wall_loss(walls: torch.Tensor, normalize=False):
        if normalize:
            loss_wall = torch.abs(walls / torch.norm(walls, dim=1).unsqueeze(1)).sum()
        else:
            loss_wall = torch.abs(walls).sum()

        loss_wall *= FloorPlanLoss.w_wall

        return loss_wall

    @staticmethod
    def compute_area_loss(
        cells: List[geometry.Polygon],
        target_areas: List[float],
        room_indices: List[int],
    ):
        current_areas = [0] * len(target_areas)

        for cell, room_index in zip(cells, room_indices):
            current_areas[room_index] += cell.area

        current_areas = torch.tensor(current_areas)
        target_areas = torch.tensor(target_areas)

        area_difference = current_areas - target_areas

        loss_area = torch.sum(area_difference)
        loss_area **= 2
        loss_area *= FloorPlanLoss.w_area

        return loss_area

    @staticmethod
    def compute_lloyd_loss(cells: List[geometry.Polygon], sites: torch.Tensor):
        loss_lloyd = 0
        for site, cell in zip(sites, cells):
            if cell.is_empty:
                continue
            site_point = geometry.Point(site.detach().numpy())
            loss_lloyd += site_point.distance(cell.centroid)

        loss_lloyd = torch.tensor(loss_lloyd)
        loss_lloyd **= 2
        loss_lloyd *= FloorPlanLoss.w_lloyd

        return loss_lloyd

    @staticmethod
    def compute_topology_loss(
        cells: List[geometry.Polygon],
        raw_cells: List[geometry.Polygon],
        sites_multipoint: geometry.MultiPoint,
        room_indices: List[int],
    ):
        rooms = [[] for _ in torch.tensor(room_indices).unique()]
        for cell, room_index in zip(cells, room_indices):
            rooms[room_index].append(cell)

        raw_rooms = [[] for _ in rooms]
        for raw_cell in raw_cells:
            buffered_raw_cell = raw_cell.buffer(1e-12)
            for ri, room in enumerate(rooms):
                found = False
                for r in room:
                    if buffered_raw_cell.contains(r):
                        raw_rooms[ri].append(raw_cell)
                        found = True
                        break
                if found:
                    break

        loss_topo = 0
        for room, raw_room in zip(rooms, raw_rooms):
            room_union = ops.unary_union(room)
            room_union = room_union.buffer(1e-12, join_style=geometry.JOIN_STYLE.mitre)
            room_union = room_union.buffer(-1e-12, join_style=geometry.JOIN_STYLE.mitre)

            if isinstance(room_union, geometry.MultiPolygon):
                largest_room, *other_rooms = sorted(room_union.geoms, key=lambda r: r.area, reverse=True)
                largest_room_sites = largest_room.intersection(sites_multipoint)

                raw_room_union = ops.unary_union(raw_room)
                raw_room_union = raw_room_union.buffer(1e-12, join_style=geometry.JOIN_STYLE.mitre)
                raw_room_union = raw_room_union.buffer(-1e-12, join_style=geometry.JOIN_STYLE.mitre)

                if isinstance(raw_room_union, geometry.MultiPolygon):
                    _, *other_raw_rooms = sorted(raw_room_union.geoms, key=lambda r: r.area, reverse=True)

                    for other_raw_room in other_raw_rooms:
                        other_room_site = other_raw_room.intersection(sites_multipoint)
                        s, t = ops.nearest_points(largest_room_sites, other_room_site)
                        loss_topo += s.distance(t)

                else:
                    print("fixing wip")

        loss_topo = torch.tensor(loss_topo)
        loss_topo **= 2
        loss_topo *= FloorPlanLoss.w_topo

        return loss_topo

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        sites: torch.Tensor,
        boundary_polygon: geometry.Polygon,
        target_areas: List[float],
        room_indices: List[int],
        save: bool = True,
    ) -> torch.Tensor:
        cells = []
        walls = []

        sites_multipoint = geometry.MultiPoint([tuple(point) for point in sites.detach().numpy()])
        raw_cells = list(shapely.voronoi_polygons(sites_multipoint, extend_to=boundary_polygon).geoms)
        for cell in raw_cells:
            intersected_cell = cell.intersection(boundary_polygon)

            intersected_cell_iter = [intersected_cell]
            if isinstance(intersected_cell, geometry.MultiPolygon):
                intersected_cell_iter = list(intersected_cell.geoms)

            for intersected_cell in intersected_cell_iter:
                exterior_coords = torch.tensor(intersected_cell.exterior.coords[:-1])
                exterior_coords_shifted = torch.roll(exterior_coords, shifts=-1, dims=0)
                walls.extend((exterior_coords - exterior_coords_shifted).tolist())
                cells.append(intersected_cell)

        cells_sorted = []
        raw_cells_sorted = []
        for site in sites:
            site_point = geometry.Point(site.detach().numpy())

            for ci, (cell, raw_cell) in enumerate(zip(cells, raw_cells)):
                if raw_cell.contains(site_point):
                    cells_sorted.append(cell)
                    raw_cells_sorted.append(raw_cell)
                    cells.pop(ci)
                    raw_cells.pop(ci)
                    break

        loss_wall = FloorPlanLoss.compute_wall_loss(torch.tensor(walls))
        loss_area = FloorPlanLoss.compute_area_loss(cells_sorted, target_areas, room_indices)
        loss_lloyd = FloorPlanLoss.compute_lloyd_loss(cells_sorted, sites)
        loss_topo = FloorPlanLoss.compute_topology_loss(cells_sorted, raw_cells_sorted, sites_multipoint, room_indices)

        if save:
            ctx.save_for_backward(sites)
            ctx.room_indices = room_indices
            ctx.target_areas = target_areas
            ctx.boundary_polygon = boundary_polygon

        return loss_wall + loss_area + loss_lloyd + loss_topo

    @staticmethod
    def _backward_one(args):
        sites, i, j, epsilon, boundary_polygon, target_areas, room_indices = args

        perturbed_sites_pos = sites.clone()
        perturbed_sites_neg = sites.clone()
        perturbed_sites_pos[i, j] += epsilon
        perturbed_sites_neg[i, j] -= epsilon

        loss_pos = FloorPlanLoss.forward(
            None, perturbed_sites_pos, boundary_polygon, target_areas, room_indices, save=False
        )

        loss_neg = FloorPlanLoss.forward(
            None, perturbed_sites_neg, boundary_polygon, target_areas, room_indices, save=False
        )

        return i, j, (loss_pos - loss_neg) / (2 * epsilon)

    @runtime_calculator
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor):
        (sites,) = ctx.saved_tensors
        room_indices = ctx.room_indices
        target_areas = ctx.target_areas
        boundary_polygon = ctx.boundary_polygon

        epsilon = 1e-6

        grads = torch.zeros_like(sites)

        multiprocessing_args = [
            (sites, i, j, epsilon, boundary_polygon, target_areas, room_indices)
            for i in range(sites.size(0))
            for j in range(sites.size(1))
        ]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(FloorPlanLoss._backward_one, multiprocessing_args)

        # Collect results
        for i, j, grad in results:
            grads[i, j] = grad

        return grads * grad_output, None, None, None, None


class FloorPlanGenerator(torch.nn.Module):
    def __init__(
        self,
        num_sites: int,
        boundary: np.ndarray,
        boundary_polygon: geometry.Polygon,
        target_areas: List[float],
        area_ratio: List[float],
    ):
        super().__init__()

        self.num_sites = num_sites
        self.boundary = boundary
        self.boundary_polygon = boundary_polygon
        self.target_areas = target_areas
        self.area_ratio = area_ratio

        self.sites = self._initialize_parameters()
        self.room_indices = random.choices(
            population=range(len(self.area_ratio)), weights=self.area_ratio, k=len(self.sites)
        )

    def _initialize_parameters(self):
        centroid = self.boundary_polygon.representative_point()
        centroid = torch.tensor(centroid.coords[0])

        sites = torch.rand((self.num_sites, 2))
        sites -= sites.mean(axis=0) - centroid

        for site in sites:
            if not geometry.Point(site.numpy()).within(self.boundary_polygon):
                vector = centroid - site
                norm = torch.norm(vector)

                t = torch.rand((1)).item()
                site += (vector / norm) * (norm + ((1 - t) * 0.05 + t * 0.15))

        for site in sites:
            assert geometry.Point(site.numpy()).within(self.boundary_polygon)

        return torch.nn.Parameter(sites)

    def sites_geom(self):
        return geometry.MultiPoint(self.sites.detach().numpy())

    def voronoi_geom(self):
        raw_cells = list(shapely.voronoi_polygons(self.sites_geom(), extend_to=self.boundary_polygon).geoms)
        cells = [vor.intersection(self.boundary_polygon) for vor in raw_cells]
        return cells, raw_cells

    def rooms_geom(self):
        rooms = [[] for _ in range(len(self.target_areas))]

        cells, raw_cells = self.voronoi_geom()

        for cell, raw_cell in zip(cells, raw_cells):
            if cell.is_empty:
                continue

            for room_index, site in zip(self.room_indices, self.sites_geom().geoms):
                if raw_cell.contains(site):
                    rooms[room_index].append(cell)
                    break

        return [ops.unary_union(r) for r in rooms]

    def visualize(self):
        plt.figure(figsize=(10, 10))

        rooms = self.rooms_geom()
        unique_indices = np.unique(self.room_indices)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_indices)))
        color_map = {idx: color for idx, color in zip(unique_indices, colors)}

        for room, idx in zip(rooms, self.room_indices):
            color = color_map[idx]
            if isinstance(room, geometry.MultiPolygon):
                for polygon in room.geoms:
                    plt.plot(*polygon.exterior.xy, color="black", linewidth=1.5)
                    plt.gca().add_patch(Polygon(polygon.exterior.coords, facecolor=color, alpha=0.5))
            else:
                plt.plot(*room.exterior.xy, color="black", linewidth=1.5)
                plt.gca().add_patch(Polygon(room.exterior.coords, facecolor=color, alpha=0.5))

        cells, _ = self.voronoi_geom()
        for cell in cells:
            if isinstance(cell, geometry.MultiPolygon):
                for polygon in cell.geoms:
                    plt.plot(*polygon.exterior.xy, color="gray", linewidth=0.5)
            else:
                plt.plot(*cell.exterior.xy, color="gray", linewidth=0.5)

        sites = self.sites.detach().numpy()
        plt.scatter(sites[:, 0], sites[:, 1], c="black", s=10)

        bounds = self.boundary_polygon.bounds
        plt.xlim(bounds[0], bounds[2])
        plt.ylim(bounds[1], bounds[3])

        plt.axis("equal")
        plt.title(f"Iteration {iteration}")
        plt.savefig(f"iteration_{iteration}.png")
        plt.close()


if __name__ == "__main__":
    seed = 777
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    shape = examples.ShapeA()

    area_ratio = [0.5, 0.2, 0.2, 0.1]
    target_areas = [shape.boundary_polygon.area * ratio for ratio in area_ratio]

    model = FloorPlanGenerator(
        num_sites=40,
        boundary=shape.coordinates,
        boundary_polygon=shape.boundary_polygon,
        target_areas=target_areas,
        area_ratio=area_ratio,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for iteration in range(501):
        if iteration == 250:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-3

        optimizer.zero_grad()
        loss = FloorPlanLoss.apply(model.sites, model.boundary_polygon, model.target_areas, model.room_indices)
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")
            model.visualize()
