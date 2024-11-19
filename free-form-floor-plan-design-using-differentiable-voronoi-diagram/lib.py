import time
import torch
import shapely
import numpy as np
import multiprocessing

from typing import List, Callable
from shapely import geometry, ops
from torch.autograd.function import FunctionCtx

from examples.examples import Duck


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
    A = "a"
    B = "b"
    C = "c"
    BC = B + C
    CC = C + C

    @staticmethod
    def compute_combination(
        point_1: geometry.Point,
        point_2: geometry.Point,
        boundary_exterior_buffered: geometry.Polygon,
        boundary_vertices_buffered: geometry.MultiPolygon,
    ):
        point_1_type = FloorPlanLoss.A
        if point_1.within(boundary_vertices_buffered):
            point_1_type = FloorPlanLoss.C
        elif point_1.within(boundary_exterior_buffered):
            point_1_type = FloorPlanLoss.B

        point_2_type = FloorPlanLoss.A
        if point_2.within(boundary_vertices_buffered):
            point_2_type = FloorPlanLoss.C
        elif point_2.within(boundary_exterior_buffered):
            point_2_type = FloorPlanLoss.B

        return "".join(sorted(point_1_type + point_2_type))

    @staticmethod
    def compute_wall_loss(wall_vectors: torch.Tensor, w_wall: float = 0.015):
        return torch.abs(wall_vectors).sum() * w_wall

    @staticmethod
    def compute_area_loss(
        cells: List[geometry.Polygon],
        sites: torch.Tensor,
        target_areas: List[float],
        room_indices: List[int],
        w_area: float = 1.0,
    ):
        current_areas = [0] * len(target_areas)
        for site, room_index in zip(sites, room_indices):
            site_point = geometry.Point(site.detach().numpy())

            for cell in cells:
                if cell.contains(site_point):
                    current_areas[room_index] += cell.area
                    break

        current_areas = torch.tensor(current_areas)
        target_areas = torch.tensor(target_areas)

        area_difference = current_areas - target_areas
        loss_area = torch.sum(area_difference**2) * w_area

        return loss_area

    @staticmethod
    def compute_lloyd_loss(cells: List[geometry.Polygon], sites: torch.Tensor, w_lloyd: float = 0.1):
        loss_lloyd = 0
        for site in sites:
            site_point = geometry.Point(site.detach().numpy())

            for cell in cells:
                if cell.contains(site_point):
                    loss_lloyd += site_point.distance(cell.centroid)
                    break

        loss_lloyd = torch.tensor(loss_lloyd**2) * w_lloyd

        return loss_lloyd

    @staticmethod
    def compute_topology_loss(
        cells: List[geometry.Polygon], sites: torch.Tensor, room_indices: List[int], w_topo: float = 1.0
    ):
        rooms = [[] for _ in torch.tensor(room_indices).unique()]
        for cell, room_index in zip(cells, room_indices):
            rooms[room_index].append(cell)

        loss_topo = 0
        for room in rooms:
            room_union = ops.unary_union(room)
            room_union = room_union.buffer(1e-12, join_style=geometry.JOIN_STYLE.mitre)
            room_union = room_union.buffer(-1e-12, join_style=geometry.JOIN_STYLE.mitre)

            if isinstance(room_union, geometry.MultiPolygon):
                sorted_room = sorted(room_union.geoms, key=lambda r: r.area, reverse=True)

                for r in sorted_room[1:]:
                    s, t = ops.nearest_points(sorted_room[0], r)
                    loss_topo += s.distance(t)

        loss_topo = torch.tensor(loss_topo**2) * w_topo

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
        boundary_exterior_buffered = boundary_polygon.exterior.buffer(1e-12)
        boundary_vertices_buffered = geometry.MultiPoint(boundary_polygon.exterior.coords).buffer(1e-12)

        cells = []
        wall_vectors = torch.tensor([])
        vertices = torch.tensor([])

        points = geometry.MultiPoint([tuple(point) for point in sites.detach().numpy()])
        for cell in shapely.voronoi_polygons(points, extend_to=boundary_polygon).geoms:
            intersected_cell = cell.intersection(boundary_polygon)

            intersected_cell_iter = [intersected_cell]
            if isinstance(intersected_cell, geometry.MultiPolygon):
                intersected_cell_iter = list(intersected_cell.geoms)

            for ic in intersected_cell_iter:
                ic_exterior_coords = torch.tensor(ic.exterior.coords[:-1])
                ic_exterior_coords_shifted = torch.cat([ic_exterior_coords[1:], ic_exterior_coords[0].unsqueeze(0)])

                for iec, iecs in zip(ic_exterior_coords, ic_exterior_coords_shifted):
                    points_combination = FloorPlanLoss.compute_combination(
                        point_1=geometry.Point(iec),
                        point_2=geometry.Point(iecs),
                        boundary_exterior_buffered=boundary_exterior_buffered,
                        boundary_vertices_buffered=boundary_vertices_buffered,
                    )

                    if points_combination in (FloorPlanLoss.BC, FloorPlanLoss.CC):
                        continue

                    wall_vector = iec - iecs
                    wall_vector = wall_vector.unsqueeze(0)
                    if len(wall_vectors) == 0:
                        wall_vectors = torch.cat([wall_vectors, wall_vector])
                        continue

                    if 2 in torch.isclose(wall_vector, wall_vectors).sum(dim=1):
                        continue

                    wall_vectors = torch.cat([wall_vectors, wall_vector])

                vertices = torch.cat([vertices, torch.tensor(ic.boundary.coords)])
                cells.append(ic)

        cells_sorted = []
        for site in sites:
            site_point = geometry.Point(site.detach().numpy())

            for ci, cell in enumerate(cells):
                if cell.contains(site_point):
                    cells_sorted.append(cell)
                    cells.pop(ci)
                    break

        loss_wall = FloorPlanLoss.compute_wall_loss(wall_vectors)
        loss_area = FloorPlanLoss.compute_area_loss(cells_sorted, sites, target_areas, room_indices)
        loss_lloyd = FloorPlanLoss.compute_lloyd_loss(cells_sorted, sites)
        loss_topo = FloorPlanLoss.compute_topology_loss(cells_sorted, sites, room_indices)

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
        self, num_sites: int, boundary: np.ndarray, boundary_polygon: geometry.Polygon, target_areas: List[float]
    ):
        super().__init__()

        self.num_sites = num_sites
        self.boundary = boundary
        self.boundary_polygon = boundary_polygon
        self.target_areas = target_areas

        self.sites = self._initialize_parameters()
        self.room_indices = torch.randint(low=0, high=len(target_areas), size=(len(self.sites),)).tolist()

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
        voronoi = shapely.voronoi_polygons(self.sites_geom(), extend_to=self.boundary_polygon)
        voronoi = [vor.intersection(self.boundary_polygon) for vor in voronoi.geoms]
        voronoi = [vor for vor in voronoi if not vor.is_empty]
        return voronoi

    def rooms_geom(self):
        rooms = [[] for _ in range(len(self.target_areas))]
        for cell in self.voronoi_geom():
            for room_index, site in zip(self.room_indices, self.sites_geom().geoms):
                if cell.contains(site):
                    rooms[room_index].append(cell)
                    break

        return [ops.unary_union(r) for r in rooms]


if __name__ == "__main__":
    import os
    import sys
    import random
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    seed = 777
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from debugvisualizer.debugvisualizer import Plotter  # noqa: F401

    duck = Duck()

    # Example usage
    area_ratio = [0.4, 0.3, 0.1, 0.1, 0.1]
    target_areas = [duck.boundary_polygon.area * ratio for ratio in area_ratio]
    model = FloorPlanGenerator(
        num_sites=40, boundary=duck.coordinates, boundary_polygon=duck.boundary_polygon, target_areas=target_areas
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for iteration in range(100):
        optimizer.zero_grad()
        loss = FloorPlanLoss.apply(model.sites, model.boundary_polygon, model.target_areas, model.room_indices)
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")

            # Create plot
            plt.figure(figsize=(10, 10))

            # Plot rooms with colors based on room_indices
            rooms = model.rooms_geom()
            unique_indices = np.unique(model.room_indices)  # Using numpy instead of torch
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_indices)))
            # Create color mapping dictionary
            color_map = {idx: color for idx, color in zip(unique_indices, colors)}

            for room, idx in zip(rooms, model.room_indices):
                color = color_map[idx]  # No need for .item() since we're using numpy
                # Handle both Polygon and MultiPolygon cases
                if isinstance(room, geometry.MultiPolygon):
                    for polygon in room.geoms:
                        plt.gca().add_patch(Polygon(polygon.exterior.coords, facecolor=color, alpha=0.5))
                else:
                    plt.gca().add_patch(Polygon(room.exterior.coords, facecolor=color, alpha=0.5))

            # Plot sites
            sites = model.sites.detach().numpy()
            plt.scatter(sites[:, 0], sites[:, 1], c="black", s=10)

            # Set plot limits based on boundary
            bounds = model.boundary_polygon.bounds
            plt.xlim(bounds[0], bounds[2])
            plt.ylim(bounds[1], bounds[3])

            plt.axis("equal")
            plt.title(f"Iteration {iteration}")
            plt.savefig(f"iteration_{iteration}.png")
            plt.close()
