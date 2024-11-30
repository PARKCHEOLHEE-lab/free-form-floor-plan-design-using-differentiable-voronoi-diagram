import time
import torch
import shapely
import multiprocessing

from typing import List, Callable
from shapely import geometry, ops
from torch.autograd.function import FunctionCtx


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
    @staticmethod
    def compute_wall_loss(rooms_group: List[List[geometry.Polygon]], w_wall: float = 1.0):
        loss_wall = 0.0
        for room_group in rooms_group:
            room_union = ops.unary_union(room_group)
            if isinstance(room_union, geometry.MultiPolygon):
                room_union = list(room_union.geoms)
            else:
                room_union = [room_union]

            for room in room_union:
                t1 = torch.tensor(room.exterior.coords[:-1])
                t2 = torch.roll(t1, shifts=-1, dims=0)
                loss_wall += torch.abs(t1 - t2).sum().item()

                for interior in room.interiors:
                    t1 = torch.tensor(interior.coords[:-1])
                    t2 = torch.roll(t1, shifts=-1, dims=0)
                    loss_wall += torch.abs(t1 - t2).sum().item()

        loss_wall = torch.tensor(loss_wall)
        loss_wall *= w_wall

        return loss_wall

    @staticmethod
    def compute_area_loss(
        cells: List[geometry.Polygon],
        target_areas: List[float],
        room_indices: List[int],
        w_area: float = 1.0,
    ):
        current_areas = [0.0] * len(target_areas)

        for cell, room_index in zip(cells, room_indices):
            current_areas[room_index] += cell.area

        current_areas = torch.tensor(current_areas)
        target_areas = torch.tensor(target_areas)

        area_difference = torch.abs(current_areas - target_areas)

        loss_area = torch.sum(area_difference)
        loss_area **= 2
        loss_area *= w_area

        return loss_area

    @staticmethod
    def compute_lloyd_loss(cells: List[geometry.Polygon], sites: torch.Tensor, w_lloyd: float = 1.0):
        valids = [(site.tolist(), cell) for site, cell in zip(sites, cells) if not cell.is_empty]
        valid_centroids = torch.tensor([cell.centroid.coords[0] for _, cell in valids])
        valid_sites = torch.tensor([site for site, _ in valids])

        loss_lloyd = torch.norm(valid_centroids - valid_sites, dim=1).sum()
        loss_lloyd **= 2
        loss_lloyd *= w_lloyd

        return loss_lloyd

    @staticmethod
    def compute_topology_loss(rooms_group: List[List[geometry.Polygon]], w_topo: float = 1.0):
        loss_topo = 0.0
        for room_group in rooms_group:
            room_union = ops.unary_union(room_group)
            if isinstance(room_union, geometry.MultiPolygon):
                largest_room, *_ = sorted(room_union.geoms, key=lambda r: r.area, reverse=True)

                loss_topo += len(room_union.geoms)

                for room in room_group:
                    if not room.intersects(largest_room) and not room.is_empty:
                        loss_topo += largest_room.centroid.distance(room)

        loss_topo = torch.tensor(loss_topo)
        loss_topo **= 2
        loss_topo *= w_topo

        return loss_topo

    @staticmethod
    def compute_bb_loss(rooms_group: List[List[geometry.Polygon]], w_bb: float = 1.0):
        loss_bb = 0.0
        for room_group in rooms_group:
            room_union = ops.unary_union(room_group)
            loss_bb += room_union.area / room_union.envelope.area

        loss_bb = torch.tensor(loss_bb)
        loss_bb **= 2
        loss_bb *= -w_bb

        return loss_bb

    @staticmethod
    def compute_cell_area_loss(cells: List[geometry.Polygon], w_cell: float = 1.0):
        sorted_cells = sorted(cells, key=lambda cell: cell.area)
        area_differences = [sorted_cells[i + 1].area - sorted_cells[i].area for i in range(len(sorted_cells) - 1)]

        loss_cell_area = torch.tensor(sum(area_differences))
        loss_cell_area *= w_cell

        return loss_cell_area

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        sites: torch.Tensor,
        boundary_polygon: geometry.Polygon,
        target_areas: List[float],
        room_indices: List[int],
        w_wall: float,
        w_area: float,
        w_lloyd: float,
        w_topo: float,
        w_bb: float,
        w_cell: float,
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
        for site_point in sites_multipoint.geoms:
            for ci, (cell, raw_cell) in enumerate(zip(cells, raw_cells)):
                if raw_cell.contains(site_point):
                    cells_sorted.append(cell)
                    cells.pop(ci)
                    raw_cells_sorted.append(raw_cell)
                    raw_cells.pop(ci)
                    break

        rooms_group = [[] for _ in torch.tensor(room_indices).unique()]
        for cell, room_index in zip(cells_sorted, room_indices):
            rooms_group[room_index].append(cell)

        loss_wall = torch.tensor(0.0)
        if w_wall > 0:
            loss_wall = FloorPlanLoss.compute_wall_loss(rooms_group, w_wall=w_wall)

        loss_area = torch.tensor(0.0)
        if w_area > 0:
            loss_area = FloorPlanLoss.compute_area_loss(cells_sorted, target_areas, room_indices, w_area=w_area)

        loss_lloyd = torch.tensor(0.0)
        if w_lloyd > 0:
            loss_lloyd = FloorPlanLoss.compute_lloyd_loss(cells_sorted, sites, w_lloyd=w_lloyd)

        loss_topo = torch.tensor(0.0)
        if w_topo > 0:
            loss_topo = FloorPlanLoss.compute_topology_loss(rooms_group, w_topo=w_topo)

        loss_bb = torch.tensor(0.0)
        if w_bb > 0:
            loss_bb = FloorPlanLoss.compute_bb_loss(rooms_group, w_bb=w_bb)

        loss_cell_area = torch.tensor(0.0)
        if w_cell > 0:
            loss_cell_area = FloorPlanLoss.compute_cell_area_loss(cells_sorted, w_cell=w_cell)

        if save:
            ctx.save_for_backward(sites)
            ctx.room_indices = room_indices
            ctx.target_areas = target_areas
            ctx.boundary_polygon = boundary_polygon
            ctx.w_wall = w_wall
            ctx.w_area = w_area
            ctx.w_lloyd = w_lloyd
            ctx.w_topo = w_topo
            ctx.w_bb = w_bb
            ctx.w_cell = w_cell

        loss = loss_wall + loss_area + loss_lloyd + loss_topo + loss_bb + loss_cell_area

        return loss, [loss_wall, loss_area, loss_lloyd, loss_topo, loss_bb, loss_cell_area]

    @staticmethod
    def _backward_one(args):
        (
            sites,
            i,
            j,
            epsilon,
            boundary_polygon,
            target_areas,
            room_indices,
            w_wall,
            w_area,
            w_lloyd,
            w_topo,
            w_bb,
            w_cell,
        ) = args

        perturbed_sites_pos = sites.clone()
        perturbed_sites_neg = sites.clone()
        perturbed_sites_pos[i, j] += epsilon
        perturbed_sites_neg[i, j] -= epsilon

        loss_pos, _ = FloorPlanLoss.forward(
            None,
            perturbed_sites_pos,
            boundary_polygon,
            target_areas,
            room_indices,
            w_wall,
            w_area,
            w_lloyd,
            w_topo,
            w_bb,
            w_cell,
            save=False,
        )

        loss_neg, _ = FloorPlanLoss.forward(
            None,
            perturbed_sites_neg,
            boundary_polygon,
            target_areas,
            room_indices,
            w_wall,
            w_area,
            w_lloyd,
            w_topo,
            w_bb,
            w_cell,
            save=False,
        )

        return i, j, (loss_pos - loss_neg) / (2 * epsilon)

    @runtime_calculator
    @staticmethod
    def backward(ctx: FunctionCtx, _: torch.Tensor, __):
        sites = ctx.saved_tensors[0]
        room_indices = ctx.room_indices
        target_areas = ctx.target_areas
        boundary_polygon = ctx.boundary_polygon
        w_wall = ctx.w_wall
        w_area = ctx.w_area
        w_lloyd = ctx.w_lloyd
        w_topo = ctx.w_topo
        w_bb = ctx.w_bb
        w_cell = ctx.w_cell

        epsilon = 1e-6

        grads = torch.zeros_like(sites)

        multiprocessing_args = [
            (
                sites,
                i,
                j,
                epsilon,
                boundary_polygon,
                target_areas,
                room_indices,
                w_wall,
                w_area,
                w_lloyd,
                w_topo,
                w_bb,
                w_cell,
            )
            for i in range(sites.size(0))
            for j in range(sites.size(1))
        ]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(FloorPlanLoss._backward_one, multiprocessing_args)

        for i, j, grad in results:
            grads[i, j] = grad

        return grads, None, None, None, None, None, None, None, None, None, None
