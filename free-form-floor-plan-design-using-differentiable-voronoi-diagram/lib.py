import torch
import shapely
import numpy as np

from shapely import geometry
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

from examples.examples import Duck


class VoronoiWithGrad(Function):
    @staticmethod
    def _get_points_combination(
        point_1: geometry.Point,
        point_2: geometry.Point,
        boundary_exterior_buffered: geometry.Polygon,
        boundary_vertices_buffered: geometry.MultiPolygon,
    ):
        point_1_type = "a"
        if point_1.within(boundary_vertices_buffered):
            point_1_type = "c"
        elif point_1.within(boundary_exterior_buffered):
            point_1_type = "b"

        point_2_type = "a"
        if point_2.within(boundary_vertices_buffered):
            point_2_type = "c"
        elif point_2.within(boundary_exterior_buffered):
            point_2_type = "b"

        return "".join(sorted(point_1_type + point_2_type))

    @staticmethod
    def forward(ctx: FunctionCtx, sites: torch.Tensor, boundary: geometry.Polygon) -> torch.Tensor:
        boundary_exterior_buffered = boundary.exterior.buffer(1e-12)
        boundary_vertices_buffered = geometry.MultiPoint(boundary.exterior.coords).buffer(1e-12)

        points = geometry.MultiPoint([tuple(point) for point in sites.numpy()])

        wall_vectors = []

        vertices = torch.tensor([])
        for cell in shapely.voronoi_polygons(points).geoms:
            intersected_cell = cell.intersection(boundary)

            intersected_cell_iter = [intersected_cell]
            if isinstance(intersected_cell, geometry.MultiPolygon):
                intersected_cell_iter = list(intersected_cell.geoms)

            # point_a: is in boundary
            # point_b: is on boundary
            # point_c: is a vertex of boundary

            for ic in intersected_cell_iter:
                ic_exterior_coords = ic.exterior.coords[:-1]
                ic_exterior_coords_shifted = [*list(ic.exterior.coords[1:]), ic_exterior_coords[0]]

                for iec, iecs in zip(ic_exterior_coords, ic_exterior_coords_shifted):
                    points_combination = VoronoiWithGrad._get_points_combination(
                        point_1=geometry.Point(iec),
                        point_2=geometry.Point(iecs),
                        boundary_exterior_buffered=boundary_exterior_buffered,
                        boundary_vertices_buffered=boundary_vertices_buffered,
                    )

                    if points_combination in ("bc", "cc"):
                        continue

                    wall_vector = torch.tensor(iec) - torch.tensor(iecs)
                    wall_vectors.append(wall_vector)

                vertices = torch.cat([vertices, torch.tensor(ic.boundary.coords)])

        ctx.save_for_backward(sites)

        return vertices

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        (sites,) = ctx.saved_tensors
        epsilon = 1e-5
        grad_sites = torch.zeros_like(sites)

        for i in range(sites.size(0)):
            for j in range(sites.size(1)):
                sites_pos, sites_neg = sites.clone(), sites.clone()
                sites_pos[i, j] += epsilon
                sites_neg[i, j] -= epsilon

                points_pos = geometry.MultiPoint([tuple(point) for point in sites_pos.numpy()])
                points_neg = geometry.MultiPoint([tuple(point) for point in sites_neg.numpy()])

                voronoi_cells_pos = shapely.voronoi_polygons(points_pos)
                voronoi_cells_neg = shapely.voronoi_polygons(points_neg)

                vertices_pos = torch.tensor(
                    [list(poly.centroid.coords[0]) for poly in voronoi_cells_pos.geoms], dtype=torch.float32
                )
                vertices_neg = torch.tensor(
                    [list(poly.centroid.coords[0]) for poly in voronoi_cells_neg.geoms], dtype=torch.float32
                )

                grad_sites[i, j] = torch.sum((vertices_pos - vertices_neg) * grad_output) / (2 * epsilon)

        return grad_sites, None


class Agent(torch.nn.Module):
    def __init__(self, num_sites: int, boundary: np.ndarray):
        super().__init__()

        self.num_sites = num_sites
        self.boundary = boundary
        self.boundary_polygon = geometry.Polygon(boundary)

        self._initialize_parameters()

    def _initialize_parameters(self):
        centroid = self.boundary_polygon.representative_point()
        centroid = torch.tensor(centroid.coords[0])

        sites = torch.rand((self.num_sites, 2))
        sites -= sites.mean(axis=0) - centroid

        for site in sites:
            if not geometry.Point(site.numpy()).within(self.boundary_polygon):
                vector = centroid - site
                norm = torch.norm(vector)

                t = torch.rand(
                    1,
                ).item()
                site += (vector / norm) * (norm + ((1 - t) * 0.05 + t * 0.15))

        for site in sites:
            assert geometry.Point(site.numpy()).within(self.boundary_polygon)

        self.sites = torch.nn.Parameter(sites)

    def forward(self):
        vertices = VoronoiWithGrad.apply(self.sites, self.boundary_polygon)
        return vertices


def wall_loss(vertices: torch.Tensor, boundary: geometry.Polygon, w_wall: float = 1.0) -> torch.Tensor:
    num_vertices = vertices.size(0)
    wall_loss = 0.0

    for i in range(num_vertices - 1):
        for j in range(i + 1, num_vertices):  # Check each pair of vertices
            edge_length = torch.sum(torch.abs(vertices[i] - vertices[j]))  # L1 distance
            wall_loss += edge_length

    wall_loss *= w_wall

    return wall_loss


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from debugvisualizer.debugvisualizer import Plotter  # noqa: F401

    duck = Duck()

    # Example usage
    model = Agent(num_sites=50, boundary=duck.coordinates)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for iteration in range(100):
        optimizer.zero_grad()

        # Forward pass with bounding box
        vertices = model()

        loss = wall_loss(vertices, model.boundary_polygon, w_wall=0.1)
        loss.backward()  # Backpropagate

        # Update sites
        optimizer.step()

        # Print loss
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")
