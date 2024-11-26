import os
import io
import json
import torch
import random
import shapely
import numpy as np
import matplotlib.pyplot as plt

from torch_kmeans import KMeans
from PIL import Image
from shapely import ops
from shapely import geometry
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from torch.utils.tensorboard import SummaryWriter


class FloorPlanGenerator(torch.nn.Module):
    def __init__(self, configs: dict):
        super().__init__()

        self.configs = configs
        self.num_sites = self.configs["num_sites"]
        self.boundary_polygon = self.configs["shape"].boundary_polygon
        self.area_ratio = self.configs["area_ratio"]
        self.log_dir = self.configs["log_dir"]
        self.target_areas = [self.boundary_polygon.area * ratio for ratio in self.area_ratio]

        self.sites = self._initialize_parameters()

        if self.configs["init_with_kmeans"]:
            kmeans = KMeans(n_clusters=len(self.area_ratio))
            self.room_indices = kmeans(self.sites.unsqueeze(0)).labels.squeeze(0).tolist()
        else:
            self.room_indices = random.choices(
                population=range(len(self.area_ratio)), weights=self.area_ratio, k=len(self.sites)
            )

        self.figs = []
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)

        configs_serializable = {}
        for key, value in self.configs.items():
            try:
                json.dumps(value)
            except TypeError:
                value = str(value)

            configs_serializable[key] = value

        configs_path = os.path.join(self.log_dir, "configs.json")
        with open(configs_path, "w") as f:
            json.dump(configs_serializable, f, indent=4)

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

    def plot(self, iteration):
        fig, ax = plt.subplots(figsize=(10, 10))

        rooms = self.rooms_geom()
        unique_indices = np.unique(self.room_indices)
        colors = [plt.cm.Accent(i) for i in range(len(unique_indices))]
        color_map = {ci: color for ci, color in zip(unique_indices, colors)}

        for ri, room in enumerate(rooms):
            color = color_map[ri]
            if isinstance(room, geometry.MultiPolygon):
                for polygon in room.geoms:
                    path = Path.make_compound_path(
                        Path(np.asarray(polygon.exterior.coords)[:, :2]),
                        *[Path(np.asarray(ring.coords)[:, :2]) for ring in polygon.interiors],
                    )

                    patch = PathPatch(path, edgecolor=color, linewidth=1.5, facecolor=color, alpha=0.5)
                    ax.add_patch(patch)
                    ax.plot(*polygon.exterior.xy, color="black", linewidth=1.5)
                    for interior in polygon.interiors:
                        ax.plot(*interior.xy, color="black", linewidth=1.5)

            else:
                path = Path.make_compound_path(
                    Path(np.asarray(room.exterior.coords)[:, :2]),
                    *[Path(np.asarray(ring.coords)[:, :2]) for ring in room.interiors],
                )

                patch = PathPatch(path, edgecolor=color, linewidth=1.5, facecolor=color, alpha=0.5)
                ax.add_patch(patch)
                ax.plot(*room.exterior.xy, color="black", linewidth=1.5)
                for interior in room.interiors:
                    ax.plot(*interior.xy, color="black", linewidth=1.5)

        cells, _ = self.voronoi_geom()
        for cell in cells:
            if isinstance(cell, geometry.MultiPolygon):
                for polygon in cell.geoms:
                    ax.plot(*polygon.exterior.xy, color="gray", linewidth=0.3)
            else:
                ax.plot(*cell.exterior.xy, color="gray", linewidth=0.3)

        sites = self.sites.detach().numpy()
        ax.scatter(sites[:, 0], sites[:, 1], c="black", s=10)

        bounds = self.boundary_polygon.bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_axis_off()

        ax.axis("equal")

        return fig

    def log(self, iteration, loss, loss_wall, loss_area, loss_lloyd, loss_topo, loss_bb, loss_cell_area):
        self.summary_writer.add_scalar("loss", loss, iteration)
        self.summary_writer.add_scalar("loss_wall", loss_wall, iteration)
        self.summary_writer.add_scalar("loss_area", loss_area, iteration)
        self.summary_writer.add_scalar("loss_lloyd", loss_lloyd, iteration)
        self.summary_writer.add_scalar("loss_topo", loss_topo, iteration)
        self.summary_writer.add_scalar("loss_bb", loss_bb, iteration)
        self.summary_writer.add_scalar("loss_cell_area", loss_cell_area, iteration)

        fig = self.plot(iteration)
        fig.canvas.draw()
        self.summary_writer.add_image(
            f"floor_plan_{iteration}", np.array(fig.canvas.renderer._renderer), iteration, dataformats="HWC"
        )
        self.figs.append(fig)
        plt.close(fig)

        print(f"Iteration {iteration}, Loss: {loss.item()}")

    def gif(self):
        print("creating .gif")

        images = []
        for fig in self.figs:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            images.append(Image.open(buf))

        gif_path = os.path.join(self.log_dir, "optimization.gif")
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=20,
        )
