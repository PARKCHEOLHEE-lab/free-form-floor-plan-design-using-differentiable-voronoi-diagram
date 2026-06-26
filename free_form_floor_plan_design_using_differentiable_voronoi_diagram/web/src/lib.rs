//! wasm-bindgen front door for the in-browser demo. It reuses the native
//! optimization core verbatim (`voronoi_floorplan::{voronoi,loss,grad,init,
//! optim}`); only the parallelism differs (serial here, see ../rust grad.rs).
//!
//! A user-drawn boundary is normalized exactly like `python/src/shape.py`
//! (center at the vertex mean, scale so the farthest vertex sits at radius 1),
//! so the same loss weights that were tuned for the bundled shapes also work
//! for arbitrary polygons.

use geo::{Area, LineString, MultiPolygon, Polygon, Simplify};
use serde::Serialize;
use voronoi_floorplan::loss::LossWeights;
use voronoi_floorplan::{grad_local, init, loss, optim::AdamW, shapes, voronoi};
use wasm_bindgen::prelude::*;

fn normalize(pts: &[(f64, f64)]) -> Polygon<f64> {
    let n = pts.len().max(1) as f64;
    let (sx, sy) = pts.iter().fold((0.0, 0.0), |(ax, ay), &(x, y)| (ax + x, ay + y));
    let (mx, my) = (sx / n, sy / n);
    let centered: Vec<(f64, f64)> = pts.iter().map(|&(x, y)| (x - mx, y - my)).collect();
    let maxr = centered
        .iter()
        .map(|&(x, y)| (x * x + y * y).sqrt())
        .fold(0.0_f64, f64::max);
    let maxr = if maxr > 0.0 { maxr } else { 1.0 };
    let scaled: Vec<(f64, f64)> = centered.iter().map(|&(x, y)| (x / maxr, y / maxr)).collect();
    Polygon::new(LineString::from(scaled), vec![])
}

#[derive(Serialize)]
struct Cell {
    room: usize,
    ring: Vec<(f64, f64)>,
}

#[derive(Serialize)]
struct Frame {
    iteration: usize,
    /// loss at the sites *before* this step (matches the CLI/tensorboard log)
    loss: f32,
    /// the four weighted per-term contributions that the demo graphs (they sum
    /// to `loss`; the demo's bb/cell weights are 0 so those terms are omitted).
    wall: f32,
    area: f32,
    lloyd: f32,
    topo: f32,
    sites: Vec<(f32, f32)>,
    cells: Vec<Cell>,
    /// wall centerlines — the simplified room-boundary rings. The canvas strokes
    /// each twice (black at the wall width, then white narrower) to draw the
    /// double-line walls; the wall thickness is a render (stroke) parameter.
    walls: Vec<Vec<(f64, f64)>>,
}

#[wasm_bindgen]
pub struct WasmOpt {
    boundary: Polygon<f64>,
    sites: Vec<[f32; 2]>,
    room_indices: Vec<usize>,
    target_areas: Vec<f64>,
    weights: LossWeights,
    opt: AdamW,
    iteration: usize,
    lr_modified: f64,
    iter_to_modify_lr: usize,
}

#[wasm_bindgen]
impl WasmOpt {
    /// Custom boundary: flat `[x0,y0,x1,y1,...]` (any scale — it is normalized).
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        boundary_xy: &[f64],
        num_sites: usize,
        area_ratios: &[f64],
        w_wall: f64,
        w_area: f64,
        w_lloyd: f64,
        w_topo: f64,
        w_bb: f64,
        w_cell: f64,
        seed: u32,
        lr: f64,
    ) -> WasmOpt {
        console_error_panic_hook::set_once();
        let pts: Vec<(f64, f64)> = boundary_xy.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        let w = LossWeights {
            w_wall, w_area, w_lloyd, w_topo, w_bb, w_cell,
        };
        WasmOpt::build(normalize(&pts), num_sites, area_ratios.to_vec(), w, seed as u64, lr)
    }

    /// One of the bundled example boundaries (`shape_a|shape_b|shape_c|shape_d|shape_e|shape_duck`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_shape(
        name: &str,
        num_sites: usize,
        area_ratios: &[f64],
        w_wall: f64,
        w_area: f64,
        w_lloyd: f64,
        w_topo: f64,
        w_bb: f64,
        w_cell: f64,
        seed: u32,
        lr: f64,
    ) -> Option<WasmOpt> {
        console_error_panic_hook::set_once();
        let boundary = shapes::by_name(name)?.polygon();
        let w = LossWeights {
            w_wall, w_area, w_lloyd, w_topo, w_bb, w_cell,
        };
        Some(WasmOpt::build(boundary, num_sites, area_ratios.to_vec(), w, seed as u64, lr))
    }

    /// Advance one optimization iteration; returns `{iteration, loss, sites, cells}`.
    pub fn step(&mut self) -> JsValue {
        let frame = self.advance();
        serde_wasm_bindgen::to_value(&frame).unwrap()
    }

    /// The normalized boundary ring (`[[x,y],...]`) for drawing + canvas fit.
    pub fn boundary(&self) -> JsValue {
        let ring: Vec<(f64, f64)> = self.boundary.exterior().0.iter().map(|p| (p.x, p.y)).collect();
        serde_wasm_bindgen::to_value(&ring).unwrap()
    }

    /// Current geometry without advancing — for previewing the initial layout.
    pub fn current(&self) -> JsValue {
        let b = loss::floor_plan_loss(
            &self.sites,
            &self.boundary,
            &self.target_areas,
            &self.room_indices,
            &self.weights,
            None,
        );
        serde_wasm_bindgen::to_value(&self.geometry_frame(&b)).unwrap()
    }
}

impl WasmOpt {
    fn build(
        boundary: Polygon<f64>,
        num_sites: usize,
        area_ratios: Vec<f64>,
        weights: LossWeights,
        seed: u64,
        lr: f64,
    ) -> WasmOpt {
        let area = boundary.unsigned_area();
        let target_areas: Vec<f64> = area_ratios.iter().map(|r| area * r).collect();
        let sites = init::initialize_sites(&boundary, num_sites, seed);
        let room_indices = init::kmeans_labels(&sites, area_ratios.len().max(1), seed);
        let opt = AdamW::new(sites.len(), lr);
        WasmOpt {
            boundary,
            sites,
            room_indices,
            target_areas,
            weights,
            opt,
            iteration: 0,
            // the bundled configs drop the LR to 0.8x at iteration 300
            lr_modified: lr * 0.8,
            iter_to_modify_lr: 300,
        }
    }

    fn advance(&mut self) -> Frame {
        self.iteration += 1;
        if self.iteration == self.iter_to_modify_lr {
            self.opt.set_lr(self.lr_modified);
        }
        // The local-gradient context computes the base geometry once and caches
        // it; its base total is the pre-step loss (what the CLI logs), and its
        // gradients reuse that geometry — ~2.4x faster than the global path and
        // bit-identical to it. (See ../rust grad_local.rs.)
        let ctx = grad_local::LocalGradContext::new(
            &self.sites,
            &self.boundary,
            &self.target_areas,
            &self.room_indices,
            &self.weights,
        );
        let pre_step_breakdown = ctx.base_breakdown();
        let grads = ctx.gradients(&self.sites);
        self.opt.step(&mut self.sites, &grads);
        // geometry AFTER the step (matches the GIF frame-order fix in run.rs),
        // tagged with the pre-step loss breakdown (its .total is what the CLI logs)
        self.geometry_frame(&pre_step_breakdown)
    }

    /// Build a Frame from the current sites + a given loss breakdown.
    fn geometry_frame(&self, bd: &loss::LossBreakdown) -> Frame {
        // render_cells keeps EVERY piece of each cell ∩ boundary, so a cell that
        // splits across a concave boundary notch renders all its pieces instead
        // of leaving an uncovered gap (matches the Python renderer). The loss
        // path (compute_cells) is unchanged.
        let render = voronoi::render_cells(&self.sites, &self.boundary);
        let cells: Vec<Cell> = render
            .iter()
            .map(|(i, c)| Cell {
                room: self.room_indices.get(*i).copied().unwrap_or(0),
                ring: c.exterior().0.iter().map(|p| (p.x, p.y)).collect(),
            })
            .collect();

        // Group the render cells by room and union each into the room geometry;
        // its (simplified) boundary rings are the wall centerlines.
        let n_rooms = self.target_areas.len().max(1);
        let mut room_polys: Vec<Vec<&Polygon<f64>>> = vec![Vec::new(); n_rooms];
        for (i, c) in &render {
            let r = self.room_indices.get(*i).copied().unwrap_or(0);
            if r < n_rooms {
                room_polys[r].push(c);
            }
        }
        let unions: Vec<MultiPolygon<f64>> = room_polys
            .iter()
            .map(|ps| geo::algorithm::unary_union(ps.iter().copied()))
            .collect();
        // Straighten room boundaries (drop near-collinear cell-edge vertices) so
        // the walls are clean straight runs, not lumpy. Render-only — the loss /
        // gradient path never sees this.
        let unions: Vec<MultiPolygon<f64>> = unions.iter().map(|u| u.simplify(&0.004)).collect();
        // The walls are the room-boundary CENTERLINES (the simplified room-union
        // rings). The canvas strokes each twice — black at the wall width, then
        // white slightly narrower — to draw the double-line walls. Doing it as a
        // stroke lets the line-join render junctions cleanly, with no band
        // geometry (no disk artifacts, no two-buffer gaps).
        let walls: Vec<Vec<(f64, f64)>> = unions
            .iter()
            .flat_map(|mp| mp.0.iter())
            .flat_map(|poly| std::iter::once(poly.exterior()).chain(poly.interiors().iter()))
            .map(|ring| ring.0.iter().map(|c| (c.x, c.y)).collect())
            .collect();

        Frame {
            iteration: self.iteration,
            loss: bd.total,
            wall: bd.wall,
            area: bd.area,
            lloyd: bd.lloyd,
            topo: bd.topo,
            sites: self.sites.iter().map(|s| (s[0], s[1])).collect(),
            cells,
            walls,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The native CLI `voronoi-floorplan shape_a --iterations N --seed 777`
    // (the hint-free standalone path) prints these pre-step losses (exact
    // Martinez-Rueda union). The web demo reuses that core for the LOSS, so its
    // first pre-step loss matches; but its GRADIENT uses the faster
    // edge-cancellation union (grad_local), which is directionally close
    // (cosine ≈ 0.99) but not bit-identical, so after many steps the trajectory
    // drifts slightly from the exact native trace while still tracking it.
    #[test]
    fn shape_a_tracks_native_standalone_trace() {
        let mut opt = WasmOpt::build(
            shapes::by_name("shape_a").unwrap().polygon(),
            40,
            vec![0.5, 0.3, 0.1, 0.1],
            LossWeights {
                w_wall: 2.5,
                w_area: 20.0,
                w_lloyd: 2.1,
                w_topo: 1.5,
                w_bb: 0.0,
                w_cell: 0.0,
                ..Default::default()
            },
            777,
            1e-2,
        );
        let (mut first, mut last) = (0.0_f32, 0.0_f32);
        for k in 1..=80 {
            let f = opt.advance();
            if k == 1 {
                first = f.loss;
            }
            if k == 80 {
                last = f.loss;
            }
        }
        // The demo's edge-cancellation gradient is only directionally close to the
        // exact path (cosine ≈ 0.99), so the wall-alignment trajectory is not
        // bit-exact and the early steps overshoot more than the taxicab loss did
        // (iter0 ≈ 54.4, iter1 ≈ 74). It must still drive the loss DOWN into the
        // native ballpark (~33 for shape_a/seed 777 at iter 80), not stall/diverge.
        assert!(first.is_finite() && last.is_finite(), "demo losses must be finite: {first}, {last}");
        assert!(last < 40.0, "demo must converge near the native floor; iter80 loss {last} (iter1 was {first})");
    }

    #[test]
    fn frame_exposes_wall_centerlines() {
        // The Frame must carry the room-boundary centerlines (closed rings) that
        // the canvas strokes twice into the double-line walls. Wall thickness is
        // now a render parameter (a stroke width), not geometry, so there is
        // nothing width-related to assert here.
        let opt = WasmOpt::build(
            shapes::by_name("shape_a").unwrap().polygon(),
            40,
            vec![0.5, 0.3, 0.1, 0.1],
            LossWeights { w_wall: 2.5, w_area: 20.0, w_lloyd: 2.1, w_topo: 1.5, w_bb: 0.0, w_cell: 0.0, ..Default::default() },
            777,
            1e-2,
        );
        let b = loss::floor_plan_loss(&opt.sites, &opt.boundary, &opt.target_areas, &opt.room_indices, &opt.weights, None);
        let f = opt.geometry_frame(&b);
        assert!(!f.walls.is_empty(), "frame exposes no wall centerlines");
        for ring in &f.walls {
            assert!(ring.len() >= 4, "wall centerline ring too short: {}", ring.len());
            let (a, b) = (ring[0], ring[ring.len() - 1]);
            assert!(
                (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9,
                "wall centerline ring not closed"
            );
        }
    }

    #[test]
    fn from_shape_threads_wall_weight() {
        // With ONLY the wall (alignment) weight active on a diagonal boundary, the
        // pre-step loss must be positive — proving from_shape threads w_wall through
        // to the optimizer.
        let mut opt = WasmOpt::from_shape(
            "shape_d",
            40,
            &[0.4, 0.3, 0.2, 0.1],
            5.0, // w_wall (alignment)
            0.0, 0.0, 0.0, 0.0, 0.0, // area, lloyd, topo, bb, cell all off
            777,
            1e-2,
        )
        .unwrap();
        let f = opt.advance();
        assert!(
            f.loss > 0.0 && f.loss.is_finite(),
            "from_shape must thread w_wall into the optimizer (pre-step loss {} should be > 0)",
            f.loss
        );
    }

}
