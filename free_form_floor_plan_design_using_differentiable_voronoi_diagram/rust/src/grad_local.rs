//! Faster, *local* finite-difference gradients for the in-browser demo.
//!
//! The global path (`grad::finite_difference_grads`) recomputes the whole
//! Voronoi diagram + full loss for each of the 2N perturbations. But a 1e-6
//! perturbation of one site only changes that site's Voronoi cell and its
//! immediate neighbors, so most of that work is redundant. This module caches
//! the base-step geometry once and, per perturbation, recomputes only the
//! affected cells and the loss delta.
//!
//! It is NOT bit-identical to the global path (the f32 loss reductions are
//! order-sensitive), so it stays a demo-only path — the native CLI keeps the
//! global gradients for its bit-exact equivalence with the Python reference.

use crate::loss::{
    compute_area_loss, compute_bb_loss, compute_cell_area_loss, compute_lloyd_loss,
    compute_topology_loss, compute_wall_loss, rooms_group, union_group, LossWeights,
};
use crate::voronoi::{compute_cells, site_neighbors, snap_cells};
use geo::{MultiPolygon, Polygon};

/// Sum the six loss components from already-built geometry, mirroring
/// `floor_plan_loss`'s guards and summation order so the result is identical.
fn total_from(
    cells: &[Polygon<f64>],
    groups: &[Vec<&Polygon<f64>>],
    unions: &[MultiPolygon<f64>],
    sites: &[[f32; 2]],
    target_areas: &[f64],
    room_indices: &[usize],
    w: &LossWeights,
) -> f32 {
    let wall = if w.w_wall > 0.0 { compute_wall_loss(unions, w.w_wall) } else { 0.0 };
    let area = if w.w_area > 0.0 { compute_area_loss(cells, target_areas, room_indices, w.w_area) } else { 0.0 };
    let lloyd = if w.w_lloyd > 0.0 { compute_lloyd_loss(cells, sites, w.w_lloyd) } else { 0.0 };
    let topo = if w.w_topo > 0.0 { compute_topology_loss(groups, unions, w.w_topo) } else { 0.0 };
    let bb = if w.w_bb > 0.0 { compute_bb_loss(unions, w.w_bb) } else { 0.0 };
    let cell = if w.w_cell > 0.0 { compute_cell_area_loss(cells, w.w_cell) } else { 0.0 };
    wall + area + lloyd + topo + bb + cell
}

/// Local finite-difference gradients: same central-difference formula as
/// `grad::finite_difference_grads`, but each forward evaluation reuses the
/// cached base geometry and only recomputes the cells the perturbed site
/// touches. Builds the base context once, then evaluates 2N local perturbations.
pub fn finite_difference_grads_local(
    sites: &[[f32; 2]],
    boundary: &Polygon<f64>,
    target_areas: &[f64],
    room_indices: &[usize],
    w: &LossWeights,
) -> Vec<[f32; 2]> {
    LocalGradContext::new(sites, boundary, target_areas, room_indices, w).gradients(sites)
}

/// Caches the base-step geometry/loss so perturbations can be evaluated
/// incrementally instead of from scratch.
pub struct LocalGradContext<'a> {
    boundary: &'a Polygon<f64>,
    target_areas: &'a [f64],
    room_indices: &'a [usize],
    w: &'a LossWeights,
    /// Voronoi neighbor site indices per site (stable under a 1e-6 perturbation).
    neighbors: Vec<Vec<usize>>,
    /// Base clipped cells (site order) and per-room unions, reused for the
    /// cells/rooms a single-site perturbation does not touch.
    base_cells: Vec<Polygon<f64>>,
    base_unions: Vec<MultiPolygon<f64>>,
    n_rooms: usize,
    base_total: f32,
}

impl<'a> LocalGradContext<'a> {
    pub fn new(
        sites: &[[f32; 2]],
        boundary: &'a Polygon<f64>,
        target_areas: &'a [f64],
        room_indices: &'a [usize],
        w: &'a LossWeights,
    ) -> Self {
        let sites_f64: Vec<[f64; 2]> = sites.iter().map(|&[x, y]| [x as f64, y as f64]).collect();
        let base_cells = compute_cells(sites, boundary, None).cells_sorted;
        let (n_rooms, base_unions, base_total) = {
            let groups = rooms_group(&base_cells, room_indices);
            let n_rooms = groups.len();
            let unions: Vec<MultiPolygon<f64>> = if w.w_wall > 0.0 || w.w_topo > 0.0 || w.w_bb > 0.0 {
                groups.iter().map(|g| union_group(g)).collect()
            } else {
                Vec::new()
            };
            let total = total_from(&base_cells, &groups, &unions, sites, target_areas, room_indices, w);
            (n_rooms, unions, total)
        };
        let neighbors = site_neighbors(&sites_f64, boundary);
        LocalGradContext { boundary, target_areas, room_indices, w, neighbors, base_cells, base_unions, n_rooms, base_total }
    }

    /// Total loss at the base (unperturbed) sites — must equal the global
    /// `floor_plan_loss` total.
    pub fn base_total(&self) -> f32 {
        self.base_total
    }

    /// Central finite-difference gradient (same formula as
    /// `grad::finite_difference_grads`) using the cached base geometry. `sites`
    /// must be the same configuration the context was built from.
    pub fn gradients(&self, sites: &[[f32; 2]]) -> Vec<[f32; 2]> {
        const EPS: f32 = 1e-6;
        let mut buf = sites.to_vec();
        (0..sites.len())
            .map(|i| {
                let mut g = [0.0f32; 2];
                for j in 0..2 {
                    let orig = sites[i][j];
                    buf[i][j] = orig + EPS;
                    let loss_pos = self.local_total(&buf, i);
                    buf[i][j] = orig - EPS;
                    let loss_neg = self.local_total(&buf, i);
                    buf[i][j] = orig;
                    g[j] = (loss_pos - loss_neg) / (2.0 * EPS);
                }
                g
            })
            .collect()
    }

    /// Recompute the clipped cells affected by perturbing one site: the site's
    /// own cell plus its Voronoi neighbors. Returns `(site_index, clipped_cell)`
    /// pairs. These must match what the global `compute_cells` produces for
    /// those sites at the same perturbed configuration.
    pub fn affected_cells(&self, perturbed_sites: &[[f32; 2]], i: usize) -> Vec<(usize, Polygon<f64>)> {
        let pert_f64: Vec<[f64; 2]> = perturbed_sites.iter().map(|&[x, y]| [x as f64, y as f64]).collect();
        let raw = crate::voronoi::raw_cells_per_site(&pert_f64, self.boundary);
        let mut affected = Vec::with_capacity(self.neighbors[i].len() + 1);
        affected.push(i);
        affected.extend(self.neighbors[i].iter().copied());
        affected
            .into_iter()
            .map(|j| {
                let (cell, _) = crate::voronoi::clip_cell(&raw[j], self.boundary, pert_f64[j]);
                (j, cell)
            })
            .collect()
    }

    /// Total loss at a configuration where exactly one site (`i`) has moved,
    /// computed locally: patch the base cells with the recomputed affected
    /// cells, then recompute the loss reusing cached unions for the rooms with
    /// no affected cell. Must match the global `floor_plan_loss` at the same
    /// configuration.
    pub fn local_total(&self, perturbed_sites: &[[f32; 2]], i: usize) -> f32 {
        let affected = self.affected_cells(perturbed_sites, i);
        let mut cells = self.base_cells.clone();
        let mut affected_rooms = vec![false; self.n_rooms];
        for (j, cell) in affected {
            affected_rooms[self.room_indices[j]] = true;
            cells[j] = cell;
        }
        // re-weld twin vertices (idempotent on the untouched base cells)
        snap_cells(&mut cells);

        let groups = rooms_group(&cells, self.room_indices);
        let need_unions = self.w.w_wall > 0.0 || self.w.w_topo > 0.0 || self.w.w_bb > 0.0;
        let unions: Vec<MultiPolygon<f64>> = if need_unions {
            (0..self.n_rooms)
                .map(|r| {
                    if affected_rooms[r] {
                        union_group(&groups[r])
                    } else {
                        self.base_unions[r].clone()
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        total_from(&cells, &groups, &unions, perturbed_sites, self.target_areas, self.room_indices, self.w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::floor_plan_loss;
    use crate::{init, shapes};

    fn shape_a_setup() -> (Vec<[f32; 2]>, Polygon<f64>, Vec<f64>, Vec<usize>, LossWeights) {
        use geo::Area;
        let boundary = shapes::by_name("shape_a").unwrap().polygon();
        let ratios = [0.5f64, 0.3, 0.1, 0.1];
        let area = boundary.unsigned_area();
        let target_areas: Vec<f64> = ratios.iter().map(|r| area * r).collect();
        let sites = init::initialize_sites(&boundary, 40, 777);
        let room_indices = init::kmeans_labels(&sites, ratios.len(), 777);
        let w = LossWeights { w_wall: 2.5, w_area: 20.0, w_lloyd: 2.1, w_topo: 1.5, w_bb: 0.0, w_cell: 0.0 };
        (sites, boundary, target_areas, room_indices, w)
    }

    #[test]
    fn local_grad_matches_global_cosine() {
        let (sites, boundary, target_areas, room_indices, w) = shape_a_setup();
        let global = crate::grad::finite_difference_grads(&sites, &boundary, &target_areas, &room_indices, &w, None);
        let local = finite_difference_grads_local(&sites, &boundary, &target_areas, &room_indices, &w);
        let dot: f32 = global.iter().zip(&local).map(|(g, l)| g[0] * l[0] + g[1] * l[1]).sum();
        let ng: f32 = global.iter().map(|g| g[0] * g[0] + g[1] * g[1]).sum::<f32>().sqrt();
        let nl: f32 = local.iter().map(|l| l[0] * l[0] + l[1] * l[1]).sum::<f32>().sqrt();
        let cos = dot / (ng * nl);
        assert!(cos > 0.999, "gradient cosine {} (global norm {}, local norm {})", cos, ng, nl);
    }

    #[test]
    fn local_total_matches_global_perturbed() {
        let (sites, boundary, target_areas, room_indices, w) = shape_a_setup();
        let ctx = LocalGradContext::new(&sites, &boundary, &target_areas, &room_indices, &w);
        // move one site enough that the loss visibly changes
        let i = 7;
        let mut pert = sites.clone();
        pert[i][1] += 1e-3;
        let global = floor_plan_loss(&pert, &boundary, &target_areas, &room_indices, &w, None).total;
        let local = ctx.local_total(&pert, i);
        assert!(
            (local - global).abs() < 1e-3,
            "local total {} vs global {} (diff {})",
            local,
            global,
            (local - global).abs()
        );
    }

    #[test]
    fn affected_cells_match_global() {
        use geo::Area;
        let (sites, boundary, target_areas, room_indices, w) = shape_a_setup();
        let ctx = LocalGradContext::new(&sites, &boundary, &target_areas, &room_indices, &w);

        // perturb site 0 by +eps and recompute only the affected cells
        let i = 0;
        let mut pert = sites.clone();
        pert[i][0] += 1e-6;
        let affected = ctx.affected_cells(&pert, i);

        // the affected set must be the site plus at least one neighbor
        assert!(affected.len() >= 2, "expected site + neighbors, got {}", affected.len());

        // each affected cell's area must match the global recompute for that site
        let global = crate::voronoi::compute_cells(&pert, &boundary, None);
        for (j, cell) in &affected {
            let g = &global.cells_sorted[*j];
            assert!(
                (cell.unsigned_area() - g.unsigned_area()).abs() < 1e-7,
                "affected cell {} area {} != global {}",
                j,
                cell.unsigned_area(),
                g.unsigned_area()
            );
        }
    }

    #[test]
    fn base_total_matches_global_loss() {
        let (sites, boundary, target_areas, room_indices, w) = shape_a_setup();
        let global = floor_plan_loss(&sites, &boundary, &target_areas, &room_indices, &w, None).total;
        let ctx = LocalGradContext::new(&sites, &boundary, &target_areas, &room_indices, &w);
        assert!(
            (ctx.base_total() - global).abs() < 1e-4,
            "local base total {} != global loss {}",
            ctx.base_total(),
            global
        );
    }
}
