//! Faster, *local* finite-difference gradients for the in-browser demo.
//!
//! The global path (`grad::finite_difference_grads`) recomputes the whole
//! Voronoi diagram + full loss for each of the 2N perturbations. But a 1e-6
//! perturbation of one site only changes that site's Voronoi cell and its
//! immediate neighbors, so most of that work is redundant. This module caches
//! the base-step geometry once and, per perturbation, recomputes only the
//! affected cells and the rooms they touch.
//!
//! The dominant cost is the per-room boolean union that wall/topo/bb need, so
//! this path uses geo's i_overlay `unary_union` (`room_union`) instead of the
//! exact Martinez-Rueda union. Its robust internal noding welds shared/twin
//! vertices, so the affected cells need no pre-snapping, and it is faster end to
//! end. The result is directionally close to the exact global gradient (cosine >
//! 0.99) but not bit-identical, so it stays a demo-only path — the native CLI
//! keeps the global gradients (exact union) for its bit-exact equivalence with
//! Python.

use crate::loss::{
    compute_area_loss, compute_bb_loss, compute_cell_area_loss, compute_lloyd_loss,
    compute_topology_loss, compute_wall_loss, rooms_group, wall_local_term, LossWeights,
};
use crate::voronoi::{compute_cells, site_neighbors};
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
    boundary: &Polygon<f64>,
) -> f32 {
    let wall = if w.w_wall > 0.0 { compute_wall_loss(unions, w.w_wall) } else { 0.0 };
    let area = if w.w_area > 0.0 { compute_area_loss(cells, target_areas, room_indices, w.w_area) } else { 0.0 };
    let lloyd = if w.w_lloyd > 0.0 { compute_lloyd_loss(cells, sites, w.w_lloyd) } else { 0.0 };
    let topo = if w.w_topo > 0.0 { compute_topology_loss(groups, unions, w.w_topo) } else { 0.0 };
    let bb = if w.w_bb > 0.0 { compute_bb_loss(unions, w.w_bb) } else { 0.0 };
    let cell = if w.w_cell > 0.0 { compute_cell_area_loss(cells, w.w_cell) } else { 0.0 };
    let wall_local = wall_local_term(unions, boundary, w);
    wall + area + lloyd + topo + bb + cell + wall_local
}

/// Per-room boolean union for the demo path, via geo's i_overlay `unary_union`.
/// Its robust internal noding welds shared/twin vertices, so the input cells
/// need NOT be pre-snapped (unlike the old edge-cancellation union, which
/// required bitwise-shared edges). Demo-only; the CLI path keeps the exact
/// geo-booleanop union for Python bit-parity.
pub(crate) fn room_union(cells: &[&Polygon<f64>]) -> MultiPolygon<f64> {
    geo::algorithm::unary_union(cells.iter().copied())
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
        let need_unions = w.w_wall > 0.0 || w.w_topo > 0.0 || w.w_bb > 0.0 || w.w_wall_local > 0.0;
        let (n_rooms, base_unions, base_total) = {
            let groups = rooms_group(&base_cells, room_indices);
            let n_rooms = groups.len();
            // Edge-cancellation union (see module docs): the base per-room unions
            // are reused for the rooms a single-site perturbation does not touch.
            let unions: Vec<MultiPolygon<f64>> = if need_unions {
                groups.iter().map(|g| room_union(g)).collect()
            } else {
                Vec::new()
            };
            let total = total_from(&base_cells, &groups, &unions, sites, target_areas, room_indices, w, boundary);
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

    /// Parallel sibling of `gradients`: fans the 2N independent perturbation
    /// evaluations across rayon threads on native (each task gets its own
    /// scratch copy of the sites); the wasm build has no threads, so it runs
    /// them serially. Bitwise identical to `gradients` — each (site, axis)
    /// evaluation depends only on that single perturbed coordinate.
    pub fn gradients_par(&self, sites: &[[f32; 2]]) -> Vec<[f32; 2]> {
        const EPS: f32 = 1e-6;
        // Each (site, axis) evaluation is independent; give every task its own
        // scratch copy so concurrent perturbations cannot race. The per-index
        // arithmetic is identical to `gradients` (serial reuses one scratch and
        // restores it after each axis, so its scratch always equals `sites` at
        // the start of a site too), hence the result is bitwise identical.
        let one = |i: usize| -> [f32; 2] {
            let mut buf = sites.to_vec();
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
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            (0..sites.len()).into_par_iter().map(one).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..sites.len()).map(one).collect()
        }
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
                // Trivial accept: a cell fully inside the boundary is unchanged
                // by clipping (cell ∩ boundary == cell), so skip the boolean
                // intersection — ~half of all affected-cell clips qualify. The
                // raw cell has the same vertex set as the clip but voronoice's
                // winding, so orient it to the clipped winding (geo Default: CCW
                // exterior); room_union's i_overlay fill rule is winding-sensitive,
                // so a mismatched cell would not dissolve shared edges. The predicate is
                // sound (never a false accept), so this only drops identity work.
                // (Demo path only; the exact CLI path keeps clip_cell.)
                let cell = if crate::voronoi::cell_inside_boundary(&raw[j], self.boundary) {
                    use geo::orient::{Direction, Orient};
                    raw[j].orient(Direction::Default)
                } else {
                    crate::voronoi::clip_cell(&raw[j], self.boundary, pert_f64[j]).0
                };
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
        // No snap: room_union (i_overlay) nodes shared/twin vertices internally,
        // so the patched cells need not be pre-welded (the old edge-cancellation
        // union required bitwise-shared edges and a snap pass; i_overlay does not).
        let groups = rooms_group(&cells, self.room_indices);

        let need_unions = self.w.w_wall > 0.0 || self.w.w_topo > 0.0 || self.w.w_bb > 0.0 || self.w.w_wall_local > 0.0;
        let unions: Vec<MultiPolygon<f64>> = if need_unions {
            (0..self.n_rooms)
                .map(|r| {
                    if affected_rooms[r] {
                        room_union(&groups[r])
                    } else {
                        self.base_unions[r].clone()
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        total_from(&cells, &groups, &unions, perturbed_sites, self.target_areas, self.room_indices, self.w, self.boundary)
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
        let w = LossWeights { w_wall: 2.5, w_area: 20.0, w_lloyd: 2.1, w_topo: 1.5, w_bb: 0.0, w_cell: 0.0, w_wall_local: 0.0, ..Default::default() };
        (sites, boundary, target_areas, room_indices, w)
    }

    #[test]
    fn local_grad_includes_wall_local_and_matches_global() {
        let (sites, boundary, ta, ri, _) = shape_a_setup();
        // only the local-frame wall term active
        let w = LossWeights {
            w_wall: 0.0, w_area: 0.0, w_lloyd: 0.0, w_topo: 0.0, w_bb: 0.0, w_cell: 0.0, w_wall_local: 5.0,
            ..Default::default()
        };
        let global = floor_plan_loss(&sites, &boundary, &ta, &ri, &w, None).total;
        assert!(global > 0.0, "precondition: the global wall_local loss must be positive");
        let ctx = LocalGradContext::new(&sites, &boundary, &ta, &ri, &w);
        // the FD path must see the same loss as the global path...
        assert!(
            (ctx.base_total() - global).abs() < 1e-4,
            "grad_local base_total {} must include wall_local and match global {}",
            ctx.base_total(),
            global
        );
        // ...and the term must actually drive a finite, non-zero gradient
        let g = ctx.gradients(&sites);
        let nonzero = g.iter().any(|c| c[0].abs() > 0.0 || c[1].abs() > 0.0);
        let finite = g.iter().all(|c| c[0].is_finite() && c[1].is_finite());
        assert!(nonzero && finite, "wall_local must drive a finite, non-zero local gradient");
    }

    #[test]
    fn local_grad_directionally_close_but_not_exact() {
        // The demo path uses the edge-cancellation union (~3x faster than the
        // pairwise Martinez-Rueda union). Its wall term is ~1 ulp off the exact
        // union, which the central finite difference (÷2e-6) amplifies, so the
        // local gradient is directionally close to the exact global gradient
        // (cosine > 0.99) but no longer bit-identical (maxAbsDiff > 0). The
        // native CLI keeps the exact union, so its Python parity is unaffected.
        let (sites, boundary, target_areas, room_indices, w) = shape_a_setup();
        let global = crate::grad::finite_difference_grads(&sites, &boundary, &target_areas, &room_indices, &w, None);
        let local = finite_difference_grads_local(&sites, &boundary, &target_areas, &room_indices, &w);
        let dot: f32 = global.iter().zip(&local).map(|(g, l)| g[0] * l[0] + g[1] * l[1]).sum();
        let ng: f32 = global.iter().map(|g| g[0] * g[0] + g[1] * g[1]).sum::<f32>().sqrt();
        let nl: f32 = local.iter().map(|l| l[0] * l[0] + l[1] * l[1]).sum::<f32>().sqrt();
        let cos = dot / (ng * nl);
        let max_abs: f32 = global
            .iter()
            .zip(&local)
            .flat_map(|(g, l)| [(g[0] - l[0]).abs(), (g[1] - l[1]).abs()])
            .fold(0.0, f32::max);
        assert!(cos > 0.99, "gradient cosine {cos} below 0.99 (global norm {ng}, local norm {nl})");
        assert!(
            max_abs > 0.0,
            "expected the faster edge-cancellation union path (maxAbsDiff > 0), got bit-identical {max_abs:e}"
        );
    }

    #[test]
    fn gradients_par_matches_serial_bitwise() {
        let (sites, boundary, target_areas, room_indices, w) = shape_a_setup();
        let ctx = LocalGradContext::new(&sites, &boundary, &target_areas, &room_indices, &w);
        let serial = ctx.gradients(&sites);
        let par = ctx.gradients_par(&sites);
        assert_eq!(serial.len(), par.len(), "length mismatch");
        for (k, (s, p)) in serial.iter().zip(&par).enumerate() {
            assert_eq!(
                (s[0].to_bits(), s[1].to_bits()),
                (p[0].to_bits(), p[1].to_bits()),
                "grad[{k}] serial {s:?} != par {p:?}"
            );
        }
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
    fn interior_skip_preserves_gradient_direction() {
        // KR2: skipping the boolean clip for interior cells must NOT change the
        // demo gradient direction — it stays cosine > 0.99 vs the exact global
        // gradient. A naive skip that returns the raw cell without matching the
        // clipped winding breaks `room_union`'s winding-sensitive fill rule
        // (shared edges no longer dissolve), collapsing the cosine: RED until the
        // skip orients the cell to the clipped winding.
        let (sites, boundary, target_areas, room_indices, w) = shape_a_setup();

        // the fixture must actually contain interior cells, or the skip never
        // exercises and the cosine check is vacuous.
        let ctx = LocalGradContext::new(&sites, &boundary, &target_areas, &room_indices, &w);
        let mut skips = 0usize;
        for i in 0..sites.len() {
            let mut pert = sites.clone();
            pert[i][0] += 1e-6;
            let pert_f64: Vec<[f64; 2]> = pert.iter().map(|&[x, y]| [x as f64, y as f64]).collect();
            let raw = crate::voronoi::raw_cells_per_site(&pert_f64, &boundary);
            for (j, _) in ctx.affected_cells(&pert, i) {
                if crate::voronoi::cell_inside_boundary(&raw[j], &boundary) {
                    skips += 1;
                }
            }
        }
        assert!(skips > 0, "fixture has no interior cells — the skip never exercises");

        let global =
            crate::grad::finite_difference_grads(&sites, &boundary, &target_areas, &room_indices, &w, None);
        let local = finite_difference_grads_local(&sites, &boundary, &target_areas, &room_indices, &w);
        let dot: f32 = global.iter().zip(&local).map(|(g, l)| g[0] * l[0] + g[1] * l[1]).sum();
        let ng: f32 = global.iter().map(|g| g[0] * g[0] + g[1] * g[1]).sum::<f32>().sqrt();
        let nl: f32 = local.iter().map(|l| l[0] * l[0] + l[1] * l[1]).sum::<f32>().sqrt();
        let cos = dot / (ng * nl);
        assert!(
            cos > 0.99,
            "interior-skip broke gradient direction: cosine {cos} (global norm {ng}, local norm {nl})"
        );
    }

    #[test]
    fn room_union_correct_on_unsnapped_cells() {
        // KR1: room_union (i_overlay) must produce a correctly dissolved room
        // union from UNSNAPPED clipped cells — its perimeter matches the exact
        // reference union (`loss::union_group`, Martinez on snapped cells).
        // i_overlay nodes internally, so the manual snap is unnecessary. A union
        // that fails to dissolve internal edges has an inflated perimeter, which
        // this catches.
        use geo::LineString;
        fn ring_len(ls: &LineString<f64>) -> f64 {
            ls.0.windows(2)
                .map(|w| ((w[1].x - w[0].x).powi(2) + (w[1].y - w[0].y).powi(2)).sqrt())
                .sum()
        }
        fn perimeter(mp: &MultiPolygon<f64>) -> f64 {
            mp.iter()
                .map(|p| ring_len(p.exterior()) + p.interiors().iter().map(ring_len).sum::<f64>())
                .sum()
        }

        let (sites, boundary, _ta, ri, _w) = shape_a_setup();
        let sites_f64: Vec<[f64; 2]> = sites.iter().map(|&[x, y]| [x as f64, y as f64]).collect();
        let raw = crate::voronoi::raw_cells_per_site(&sites_f64, &boundary);
        // clipped but UNSNAPPED cells (site order)
        let unsnapped: Vec<Polygon<f64>> = (0..sites.len())
            .map(|j| crate::voronoi::clip_cell(&raw[j], &boundary, sites_f64[j]).0)
            .collect();
        // snapped reference copy
        let mut snapped = unsnapped.clone();
        crate::voronoi::snap_cells(&mut snapped);

        // pick the room with the most cells (guarantees internal shared edges)
        let n_rooms = ri.iter().copied().max().unwrap() + 1;
        let r = (0..n_rooms).max_by_key(|&r| ri.iter().filter(|&&x| x == r).count()).unwrap();
        let snapped_room: Vec<&Polygon<f64>> =
            (0..sites.len()).filter(|&j| ri[j] == r).map(|j| &snapped[j]).collect();
        let unsnapped_room: Vec<&Polygon<f64>> =
            (0..sites.len()).filter(|&j| ri[j] == r).map(|j| &unsnapped[j]).collect();

        let reference = perimeter(&crate::loss::union_group(&snapped_room));
        let got = perimeter(&room_union(&unsnapped_room));
        assert!(reference > 0.0, "degenerate reference");
        assert!(
            (got - reference).abs() / reference < 1e-3,
            "room_union perimeter {got} vs reference {reference} (rel {})",
            (got - reference).abs() / reference
        );
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
