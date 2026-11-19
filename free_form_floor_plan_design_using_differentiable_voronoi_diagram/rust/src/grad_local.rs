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
    area_loss_from_areas, cell_area_for_loss, compute_bb_loss, compute_cell_area_loss,
    compute_topology_loss, compute_wall_terms, rooms_group, LossBreakdown, LossWeights, WallLossMode,
    compute_wall_local_loss_reusing, lloyd_cell_distance, lloyd_loss_from_distances, local_wall_ring_scores,
};
use crate::voronoi::compute_cells;
use crate::wall_shape::WallShapeCache;
use geo::{MultiPolygon, Polygon};
use crate::program::RoomProgram;

#[cfg(test)]
thread_local! {
    pub(crate) static CELL_WORKSPACES_BUILT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The six per-term loss components from already-built geometry, mirroring
/// `floor_plan_loss`'s guards and summation order so the result is identical.
fn breakdown_from(
    cells: &[Polygon<f64>],
    groups: &[Vec<&Polygon<f64>>],
    unions: &[MultiPolygon<f64>],
    area: f32,
    lloyd: f32,
    w: &LossWeights,
    boundary: &Polygon<f64>,
    program: Option<&RoomProgram>,
    shared: Option<&crate::walls::SharedWalls>,
    wall_shape: Option<&WallShapeCache<'_>>,
    legacy_wall: impl FnOnce() -> f32,
) -> LossBreakdown {
    let (wall,jog,connection) = compute_wall_terms(groups,boundary,w,program,shared,wall_shape,legacy_wall);
    let topo = if w.w_topo > 0.0 { compute_topology_loss(groups, unions, w.w_topo) } else { 0.0 };
    let bb = if w.w_bb > 0.0 { compute_bb_loss(unions, w.w_bb) } else { 0.0 };
    let cell = if w.w_cell > 0.0 { compute_cell_area_loss(cells, w.w_cell) } else { 0.0 };
    let total = wall + area + lloyd + topo + bb + cell + connection + jog;
    LossBreakdown { jog, total, wall, area, lloyd, topo, bb, cell, connection }
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
    program: Option<&'a RoomProgram>,
    /// Raw geometry detects newly created neighbors at degenerate diagrams too.
    base_raw: Vec<Polygon<f64>>,
    base_raw_vertices: crate::voronoi::VertexWelder,
    base_vertices: crate::voronoi::VertexWelder,
    shared_cells: Option<crate::walls::SharedWallCells>,
    wall_shape: Option<WallShapeCache<'a>>,
    wall_rings: Option<Vec<Vec<f32>>>,
    /// Base clipped cells (site order) and per-room unions, reused for the
    /// cells/rooms a single-site perturbation does not touch.
    base_cells: Vec<Polygon<f64>>,
    base_unions: Vec<MultiPolygon<f64>>,
    base_cell_areas: Option<Vec<f64>>,
    base_lloyd_distances: Option<Vec<Option<f32>>>,
    n_rooms: usize,
    /// Per-term loss at the base (unperturbed) sites. `base_total()` reads its
    /// `.total`; the per-term fields feed the demo's per-term loss graphs.
    base_breakdown: LossBreakdown,
}

impl<'a> LocalGradContext<'a> {
    pub fn new(
        sites: &[[f32; 2]],
        boundary: &'a Polygon<f64>,
        target_areas: &'a [f64],
        room_indices: &'a [usize],
        w: &'a LossWeights,
    ) -> Self { Self::new_with_program(sites,boundary,target_areas,room_indices,w,None) }

    pub fn new_with_program(
        sites: &[[f32; 2]], boundary: &'a Polygon<f64>, target_areas: &'a [f64],
        room_indices: &'a [usize], w: &'a LossWeights, program: Option<&'a RoomProgram>,
    ) -> Self {
        let sites_f64: Vec<[f64; 2]> = sites.iter().map(|&[x, y]| [x as f64, y as f64]).collect();
        let base_cells = compute_cells(sites, boundary, None).cells_sorted;
        let base_cell_areas: Option<Vec<_>> = (w.w_area > 0.0)
            .then(|| base_cells.iter().map(cell_area_for_loss).collect());
        let base_lloyd_distances: Option<Vec<_>> = (w.w_lloyd > 0.0)
            .then(|| base_cells.iter().zip(sites).map(|(cell, site)| lloyd_cell_distance(cell, site)).collect());
        let area = base_cell_areas.as_ref().map_or(0.0, |areas| {
            area_loss_from_areas(areas.iter().copied(), target_areas, room_indices, w.w_area)
        });
        let lloyd = base_lloyd_distances.as_ref().map_or(0.0, |distances| {
            lloyd_loss_from_distances(distances.iter().copied().flatten(), w.w_lloyd)
        });
        let base_vertices=crate::voronoi::VertexWelder::for_cells(&base_cells);
        let score_legacy_wall = w.w_wall > 0. && matches!(w.wall_mode, WallLossMode::LegacyL1);
        let need_unions = score_legacy_wall || w.w_topo > 0.0 || w.w_bb > 0.0;
        let (n_rooms, base_unions, base_breakdown, shared_cells, wall_shape, wall_rings) = {
            let groups = rooms_group(&base_cells, room_indices);
            let n_rooms = groups.len();
            // Edge-cancellation union (see module docs): the base per-room unions
            // are reused for the rooms a single-site perturbation does not touch.
            let unions: Vec<MultiPolygon<f64>> = if need_unions {
                groups.iter().map(|g| room_union(g)).collect()
            } else {
                Vec::new()
            };
            let (score_direction, score_jog) = w.shared_wall_shape();
            let score_shape = score_direction || score_jog;
            let shared_cells = if score_shape || program.is_some_and(|p| p.has_connections()) {
                Some(crate::walls::SharedWallCells::new(&base_cells, room_indices))
            } else {
                None
            };
            let shared = shared_cells.as_ref().map(|cells| cells.snapshot());
            let wall_shape = score_shape.then(|| {
                WallShapeCache::new(shared.as_ref().unwrap(), boundary, score_direction, score_jog)
            });
            let wall_rings: Option<Vec<Vec<f32>>> = score_legacy_wall.then(|| {
                unions.iter().map(|union| local_wall_ring_scores(union, boundary).collect()).collect()
            });
            let legacy_wall = || compute_wall_local_loss_reusing(&unions, boundary, w.w_wall, |room| {
                wall_rings.as_ref().map(|rings| rings[room].as_slice())
            });
            let bd = breakdown_from(&base_cells, &groups, &unions, area, lloyd, w, boundary, program, shared.as_ref(), wall_shape.as_ref(), legacy_wall);
            (n_rooms, unions, bd, shared_cells, wall_shape, wall_rings)
        };
        let base_raw = crate::voronoi::raw_cells_per_site(&sites_f64, boundary);
        let base_raw_vertices=crate::voronoi::VertexWelder::for_cells(&base_raw);
        LocalGradContext { boundary, target_areas, room_indices, w, program, base_raw, base_raw_vertices, base_vertices, shared_cells, wall_shape, wall_rings, base_cells, base_unions, base_cell_areas, base_lloyd_distances, n_rooms, base_breakdown }
    }

    /// Total loss at the base (unperturbed) sites — must equal the global
    /// `floor_plan_loss` total.
    pub fn base_total(&self) -> f32 {
        self.base_breakdown.total
    }

    /// Per-term loss at the base (unperturbed) sites — the components that sum
    /// to `base_total()`. Used to stream the per-term loss graphs in the demo.
    pub fn base_breakdown(&self) -> LossBreakdown {
        self.base_breakdown
    }

    /// Central finite-difference gradient (same formula as
    /// `grad::finite_difference_grads`) using the cached base geometry. `sites`
    /// must be the same configuration the context was built from.
    pub fn gradients(&self, sites: &[[f32; 2]]) -> Vec<[f32; 2]> {
        const EPS: f32 = 1e-6;
        let mut buf = sites.to_vec();
        let mut cells = self.cell_workspace();
        (0..sites.len())
            .map(|i| {
                let mut g = [0.0f32; 2];
                for j in 0..2 {
                    let orig = sites[i][j];
                    buf[i][j] = orig + EPS;
                    let loss_pos = self.local_total_in(&buf, i, &mut cells);
                    buf[i][j] = orig - EPS;
                    let loss_neg = self.local_total_in(&buf, i, &mut cells);
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
        #[cfg(not(feature = "parallel"))]
        {
            self.gradients(sites)
        }
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            const EPS: f32 = 1e-6;
            // Each parallel task owns its scratch cells and restores them
            // between evaluations, just as the serial path does.
            (0..sites.len())
                .into_par_iter()
                .map_init(|| self.cell_workspace(), |cells, i| {
                    let mut buf = sites.to_vec();
                    let mut g = [0.0f32; 2];
                    for j in 0..2 {
                        let orig = sites[i][j];
                        buf[i][j] = orig + EPS;
                        let loss_pos = self.local_total_in(&buf, i, cells);
                        buf[i][j] = orig - EPS;
                        let loss_neg = self.local_total_in(&buf, i, cells);
                        buf[i][j] = orig;
                        g[j] = (loss_pos - loss_neg) / (2.0 * EPS);
                    }
                    g
                })
                .collect()
        }
    }

    /// Recompute the clipped cells affected by perturbing one site: the site's
    /// own cell plus its Voronoi neighbors. Returns `(site_index, clipped_cell)`
    /// pairs. These must match what the global `compute_cells` produces for
    /// those sites at the same perturbed configuration.
    pub fn affected_cells(&self, perturbed_sites: &[[f32; 2]], i: usize) -> Vec<(usize, Polygon<f64>)> {
        let pert_f64: Vec<[f64; 2]> = perturbed_sites.iter().map(|&[x, y]| [x as f64, y as f64]).collect();
        // A perturbation can flip a Delaunay diagonal. Compare the raw cells
        // before allocating polygons so new neighbors cannot retain stale geometry.
        crate::voronoi::affected_raw_cells(&pert_f64, self.boundary, &self.base_raw, &self.base_raw_vertices, i)
            .into_iter()
            .map(|(j, raw)| {
                // Trivial accept: a cell fully inside the boundary is unchanged
                // by clipping (cell ∩ boundary == cell), so skip the boolean
                // intersection — ~half of all affected-cell clips qualify. The
                // raw cell has the same vertex set as the clip but voronoice's
                // winding, so orient it to the clipped winding (geo Default: CCW
                // exterior); room_union's i_overlay fill rule is winding-sensitive,
                // so a mismatched cell would not dissolve shared edges. The predicate is
                // sound (never a false accept), so this only drops identity work.
                // (Demo path only; the exact CLI path keeps clip_cell.)
                let cell = if crate::voronoi::cell_inside_boundary(&raw, self.boundary) {
                    use geo::orient::{Direction, Orient};
                    raw.orient(Direction::Default)
                } else {
                    crate::voronoi::clip_cell(&raw, self.boundary, pert_f64[j]).0
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
        self.local_total_in(perturbed_sites, i, &mut self.cell_workspace())
    }

    fn cell_workspace(&self) -> Vec<Polygon<f64>> {
        #[cfg(test)]
        CELL_WORKSPACES_BUILT.set(CELL_WORKSPACES_BUILT.get() + 1);
        self.base_cells.clone()
    }

    fn local_total_in(
        &self,
        perturbed_sites: &[[f32; 2]],
        i: usize,
        cells: &mut [Polygon<f64>],
    ) -> f32 {
        let mut affected = self.affected_cells(perturbed_sites, i);
        let mut affected_rooms = vec![false; self.n_rooms];
        let mut vertices=crate::voronoi::VertexWelder::default();
        let mut changed=Vec::with_capacity(affected.len());
        for &mut (j, ref mut cell) in &mut affected {
            crate::voronoi::snap_cells_using(std::slice::from_mut(cell),&mut vertices,Some(&self.base_vertices));
            affected_rooms[self.room_indices[j]] = true;
            std::mem::swap(&mut cells[j], cell);
            changed.push(j);
        }
        // Only changed cells were welded. Cached room geometry remains valid.
        let groups = rooms_group(cells, self.room_indices);

        let need_unions = self.wall_rings.is_some() || self.w.w_topo > 0.0 || self.w.w_bb > 0.0;
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
        let shared=self.shared_cells.as_ref().map(|c|c.updated(cells,&changed));
        let legacy_wall = || compute_wall_local_loss_reusing(&unions, self.boundary, self.w.w_wall, |room| {
            self.wall_rings.as_ref().filter(|_| !affected_rooms[room]).map(|rings| rings[room].as_slice())
        });
        let area = self.base_cell_areas.as_ref().map_or(0.0, |saved| {
            let mut changed_here = changed_sites_in_order(&changed);
            let areas = cells.iter().zip(saved).enumerate().map(|(i, (cell, &previous))| {
                if changed_here(i) { cell_area_for_loss(cell) } else { previous }
            });
            area_loss_from_areas(areas, self.target_areas, self.room_indices, self.w.w_area)
        });
        let lloyd = self.base_lloyd_distances.as_ref().map_or(0.0, |saved| {
            let mut changed_here = changed_sites_in_order(&changed);
            let distances = cells.iter().zip(perturbed_sites).zip(saved).enumerate()
                .filter_map(|(i, ((cell, site), &previous))| {
                    if changed_here(i) { lloyd_cell_distance(cell, site) } else { previous }
                });
            lloyd_loss_from_distances(distances, self.w.w_lloyd)
        });
        let total = breakdown_from(cells, &groups, &unions, area, lloyd, self.w, self.boundary, self.program, shared.as_ref(), self.wall_shape.as_ref(), legacy_wall).total;
        for (site, original) in affected {
            cells[site] = original;
        }
        total
    }
}

// Affected cells arrive in site order, matching the two loss folds.
fn changed_sites_in_order(changed: &[usize]) -> impl FnMut(usize) -> bool + '_ {
    let mut next = changed.iter().copied().peekable();
    move |site| next.next_if(|index| *index == site).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::floor_plan_loss;
    use crate::{init, shapes};

    #[test]
    fn materializes_raw_polygons_only_for_affected_cells() {
        use crate::voronoi::RAW_POLYGONS_BUILT;

        let boundary = shapes::by_name("shape_e").unwrap().polygon();
        let sites = init::initialize_sites(&boundary, 80, 4971);
        let labels = init::kmeans_labels(&sites, 6, 4971);
        let targets = [0.2; 6];
        let weights = LossWeights::default();
        let context = LocalGradContext::new(&sites, &boundary, &targets, &labels, &weights);
        for offset in [0., 1e-6, -1e-6] {
            let mut moved = sites.clone();
            moved[11][0] += offset;
            RAW_POLYGONS_BUILT.set(0);
            let affected = context.affected_cells(&moved, 11);
            assert!(affected.iter().any(|(site, _)| *site == 11));
            assert!(affected.len() < sites.len(), "the fixture must leave raw cells unchanged");
            assert_eq!(RAW_POLYGONS_BUILT.get(), affected.len(),
                "unchanged raw rings do not need owned polygons for one perturbation");
        }
    }

    #[test]
    fn recomputes_area_and_lloyd_only_for_affected_cells() {
        use crate::loss::CELL_LOSS_WORK;
        let (sites, boundary, target_areas, room_indices, _) = shape_a_setup();
        let weights = LossWeights { w_area: 1., w_lloyd: 1., ..Default::default() };
        let context = LocalGradContext::new(&sites, &boundary, &target_areas, &room_indices, &weights);
        let mut moved = sites.clone();
        moved[11][0] += 1e-6;
        let affected = context.affected_cells(&moved, 11);
        assert!(affected.len() > 1 && affected.len() < sites.len());
        CELL_LOSS_WORK.set([0; 2]);
        let actual = context.local_total(&moved, 11);
        assert!(actual.is_finite() && actual > 0.);
        assert_eq!(CELL_LOSS_WORK.get(), [affected.len(); 2],
            "reuse area and centroid distance for unchanged cells");
    }

    fn shape_a_setup() -> (Vec<[f32; 2]>, Polygon<f64>, Vec<f64>, Vec<usize>, LossWeights) {
        use geo::Area;
        let boundary = shapes::by_name("shape_a").unwrap().polygon();
        let ratios = [0.5f64, 0.3, 0.1, 0.1];
        let area = boundary.unsigned_area();
        let target_areas: Vec<f64> = ratios.iter().map(|r| area * r).collect();
        let sites = init::initialize_sites(&boundary, 40, 777);
        let room_indices = init::kmeans_labels(&sites, ratios.len(), 777);
        let w = LossWeights { w_wall: 2.5, w_area: 20.0, w_lloyd: 2.1, w_topo: 1.5, w_bb: 0.0, w_cell: 0.0, ..Default::default() };
        (sites, boundary, target_areas, room_indices, w)
    }

    #[test]
    fn local_grad_wall_term_matches_global() {
        let (sites, boundary, ta, ri, _) = shape_a_setup();
        // only the wall (alignment) term active
        let w = LossWeights {
            w_wall: 5.0, w_area: 0.0, w_lloyd: 0.0, w_topo: 0.0, w_bb: 0.0, w_cell: 0.0,
            ..Default::default()
        };
        let global = floor_plan_loss(&sites, &boundary, &ta, &ri, &w, None).total;
        assert!(global > 0.0, "precondition: the global wall loss must be positive");
        let ctx = LocalGradContext::new(&sites, &boundary, &ta, &ri, &w);
        // the FD path must see the same loss as the global path...
        assert!(
            (ctx.base_total() - global).abs() < 1e-4,
            "grad_local base_total {} must include the wall term and match global {}",
            ctx.base_total(),
            global
        );
        // ...and the term must actually drive a finite, non-zero gradient
        let g = ctx.gradients(&sites);
        let nonzero = g.iter().any(|c| c[0].abs() > 0.0 || c[1].abs() > 0.0);
        let finite = g.iter().all(|c| c[0].is_finite() && c[1].is_finite());
        assert!(nonzero && finite, "the wall term must drive a finite, non-zero local gradient");
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
    #[test]
    fn cached_shared_terms_match_full_six_room_objective(){
        use geo::Area;
        let fixture:serde_json::Value=serde_json::from_str(include_str!("../../web-tests/fixtures/room-program-six.json")).unwrap();
        let circles:Vec<[f64;3]>=fixture["rooms"].as_array().unwrap().iter().map(|r|[r["center"][0].as_f64().unwrap(),r["center"][1].as_f64().unwrap(),r["radius"].as_f64().unwrap()]).collect();
        let b=shapes::by_name("shape_e").unwrap().polygon();let mut p=RoomProgram::new(&b,&circles).unwrap();p.set_connections(&[[0,1],[2,4]],0.1).unwrap();
        let (sites,labels)=p.initialize(&b,80,4971).unwrap();let targets:Vec<_>=p.ratios().iter().map(|r|r*b.unsigned_area()).collect();
        let w=LossWeights{wall_mode:WallLossMode::CosineSquared,w_wall:1.,w_jog:0.35,w_area:0.15/0.49,w_lloyd:0.01/5.3,w_topo:0.2/4.,..Default::default()};
        let ctx=LocalGradContext::new_with_program(&sites,&b,&targets,&labels,&w,Some(&p));
        let mut uncached=LocalGradContext::new_with_program(&sites,&b,&targets,&labels,&w,Some(&p));
        uncached.shared_cells=None;
        uncached.wall_shape=None;
        let full=crate::loss::floor_plan_loss_with_program(&sites,&b,&targets,&labels,&w,None,Some(&p));
        assert!((ctx.base_total()-full.total).abs()<1e-4);
        for i in 0..sites.len(){for axis in 0..2{for sign in [-1.,1.]{
            let mut moved=sites.clone();moved[i][axis]+=sign*1e-6;
            let full=crate::loss::floor_plan_loss_with_program(&moved,&b,&targets,&labels,&w,None,Some(&p));
            let local=ctx.local_total(&moved,i);
            assert_eq!(local.to_bits(),uncached.local_total(&moved,i).to_bits(),"site{i} axis{axis} sign{sign}: cached ownership must preserve each finite-difference sample exactly");
            assert!((local-full.total).abs()<1e-4,"site{i} axis{axis} sign{sign} must preserve the full wall/jog/door objective");
        }}}
        let mut moved=sites.clone();moved[11][0]+=1e-3;
        let full=crate::loss::floor_plan_loss_with_program(&moved,&b,&targets,&labels,&w,None,Some(&p));
        assert!((ctx.local_total(&moved,11)-full.total).abs()<1e-4,"incremental evaluation must follow a real geometric change too");
        assert!((full.total-ctx.base_total()).abs()>1e-4,"the probe must exercise an actual objective change");
    }

}
