//! Forward geometry: Voronoi diagram of the sites, clipped to the boundary.
//! Two cell→site pairings live here, picked by whether a `GeosOrderHint` is
//! supplied:
//!
//! - Checkpoint path (hint present): the literal `loss.py` algorithm —
//!   positional zip + pop-on-containment. It is order-sensitive: when a clipped
//!   cell splits into a MultiPolygon, every later position shifts by one and the
//!   final piece drops out of the pairing — exactly as in Python. Used only by
//!   the fixture checkpoint tests.
//! - Standalone path (`hint` is `None` — the CLI / GIF / outcome runs): the
//!   direct voronoice site→cell mapping (each site keeps the cell it generated),
//!   which is order-independent and robust around splits. See
//!   `compute_cells_direct`.
//!
//! `GeosOrderHint` carries the two GEOS-internal orderings the Python pairing
//! depends on but that no reimplementation can recompute: the raw cell
//! iteration order and, at MultiPolygon splits, the piece iteration order
//! (given as piece areas). It is injected by the checkpoint tests only.

use geo::{Contains, LineString, MultiPolygon, Point, Polygon};
use geo_booleanop::boolean::BooleanOp;
use voronoice::{BoundingBox, Point as VPoint, VoronoiBuilder};

pub struct GeosOrderHint<'a> {
    /// raw cell position -> generating site index
    pub cell_order: &'a [usize],
    /// (raw position, piece areas in GEOS iteration order) per splitting cell
    pub split_pieces: &'a [(usize, Vec<f64>)],
}

pub struct CellGeometry {
    /// Clipped cell paired to each site, in site order. A site whose
    /// containment search finds no remaining raw cell is skipped, exactly as
    /// in Python — the list may then be shorter than the site count.
    pub cells_sorted: Vec<Polygon<f64>>,
    pub n_raw_cells: usize,
    pub n_pieces: usize,
    pub split_positions: Vec<(usize, usize)>,
}

pub fn empty_cell() -> Polygon<f64> {
    Polygon::new(LineString::new(vec![]), vec![])
}

pub fn is_empty_cell(cell: &Polygon<f64>) -> bool {
    cell.exterior().0.is_empty()
}

/// Build the voronoice diagram in a box that generously contains both the
/// boundary and all sites. The box only matters through `cell ∩ boundary`,
/// which is identical for any box ⊇ boundary.
fn build_voronoi(sites_f64: &[[f64; 2]], boundary: &Polygon<f64>) -> voronoice::Voronoi {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for c in boundary.exterior().coords() {
        min_x = min_x.min(c.x);
        min_y = min_y.min(c.y);
        max_x = max_x.max(c.x);
        max_y = max_y.max(c.y);
    }
    for &[x, y] in sites_f64 {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let span = (max_x - min_x).max(max_y - min_y).max(1e-9);
    min_x -= span;
    min_y -= span;
    max_x += span;
    max_y += span;

    let center = VPoint {
        x: (min_x + max_x) / 2.0,
        y: (min_y + max_y) / 2.0,
    };
    VoronoiBuilder::default()
        .set_sites(
            sites_f64
                .iter()
                .map(|&[x, y]| VPoint { x, y })
                .collect::<Vec<_>>(),
        )
        .set_bounding_box(BoundingBox::new(center, max_x - min_x, max_y - min_y))
        .build()
        .expect("voronoi construction failed")
}

/// Voronoi cell polygon per site (site order).
pub fn raw_cells_per_site(sites_f64: &[[f64; 2]], boundary: &Polygon<f64>) -> Vec<Polygon<f64>> {
    build_voronoi(sites_f64, boundary)
        .iter_cells()
        .map(|cell| {
            Polygon::new(
                LineString::from(
                    cell.iter_vertices()
                        .map(|v| (v.x, v.y))
                        .collect::<Vec<_>>(),
                ),
                vec![],
            )
        })
        .collect()
}

/// Voronoi neighbor site indices per site (the sites sharing a Voronoi edge).
/// Stable under a tiny site perturbation, so the local-gradient path caches
/// this once per step and reuses it for all 2N perturbations.
pub fn site_neighbors(sites_f64: &[[f64; 2]], boundary: &Polygon<f64>) -> Vec<Vec<usize>> {
    let voronoi = build_voronoi(sites_f64, boundary);
    let mut neighbors = vec![Vec::new(); sites_f64.len()];
    for cell in voronoi.iter_cells() {
        neighbors[cell.site()] = cell.iter_neighbors().collect();
    }
    neighbors
}

/// Clip one raw Voronoi cell to the boundary, returning the chosen piece and
/// the number of intersection pieces. On a MultiPolygon split, keep the piece
/// containing the site, or — if the site has drifted outside the boundary —
/// the largest piece (see `compute_cells_direct`).
pub fn clip_cell(raw: &Polygon<f64>, boundary: &Polygon<f64>, site: [f64; 2]) -> (Polygon<f64>, usize) {
    use geo::Area;
    let inter: MultiPolygon<f64> = raw.intersection(boundary);
    let pieces: Vec<Polygon<f64>> = inter.0;
    let n = pieces.len();
    let cell = match n {
        0 => empty_cell(),
        1 => pieces.into_iter().next().unwrap(),
        _ => {
            let s = Point::new(site[0], site[1]);
            pieces
                .into_iter()
                .max_by(|a, b| {
                    let key = |p: &Polygon<f64>| (p.contains(&s), p.unsigned_area());
                    key(a).partial_cmp(&key(b)).unwrap()
                })
                .unwrap_or_else(empty_cell)
        }
    };
    (cell, n)
}

/// True iff the raw Voronoi cell lies entirely inside the boundary, so clipping
/// it is the identity (`cell ∩ boundary == cell`) and the boolean intersection
/// can be skipped. SOUND but conservative: it may return `false` for a cell that
/// is in fact interior (a missed skip — never wrong), but it never returns
/// `true` for a cell the boundary actually cuts. Used only by the local demo
/// gradient path (`grad_local`); the exact CLI path keeps `clip_cell`.
pub(crate) fn cell_inside_boundary(raw: &Polygon<f64>, boundary: &Polygon<f64>) -> bool {
    let pts = &raw.exterior().0;
    if pts.len() < 3 {
        return false;
    }
    // cell bounding box
    let (mut cx0, mut cy0, mut cx1, mut cy1) =
        (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for c in pts {
        cx0 = cx0.min(c.x);
        cy0 = cy0.min(c.y);
        cx1 = cx1.max(c.x);
        cy1 = cy1.max(c.y);
    }
    // If any boundary edge's bbox overlaps the cell bbox, the boundary may cut
    // through the cell — bail to the real clip (conservative: never a false
    // accept, only a missed skip).
    for e in boundary.exterior().lines() {
        let (ex0, ex1) = (e.start.x.min(e.end.x), e.start.x.max(e.end.x));
        let (ey0, ey1) = (e.start.y.min(e.end.y), e.start.y.max(e.end.y));
        if cx0 <= ex1 && cx1 >= ex0 && cy0 <= ey1 && cy1 >= ey0 {
            return false;
        }
    }
    // No boundary edge is near the cell, so it is wholly inside or wholly
    // outside; one point-in-polygon settles which.
    boundary.contains(&Point::new(pts[0].x, pts[0].y))
}

/// Welds floating-point-twin vertices across the clipped pieces.
///
/// Where a shared Voronoi edge crosses the boundary, the seam intersection
/// point is computed once per neighboring cell with the segment endpoints in
/// opposite order, so the two results can differ in the last bits. Those
/// near-duplicate vertices keep the room unions from dissolving internal
/// edges. Snapping to a canonical representative within 1e-12 (nine orders
/// below any real feature size; displacement is invisible at the f32 loss
/// precision) restores the bitwise-shared edges that GEOS gets for free from
/// its single noding step.
pub(crate) fn snap_cells(cells: &mut [Polygon<f64>]) {
    use std::collections::HashMap;
    const TOL: f64 = 1e-12;
    let mut canon: HashMap<(i64, i64), Vec<(f64, f64)>> = HashMap::new();

    fn canonical(
        canon: &mut HashMap<(i64, i64), Vec<(f64, f64)>>,
        x: f64,
        y: f64,
    ) -> (f64, f64) {
        let kx = (x / TOL).round() as i64;
        let ky = (y / TOL).round() as i64;
        for dx in -1..=1i64 {
            for dy in -1..=1i64 {
                if let Some(pts) = canon.get(&(kx + dx, ky + dy)) {
                    for &(px, py) in pts {
                        if (px - x).abs() <= TOL && (py - y).abs() <= TOL {
                            return (px, py);
                        }
                    }
                }
            }
        }
        canon.entry((kx, ky)).or_default().push((x, y));
        (x, y)
    }

    for cell in cells.iter_mut() {
        if is_empty_cell(cell) {
            continue;
        }
        let pts = &cell.exterior().0;
        let open = &pts[..pts.len() - 1];
        let mut snapped: Vec<(f64, f64)> = Vec::with_capacity(open.len());
        for c in open {
            let p = canonical(&mut canon, c.x, c.y);
            if snapped.last() != Some(&p) {
                snapped.push(p);
            }
        }
        while snapped.len() > 1 && snapped.first() == snapped.last() {
            snapped.pop();
        }
        *cell = if snapped.len() < 3 {
            empty_cell()
        } else {
            Polygon::new(LineString::from(snapped), vec![])
        };
    }
}

/// Reorders intersection pieces to match the GEOS piece order recorded in the
/// fixture (greedy nearest-area matching; areas at a split differ by orders
/// of magnitude, so the matching is unambiguous).
fn order_pieces_by_area(mut pieces: Vec<Polygon<f64>>, areas: &[f64]) -> Vec<Polygon<f64>> {
    use geo::Area;
    let mut ordered = Vec::with_capacity(pieces.len());
    for &target in areas {
        if pieces.is_empty() {
            break;
        }
        let (best, _) = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (i, (p.unsigned_area() - target).abs()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        ordered.push(pieces.remove(best));
    }
    ordered.extend(pieces);
    ordered
}

pub fn compute_cells(
    sites: &[[f32; 2]],
    boundary: &Polygon<f64>,
    hint: Option<&GeosOrderHint>,
) -> CellGeometry {
    let n = sites.len();
    let sites_f64: Vec<[f64; 2]> = sites.iter().map(|&[x, y]| [x as f64, y as f64]).collect();

    let cell_of_site = raw_cells_per_site(&sites_f64, boundary);

    // Standalone runs (GIF/CLI/outcome loop) use the direct voronoice
    // site->cell mapping, which is robust and order-independent. The
    // order-dependent zip-pop pairing below exists ONLY to bit-replicate
    // loss.py's GEOS-order-dependent piece dropping for the fixture checkpoint
    // (where the hint is injected); without the hint that re-pairing could
    // mis-assign cells around a MultiPolygon split (shape_b: 10/40 sites).
    if hint.is_none() {
        return compute_cells_direct(&sites_f64, boundary, &cell_of_site);
    }

    let identity: Vec<usize> = (0..n).collect();
    let order: &[usize] = hint.map(|h| h.cell_order).unwrap_or(&identity);

    // clip cells to the boundary in raw order, expanding MultiPolygon splits
    // positionally — `raws` is NOT expanded, mirroring loss.py
    let mut cells: Vec<Polygon<f64>> = Vec::with_capacity(n + 4);
    let mut raws: Vec<&Polygon<f64>> = Vec::with_capacity(n);
    let mut split_positions = Vec::new();
    for (pos, &site_idx) in order.iter().enumerate() {
        let raw = &cell_of_site[site_idx];
        // Martinez-Rueda boolean op: pure f64, preserves uncut vertices
        // bitwise — neighboring clipped cells keep bitwise-shared edges, so
        // the room unions downstream weld without slivers.
        let inter: MultiPolygon<f64> = raw.intersection(boundary);
        let mut pieces: Vec<Polygon<f64>> = inter.0;
        match pieces.len() {
            0 => cells.push(empty_cell()),
            1 => cells.push(pieces.pop().unwrap()),
            k => {
                split_positions.push((pos, k));
                if let Some(h) = hint {
                    if let Some((_, areas)) =
                        h.split_pieces.iter().find(|(p, _)| *p == pos)
                    {
                        pieces = order_pieces_by_area(pieces, areas);
                    }
                }
                cells.extend(pieces);
            }
        }
        raws.push(raw);
    }
    let n_pieces = cells.len();

    snap_cells(&mut cells);

    // literal zip-pop pairing: first remaining (cell, raw) whose raw cell
    // contains the site claims that positional pair
    let mut rem_cells = cells;
    let mut rem_raws = raws;
    let mut cells_sorted = Vec::with_capacity(n);
    for &[x, y] in &sites_f64 {
        let pt = Point::new(x, y);
        let len = rem_cells.len().min(rem_raws.len());
        if let Some(ci) = (0..len).find(|&ci| rem_raws[ci].contains(&pt)) {
            cells_sorted.push(rem_cells.remove(ci));
            rem_raws.remove(ci);
        }
    }

    CellGeometry {
        cells_sorted,
        n_raw_cells: cell_of_site.len(),
        n_pieces,
        split_positions,
    }
}

/// Direct voronoice site->cell pairing (no hint): each site i is the generator
/// of `cell_of_site[i]`, so we clip that cell and keep it for site i. At a
/// MultiPolygon split we keep the piece that geometrically contains the site;
/// if the site has drifted OUTSIDE the boundary (so no clipped piece contains
/// it — legitimate mid-optimization) we keep the LARGEST piece instead, so the
/// cell still covers its boundary region. Dropping it to empty there would
/// leave a hole at a concave boundary notch (shape_duck ~iter 11). No
/// containment re-search, so the assignment cannot drift around a split.
fn compute_cells_direct(
    sites_f64: &[[f64; 2]],
    boundary: &Polygon<f64>,
    cell_of_site: &[Polygon<f64>],
) -> CellGeometry {
    let n = sites_f64.len();
    let mut cells_sorted: Vec<Polygon<f64>> = Vec::with_capacity(n);
    let mut n_pieces = 0usize;
    let mut split_positions = Vec::new();
    for (i, &site) in sites_f64.iter().enumerate() {
        // prefer the piece containing the site; else (site outside boundary)
        // the largest piece — see `clip_cell`.
        let (cell, np) = clip_cell(&cell_of_site[i], boundary, site);
        n_pieces += np.max(1);
        if np > 1 {
            split_positions.push((i, np));
        }
        cells_sorted.push(cell);
    }

    snap_cells(&mut cells_sorted);

    CellGeometry {
        cells_sorted,
        n_raw_cells: cell_of_site.len(),
        n_pieces,
        split_positions,
    }
}

/// Render-only cells: every piece of each site's Voronoi cell ∩ boundary, tagged
/// with the site index. Unlike `compute_cells` (which keeps ONE piece per site
/// for the bit-exact loss pairing), this keeps ALL pieces, so the rendered plan
/// covers the whole boundary — matching the Python renderer (`generator.py`,
/// which iterates `cell.geoms` for a MultiPolygon). The loss/gradient path is
/// unaffected.
pub fn render_cells(sites: &[[f32; 2]], boundary: &Polygon<f64>) -> Vec<(usize, Polygon<f64>)> {
    let sites_f64: Vec<[f64; 2]> = sites.iter().map(|&[x, y]| [x as f64, y as f64]).collect();
    let raw = raw_cells_per_site(&sites_f64, boundary);
    let mut out = Vec::with_capacity(raw.len());
    for (i, raw_cell) in raw.iter().enumerate() {
        // keep EVERY piece of this cell's intersection with the boundary, so the
        // pieces of all cells tile the boundary with no uncovered gap.
        let inter: MultiPolygon<f64> = raw_cell.intersection(boundary);
        for piece in inter.0 {
            if !is_empty_cell(&piece) {
                out.push((i, piece));
            }
        }
    }
    out
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::loss::LossWeights;
    use crate::optim::AdamW;
    use crate::{config, grad, init, shapes};
    use geo::Area;

    #[test]
    fn render_cells_covers_boundary_at_multipart_split() {
        let cfg = config::by_name("shape_a").unwrap();
        let boundary = shapes::by_name("shape_a").unwrap().polygon();
        let barea = boundary.unsigned_area();
        let ta: Vec<f64> = cfg.area_ratio.iter().map(|r| barea * r).collect();
        let w = LossWeights {
            w_wall: cfg.w_wall, w_area: cfg.w_area, w_lloyd: cfg.w_lloyd,
            w_topo: cfg.w_topo, w_bb: cfg.w_bb, w_cell: cfg.w_cell,
        };
        let mut sites = init::initialize_sites(&boundary, cfg.num_sites, 777);
        let ri = init::kmeans_labels(&sites, cfg.area_ratio.len(), 777);
        let mut opt = AdamW::new(sites.len(), cfg.lr_initial);

        // advance until a cell ∩ boundary splits into comparable pieces, leaving
        // a significant one-piece coverage gap (deterministic trajectory; break
        // early to keep the test fast in debug mode).
        let mut split_sites = None;
        for _ in 0..30 {
            let cov: f64 = compute_cells(&sites, &boundary, None)
                .cells_sorted.iter().map(|c| c.unsigned_area()).sum();
            if barea - cov > 1.5e-3 { split_sites = Some(sites.clone()); break; }
            let g = grad::finite_difference_grads(&sites, &boundary, &ta, &ri, &w, None);
            opt.step(&mut sites, &g);
        }
        let worst_sites = split_sites.expect("expected a significant one-piece coverage gap within 30 iters");

        // at that config, the one-piece compute_cells path leaves the gap...
        let geom = compute_cells(&worst_sites, &boundary, None);
        let one_piece: f64 = geom.cells_sorted.iter().map(|c| c.unsigned_area()).sum();
        assert!(barea - one_piece > 1e-3, "one-piece gap should be present: {}", barea - one_piece);

        // ...but render_cells keeps ALL pieces -> covers the whole boundary.
        let pieces = render_cells(&worst_sites, &boundary);
        let all: f64 = pieces.iter().map(|(_, p)| p.unsigned_area()).sum();
        assert!((barea - all).abs() < 1e-5, "render_cells must cover the boundary; gap = {}", barea - all);
        assert!(pieces.len() > geom.cells_sorted.len(), "the split must expand the piece count: {} vs {}", pieces.len(), geom.cells_sorted.len());
    }
}

#[cfg(test)]
mod interior_tests {
    use super::*;
    use crate::{init, shapes};
    use geo::Area;

    // KR1: `cell_inside_boundary` is a SOUND trivial-accept predicate — every
    // cell it calls "inside" must have clip-identity (the boolean clip leaves the
    // area unchanged), and both classes must occur on real geometry (it is not
    // vacuously all-true or all-false). The sound direction is the safety
    // property: a `true` that is not actually clip-identity would let the caller
    // skip a clip the boundary needed, corrupting the cell.
    #[test]
    fn cell_inside_boundary_is_sound_and_nonvacuous() {
        let boundary = shapes::by_name("shape_a").unwrap().polygon();
        let sites = init::initialize_sites(&boundary, 40, 777);
        let sites_f64: Vec<[f64; 2]> = sites.iter().map(|&[x, y]| [x as f64, y as f64]).collect();
        let raw = raw_cells_per_site(&sites_f64, &boundary);

        let (mut n_true, mut n_false) = (0usize, 0usize);
        for (i, rc) in raw.iter().enumerate() {
            let pred = cell_inside_boundary(rc, &boundary);
            // reference truth: clipping is identity iff the cell is fully inside
            let (clipped, _) = clip_cell(rc, &boundary, sites_f64[i]);
            let ra = rc.unsigned_area();
            let ca = clipped.unsigned_area();
            let identity = ra > 0.0 && ((ra - ca).abs() / ra < 1e-9);
            if pred {
                n_true += 1;
                assert!(
                    identity,
                    "cell {i}: predicate said inside but clip changed area (raw {ra}, clipped {ca})"
                );
            } else {
                n_false += 1;
            }
        }
        assert!(n_true > 0, "predicate never returned true — it recognizes no interior cells");
        assert!(n_false > 0, "predicate never returned false — it recognizes no boundary cells");
    }
}
