//! Forward geometry: Voronoi diagram of the sites, clipped to the boundary,
//! with clipped pieces paired to sites by the literal `loss.py` algorithm
//! (positional zip + pop-on-containment). The pairing is order-sensitive:
//! when a clipped cell splits into a MultiPolygon, every later position
//! shifts by one and the final piece drops out of the pairing — exactly as
//! in Python.
//!
//! `GeosOrderHint` carries the two GEOS-internal orderings that the Python
//! pairing depends on but that no reimplementation can recompute: the raw
//! cell iteration order and, at MultiPolygon splits, the piece iteration
//! order (given as piece areas). It is injected by the checkpoint tests;
//! standalone runs pass `None` and get natural site order.

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

/// Voronoi cell polygon per site (site order), clipped to a box that
/// generously contains both the boundary and all sites. The box only matters
/// through `cell ∩ boundary`, which is identical for any box ⊇ boundary.
pub fn raw_cells_per_site(sites_f64: &[[f64; 2]], boundary: &Polygon<f64>) -> Vec<Polygon<f64>> {
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
    let voronoi = VoronoiBuilder::default()
        .set_sites(
            sites_f64
                .iter()
                .map(|&[x, y]| VPoint { x, y })
                .collect::<Vec<_>>(),
        )
        .set_bounding_box(BoundingBox::new(center, max_x - min_x, max_y - min_y))
        .build()
        .expect("voronoi construction failed");

    voronoi
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
fn snap_cells(cells: &mut [Polygon<f64>]) {
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
    use geo::Area;
    let n = sites_f64.len();
    let mut cells_sorted: Vec<Polygon<f64>> = Vec::with_capacity(n);
    let mut n_pieces = 0usize;
    let mut split_positions = Vec::new();
    for (i, &[x, y]) in sites_f64.iter().enumerate() {
        let raw = &cell_of_site[i];
        let inter: MultiPolygon<f64> = raw.intersection(boundary);
        let pieces: Vec<Polygon<f64>> = inter.0;
        n_pieces += pieces.len().max(1);
        let cell = match pieces.len() {
            0 => empty_cell(),
            1 => pieces.into_iter().next().unwrap(),
            _ => {
                split_positions.push((i, pieces.len()));
                let site = Point::new(x, y);
                // prefer the piece containing the site; else (site outside
                // boundary) the largest piece. Tuple key (contains, area)
                // sorts a containing piece above all, then by area.
                pieces
                    .into_iter()
                    .max_by(|a, b| {
                        let key = |p: &Polygon<f64>| (p.contains(&site), p.unsigned_area());
                        key(a).partial_cmp(&key(b)).unwrap()
                    })
                    .unwrap_or_else(empty_cell)
            }
        };
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
