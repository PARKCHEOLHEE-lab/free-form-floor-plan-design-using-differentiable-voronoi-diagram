//! Loss components, ported 1:1 from `loss.py`.
//!
//! Precision contract: torch's default dtype is float32, so every place
//! Python materializes a tensor is an f64 -> f32 cast, and every tensor op
//! happens in f32. Geometry (areas, coordinates, unions) stays f64 until the
//! moment Python would have cast it. The comments mark each boundary.

use geo::{Area, LineString, MultiPolygon, Polygon};
use geo_booleanop::boolean::BooleanOp;

/// Groups the paired cells by room index, mirroring
/// `rooms_group[room_index].append(cell)` over `zip(cells_sorted, room_indices)`
/// (zip truncates to the shorter list, as in Python).
pub fn rooms_group<'a>(
    cells_sorted: &'a [Polygon<f64>],
    room_indices: &[usize],
) -> Vec<Vec<&'a Polygon<f64>>> {
    let n_rooms = {
        let mut uniq: Vec<usize> = room_indices.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        uniq.len()
    };
    let mut groups: Vec<Vec<&Polygon<f64>>> = vec![Vec::new(); n_rooms];
    for (cell, &ri) in cells_sorted.iter().zip(room_indices) {
        groups[ri].push(cell);
    }
    groups
}

/// `ops.unary_union(room_group)` equivalent: f64 set-union of the group's
/// polygons via Martinez-Rueda (pure f64, no coordinate re-quantization).
/// The clipped cells of neighboring sites share edges bitwise (see
/// `voronoi.rs`), so the fold welds them exactly, matching GEOS's
/// unary_union semantics. Piece order may differ from GEOS; every consumer
/// folds pieces through order-insensitive f64 accumulation, so the result is
/// observationally identical after the f32 casts.
pub fn union_group(group: &[&Polygon<f64>]) -> MultiPolygon<f64> {
    let mut acc = MultiPolygon::<f64>::new(vec![]);
    for poly in group {
        if poly.exterior().0.is_empty() {
            continue;
        }
        acc = acc.union(*poly);
    }
    acc
}

/// One ring's `torch.abs(t1 - t2).sum().item()`: coordinates cast to f32
/// (tensor materialization), rolled difference and |.| summed in f32 in
/// row-major element order, widened to f64 by `.item()`.
fn ring_wall_sum(ring: &LineString<f64>) -> f32 {
    let pts = &ring.0;
    let n = pts.len().saturating_sub(1); // coords[:-1] drops the closing dup
    if n == 0 {
        return 0.0;
    }
    let t: Vec<[f32; 2]> = pts[..n].iter().map(|c| [c.x as f32, c.y as f32]).collect();
    let mut s: f32 = 0.0;
    for i in 0..n {
        let j = (i + 1) % n; // torch.roll(t1, -1, 0)
        s += (t[i][0] - t[j][0]).abs();
        s += (t[i][1] - t[j][1]).abs();
    }
    s
}

pub fn compute_wall_loss(room_unions: &[MultiPolygon<f64>], w_wall: f64) -> f32 {
    // Python accumulates the per-ring f32 sums into a Python float (f64) ...
    let mut loss_wall: f64 = 0.0;
    for room_union in room_unions {
        for room in room_union {
            loss_wall += ring_wall_sum(room.exterior()) as f64;
            for interior in room.interiors() {
                loss_wall += ring_wall_sum(interior) as f64;
            }
        }
    }
    // ... then `torch.tensor(loss_wall)` casts to f32 and `*= w_wall` stays f32
    (loss_wall as f32) * (w_wall as f32)
}

/// Local-frame wall loss: like `compute_wall_loss`, but every room-boundary edge
/// is measured as a *rotated* taxicab length in the frame of the nearest
/// domain-boundary segment (its orientation φ). An edge parallel or
/// perpendicular to the local boundary costs its Euclidean length; a 45°-skewed
/// edge costs √2×. Reduces exactly to `compute_wall_loss` when every φ = 0.
///
/// PARITY (#3): this term is Rust-only — `loss.py` has no `wall_local` and sums
/// only the other six terms. Python↔Rust parity therefore holds only at
/// `w_wall_local = 0`; every golden parity trace is generated with it off.
pub fn compute_wall_local_loss(
    room_unions: &[MultiPolygon<f64>],
    boundary: &Polygon<f64>,
    w_wall_local: f64,
) -> f32 {
    let mut loss: f64 = 0.0;
    for room_union in room_unions {
        for room in room_union {
            loss += ring_wall_local_sum(room.exterior(), boundary) as f64;
            for interior in room.interiors() {
                loss += ring_wall_local_sum(interior, boundary) as f64;
            }
        }
    }
    (loss as f32) * (w_wall_local as f32)
}

/// Experimental "Blend" wall_local (`WallLocalMode::Blend`): continuous
/// distance-weighted, length-independent pure alignment (Codex #1 + #6). Sums the
/// per-edge `edge_alignment_penalty_sampled` over every room-boundary ring.
fn compute_wall_local_loss_blend(
    room_unions: &[MultiPolygon<f64>],
    boundary: &Polygon<f64>,
    w_wall_local: f64,
) -> f32 {
    let mut loss: f64 = 0.0;
    for room_union in room_unions {
        for room in room_union {
            loss += ring_wall_local_blend_sum(room.exterior(), boundary) as f64;
            for interior in room.interiors() {
                loss += ring_wall_local_blend_sum(interior, boundary) as f64;
            }
        }
    }
    (loss as f32) * (w_wall_local as f32)
}

/// Dispatch the wall_local term by `WallLocalMode` (0 when `w_wall_local ≤ 0`).
/// The SINGLE source used by both `floor_plan_loss` (native/global path) and
/// `grad_local::total_from` (demo/local path), so the mode reaches every code
/// path — otherwise the demo would silently ignore `Blend`.
pub(crate) fn wall_local_term(
    room_unions: &[MultiPolygon<f64>],
    boundary: &Polygon<f64>,
    w: &LossWeights,
) -> f32 {
    if w.w_wall_local <= 0.0 {
        return 0.0;
    }
    match w.wall_local_mode {
        WallLocalMode::Nearest => compute_wall_local_loss(room_unions, boundary, w.w_wall_local),
        WallLocalMode::Blend => compute_wall_local_loss_blend(room_unions, boundary, w.w_wall_local),
    }
}

/// Per-edge local-frame ALIGNMENT penalty for the Blend mode, blended continuously
/// over nearby boundary segments. Each segment `k` contributes a pure-alignment
/// penalty `g_k = f(θ−φ_k) − 1` (0 when the edge is parallel/perpendicular to
/// segment `k`, up to √2−1 at 45°), combined by distance weight `w_k = 1/(d_k²+ε)`:
/// `penalty = Σ w_k g_k / Σ w_k`. Blending the SCALAR penalties (not the angle)
/// makes it CONTINUOUS in the edge position (Codex #1 — an averaged angle has
/// cross-field singularities); the `−1` makes it length-independent pure alignment
/// (#6). Exterior + interior rings vote (#4); zero-length edges/segments skipped (#7).
fn edge_alignment_penalty(mx: f64, my: f64, dx: f32, dy: f32, boundary: &Polygon<f64>) -> f32 {
    let l = ((dx * dx + dy * dy) as f64).sqrt();
    if l == 0.0 {
        return 0.0;
    }
    let mut sum_w = 0.0_f64;
    let mut sum_wg = 0.0_f64;
    let segments = boundary
        .exterior()
        .lines()
        .chain(boundary.interiors().iter().flat_map(|ring| ring.lines()));
    for line in segments {
        let (ax, ay) = (line.start.x, line.start.y);
        let (ex, ey) = (line.end.x - ax, line.end.y - ay);
        let len2 = ex * ex + ey * ey;
        if len2 == 0.0 {
            continue;
        }
        let t = (((mx - ax) * ex + (my - ay) * ey) / len2).clamp(0.0, 1.0);
        let (px, py) = (ax + t * ex, ay + t * ey);
        let d2 = (mx - px) * (mx - px) + (my - py) * (my - py);
        let w = 1.0 / (d2 + 1e-9);
        let phi = ey.atan2(ex);
        let (c, sn) = (phi.cos() as f32, phi.sin() as f32);
        let du = dx * c + dy * sn;
        let dv = -dx * sn + dy * c;
        let f = (du.abs() + dv.abs()) as f64 / l;
        sum_w += w;
        sum_wg += w * (f - 1.0);
    }
    if sum_w == 0.0 {
        return 0.0;
    }
    (sum_wg / sum_w) as f32
}

/// Blend edge penalty averaged over K=3 points along the edge (#5), so a long
/// edge spanning more than one orientation is not decided by its midpoint alone.
fn edge_alignment_penalty_sampled(
    xi: f64,
    yi: f64,
    xj: f64,
    yj: f64,
    dx: f32,
    dy: f32,
    boundary: &Polygon<f64>,
) -> f32 {
    const K: usize = 3;
    let mut sum = 0.0_f32;
    for s in 0..K {
        let t = (s as f64 + 0.5) / K as f64; // 1/6, 1/2, 5/6
        let mx = xi * (1.0 - t) + xj * t;
        let my = yi * (1.0 - t) + yj * t;
        sum += edge_alignment_penalty(mx, my, dx, dy, boundary);
    }
    sum / K as f32
}

/// Sum of per-edge Blend alignment penalties over a ring (mirrors
/// `ring_wall_local_sum` but uses the continuous sampled blend).
fn ring_wall_local_blend_sum(ring: &LineString<f64>, boundary: &Polygon<f64>) -> f32 {
    let pts = &ring.0;
    let n = pts.len().saturating_sub(1);
    if n == 0 {
        return 0.0;
    }
    let t: Vec<[f32; 2]> = pts[..n].iter().map(|c| [c.x as f32, c.y as f32]).collect();
    let mut s: f32 = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = t[i][0] - t[j][0];
        let dy = t[i][1] - t[j][1];
        s += edge_alignment_penalty_sampled(pts[i].x, pts[i].y, pts[j].x, pts[j].y, dx, dy, boundary);
    }
    s
}

/// Orientation (radians) of the boundary segment nearest to point `(mx, my)`.
/// The rotated-L1 below only uses this angle mod 90°, so segment direction sign
/// is irrelevant.
fn nearest_boundary_angle(mx: f64, my: f64, boundary: &Polygon<f64>) -> f64 {
    let mut best_d2 = f64::INFINITY;
    let mut ang = 0.0;
    for line in boundary.exterior().lines() {
        let (ax, ay) = (line.start.x, line.start.y);
        let (ex, ey) = (line.end.x - ax, line.end.y - ay);
        let len2 = ex * ex + ey * ey;
        if len2 == 0.0 {
            continue; // #7: a zero-length segment carries no orientation — skip it
        }
        // projection parameter of (mx,my) onto the segment, clamped to [0,1]
        let t = (((mx - ax) * ex + (my - ay) * ey) / len2).clamp(0.0, 1.0);
        let (px, py) = (ax + t * ex, ay + t * ey);
        let d2 = (mx - px) * (mx - px) + (my - py) * (my - py);
        if d2 < best_d2 {
            best_d2 = d2;
            ang = ey.atan2(ex);
        }
    }
    ang
}

/// `ring_wall_sum` measured in the local boundary frame: each edge vector is
/// rotated by −φ (φ = nearest-boundary orientation at the edge midpoint) before
/// the |Δu| + |Δv| taxicab sum. Mirrors `ring_wall_sum`'s f32-first casting, so
/// at φ = 0 it is bit-identical to `ring_wall_sum`.
fn ring_wall_local_sum(ring: &LineString<f64>, boundary: &Polygon<f64>) -> f32 {
    let pts = &ring.0;
    let n = pts.len().saturating_sub(1);
    if n == 0 {
        return 0.0;
    }
    let t: Vec<[f32; 2]> = pts[..n].iter().map(|c| [c.x as f32, c.y as f32]).collect();
    let mut s: f32 = 0.0;
    for i in 0..n {
        let j = (i + 1) % n; // torch.roll(t1, -1, 0)
        let dx = t[i][0] - t[j][0];
        let dy = t[i][1] - t[j][1];
        let mx = (pts[i].x + pts[j].x) * 0.5;
        let my = (pts[i].y + pts[j].y) * 0.5;
        let phi = nearest_boundary_angle(mx, my, boundary);
        let (c, sn) = (phi.cos() as f32, phi.sin() as f32);
        let du = dx * c + dy * sn; // edge rotated by −φ
        let dv = -dx * sn + dy * c;
        s += du.abs() + dv.abs();
    }
    s
}

pub fn compute_area_loss(
    cells_sorted: &[Polygon<f64>],
    target_areas: &[f64],
    room_indices: &[usize],
    w_area: f64,
) -> f32 {
    // f64 accumulation of shapely-style cell areas per room
    let mut current_areas: Vec<f64> = vec![0.0; target_areas.len()];
    for (cell, &ri) in cells_sorted.iter().zip(room_indices) {
        current_areas[ri] += cell.unsigned_area();
    }
    // torch.tensor(...) casts both vectors to f32
    let mut sum: f32 = 0.0;
    for (&cur, &tgt) in current_areas.iter().zip(target_areas) {
        sum += ((cur as f32) - (tgt as f32)).abs();
    }
    // loss_area **= 2; loss_area *= w_area  (both f32)
    sum * sum * (w_area as f32)
}

/// `compute_lloyd_loss`: zip(sites, cells) truncates to the shorter list;
/// empty cells are filtered; centroids and sites materialize as f32 tensors;
/// per-row L2 norm and the sum run in f32.
pub fn compute_lloyd_loss(
    cells_sorted: &[Polygon<f64>],
    sites: &[[f32; 2]],
    w_lloyd: f64,
) -> f32 {
    use geo::Centroid;
    let mut sum: f32 = 0.0;
    for (site, cell) in sites.iter().zip(cells_sorted) {
        if crate::voronoi::is_empty_cell(cell) {
            continue;
        }
        let c = cell.centroid().expect("nonempty cell has a centroid");
        let dx = (c.x() as f32) - site[0];
        let dy = (c.y() as f32) - site[1];
        sum += (dx * dx + dy * dy).sqrt();
    }
    // loss_lloyd **= 2; loss_lloyd *= w_lloyd  (both f32)
    sum * sum * (w_lloyd as f32)
}

/// `compute_topology_loss`: rooms whose union is a MultiPolygon (>1 piece)
/// add their piece count plus, for each member cell that does not intersect
/// the largest piece, the f64 distance from the largest piece's centroid to
/// that cell. Accumulation is a Python float (f64); the final value is cast
/// to f32, squared and weighted in f32.
pub fn compute_topology_loss(
    rooms_group: &[Vec<&Polygon<f64>>],
    room_unions: &[MultiPolygon<f64>],
    w_topo: f64,
) -> f32 {
    // `EuclideanDistance` is deprecated in geo 0.30 in favor of the `Distance`
    // trait, but that replacement only covers point-to-point in this version —
    // point-to-polygon distance (what `largest_centroid.distance(room)` needs)
    // still lives only on the deprecated trait here.
    #[allow(deprecated)]
    use geo::EuclideanDistance;
    use geo::{Centroid, Intersects};
    let mut loss_topo: f64 = 0.0;
    for (group, room_union) in rooms_group.iter().zip(room_unions) {
        if room_union.0.len() > 1 {
            // sorted(..., reverse=True)[0]: stable sort keeps the first of
            // equal areas, i.e. strictly-greater replaces
            let largest = room_union
                .iter()
                .max_by(|a, b| {
                    a.unsigned_area()
                        .partial_cmp(&b.unsigned_area())
                        .unwrap()
                })
                .expect("nonempty multipolygon");

            loss_topo += room_union.0.len() as f64;

            let largest_centroid = largest.centroid().expect("nonempty piece");
            for room in group {
                if !room.intersects(largest) && !crate::voronoi::is_empty_cell(room) {
                    #[allow(deprecated)]
                    let d = largest_centroid.euclidean_distance(*room);
                    loss_topo += d;
                }
            }
        }
    }
    let t = loss_topo as f32;
    t * t * (w_topo as f32)
}

/// `compute_bb_loss`: per room, union area over axis-aligned envelope area
/// accumulated in f64, then cast, squared and NEGATIVELY weighted in f32.
pub fn compute_bb_loss(room_unions: &[MultiPolygon<f64>], w_bb: f64) -> f32 {
    use geo::BoundingRect;
    let mut loss_bb: f64 = 0.0;
    for room_union in room_unions {
        let rect = room_union
            .bounding_rect()
            .expect("bb loss on an empty room union (Python raises here too)");
        loss_bb += room_union.unsigned_area() / (rect.width() * rect.height());
    }
    let b = loss_bb as f32;
    b * b * (-(w_bb) as f32)
}

/// `compute_cell_area_loss`: cells sorted by area ascending (empties
/// included), adjacent differences summed in f64, cast and weighted in f32.
pub fn compute_cell_area_loss(cells_sorted: &[Polygon<f64>], w_cell: f64) -> f32 {
    let mut areas: Vec<f64> = cells_sorted.iter().map(|c| c.unsigned_area()).collect();
    areas.sort_by(|a, b| a.total_cmp(b));
    let mut sum: f64 = 0.0;
    for pair in areas.windows(2) {
        sum += pair[1] - pair[0];
    }
    (sum as f32) * (w_cell as f32)
}

/// Which algorithm computes the local-frame wall-alignment term.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum WallLocalMode {
    /// Original (production default): nearest-boundary-segment φ, length-coupled
    /// rotated-L1. Bit-exactly reduces to the global wall loss at φ = 0.
    #[default]
    Nearest,
    /// Experimental, opt-in: continuous distance-weighted, length-independent
    /// pure alignment (Codex #1 continuity + #6 length-decoupling).
    Blend,
}

#[derive(Default)]
pub struct LossWeights {
    pub w_wall: f64,
    pub w_area: f64,
    pub w_lloyd: f64,
    pub w_topo: f64,
    pub w_bb: f64,
    pub w_cell: f64,
    /// Local-frame wall alignment (rotated-L1, nearest-boundary φ). Separate from
    /// `w_wall` so the global and local terms can be weighted independently.
    pub w_wall_local: f64,
    /// Selects the wall_local algorithm (default `Nearest` = production original).
    pub wall_local_mode: WallLocalMode,
}

pub struct LossBreakdown {
    pub total: f32,
    pub wall: f32,
    pub area: f32,
    pub lloyd: f32,
    pub topo: f32,
    pub bb: f32,
    pub cell: f32,
    pub wall_local: f32,
}

/// `FloorPlanLoss.forward`: builds the cell geometry, groups rooms, computes
/// each component only when its weight is positive (Python's `if w > 0`
/// guards), and sums the six f32 scalars left to right.
pub fn floor_plan_loss(
    sites: &[[f32; 2]],
    boundary: &Polygon<f64>,
    target_areas: &[f64],
    room_indices: &[usize],
    w: &LossWeights,
    hint: Option<&crate::voronoi::GeosOrderHint>,
) -> LossBreakdown {
    let geom = crate::voronoi::compute_cells(sites, boundary, hint);
    let groups = rooms_group(&geom.cells_sorted, room_indices);

    // The per-room union (Martinez-Rueda) is the heaviest geometry in the loss
    // after the Voronoi build, and wall/topo/bb each consume the *same* union.
    // Compute it once per group here and share it (it used to be recomputed
    // inside each component). The result is identical, so the f32 losses are
    // bitwise unchanged; this only removes the duplicate boolean ops.
    let unions: Vec<MultiPolygon<f64>> = if w.w_wall > 0.0 || w.w_topo > 0.0 || w.w_bb > 0.0 || w.w_wall_local > 0.0 {
        groups.iter().map(|g| union_group(g)).collect()
    } else {
        Vec::new()
    };

    let wall = if w.w_wall > 0.0 {
        compute_wall_loss(&unions, w.w_wall)
    } else {
        0.0
    };
    let area = if w.w_area > 0.0 {
        compute_area_loss(&geom.cells_sorted, target_areas, room_indices, w.w_area)
    } else {
        0.0
    };
    let lloyd = if w.w_lloyd > 0.0 {
        compute_lloyd_loss(&geom.cells_sorted, sites, w.w_lloyd)
    } else {
        0.0
    };
    let topo = if w.w_topo > 0.0 {
        compute_topology_loss(&groups, &unions, w.w_topo)
    } else {
        0.0
    };
    let bb = if w.w_bb > 0.0 {
        compute_bb_loss(&unions, w.w_bb)
    } else {
        0.0
    };
    let cell = if w.w_cell > 0.0 {
        compute_cell_area_loss(&geom.cells_sorted, w.w_cell)
    } else {
        0.0
    };
    let wall_local = wall_local_term(&unions, boundary, w);

    // loss = loss_wall + loss_area + loss_lloyd + loss_topo + loss_bb + loss_cell (+ loss_wall_local)
    let total = wall + area + lloyd + topo + bb + cell + wall_local;
    LossBreakdown {
        total,
        wall,
        area,
        lloyd,
        topo,
        bb,
        cell,
        wall_local,
    }
}

#[cfg(test)]
mod wall_local_tests {
    use super::*;

    /// A unit square rotated 45° — every boundary edge runs at ±45°, so the
    /// local frame φ near the center is ±45° (mod 90°).
    fn diamond_boundary() -> Polygon<f64> {
        Polygon::new(
            LineString::from(vec![(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]),
            vec![],
        )
    }

    /// A small square room centered at the origin; `rot45` rotates the SAME
    /// square 45° (edges become ±45°), so both variants have identical edge
    /// lengths and differ only in orientation relative to the local frame.
    fn square_room(rot45: bool) -> MultiPolygon<f64> {
        let s = 0.1;
        let d = s * std::f64::consts::SQRT_2;
        let pts = if rot45 {
            vec![(0.0, d), (-d, 0.0), (0.0, -d), (d, 0.0)]
        } else {
            vec![(s, s), (-s, s), (-s, -s), (s, -s)]
        };
        MultiPolygon::new(vec![Polygon::new(LineString::from(pts), vec![])])
    }

    #[test]
    fn wall_local_prefers_wall_parallel_to_local_diagonal_boundary() {
        let b = diamond_boundary();
        let aligned = compute_wall_local_loss(&[square_room(true)], &b, 1.0); // edges ∥/⟂ to 45° boundary
        let axis = compute_wall_local_loss(&[square_room(false)], &b, 1.0); // edges at 0°/90°
        assert!(
            aligned < axis,
            "wall_local must reward walls aligned to the local diagonal boundary: aligned={aligned} axis={axis}"
        );
    }

    #[test]
    fn wall_local_reduces_to_global_wall_loss_on_axis_aligned_boundary() {
        // Axis-aligned square boundary → every nearest-boundary angle is 0°/90°
        // (≡ 0 mod 90°), so the rotated-L1 collapses to the plain L1 and
        // compute_wall_local_loss must equal compute_wall_loss exactly.
        let b = Polygon::new(
            LineString::from(vec![(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]),
            vec![],
        );
        let room = square_room(false); // axis-aligned room
        let global = compute_wall_loss(&[room.clone()], 2.5);
        let local = compute_wall_local_loss(&[room], &b, 2.5);
        assert_eq!(
            local, global,
            "wall_local must reduce to the global wall loss on an axis-aligned boundary"
        );
    }

    fn shape_a_geom() -> (Vec<[f32; 2]>, Polygon<f64>, Vec<f64>, Vec<usize>) {
        let boundary = crate::shapes::by_name("shape_a").unwrap().polygon();
        let barea = boundary.unsigned_area();
        let ratios = [0.5, 0.3, 0.1, 0.1];
        let target_areas: Vec<f64> = ratios.iter().map(|r| barea * r).collect();
        let sites = crate::init::initialize_sites(&boundary, 40, 777);
        let room_indices = crate::init::kmeans_labels(&sites, ratios.len(), 777);
        (sites, boundary, target_areas, room_indices)
    }

    #[test]
    fn floor_plan_loss_includes_wall_local_when_weighted() {
        let (sites, boundary, ta, ri) = shape_a_geom();
        // every weight off but w_wall_local: total must come ENTIRELY from the
        // new term, and the term must be reported in the breakdown.
        let off = LossWeights { w_wall: 0.0, w_area: 0.0, w_lloyd: 0.0, w_topo: 0.0, w_bb: 0.0, w_cell: 0.0, w_wall_local: 0.0, ..Default::default() };
        let on = LossWeights { w_wall: 0.0, w_area: 0.0, w_lloyd: 0.0, w_topo: 0.0, w_bb: 0.0, w_cell: 0.0, w_wall_local: 5.0, ..Default::default() };
        let b0 = floor_plan_loss(&sites, &boundary, &ta, &ri, &off, None);
        let b1 = floor_plan_loss(&sites, &boundary, &ta, &ri, &on, None);
        assert!(
            b1.total > b0.total && b1.wall_local > 0.0,
            "w_wall_local>0 must add a positive wall_local term to total: total {} -> {}, wall_local={}",
            b0.total, b1.total, b1.wall_local
        );
    }

    /// #7: a boundary with a DUPLICATED vertex has a zero-length segment whose
    /// orientation `atan2(0,0)` is a meaningless 0. A query point nearest that
    /// corner must still get the real adjacent edges' local frame, not φ = 0.
    #[test]
    fn degenerate_boundary_segment_does_not_corrupt_local_angle() {
        let clean = diamond_boundary();
        let dup = Polygon::new(
            LineString::from(vec![(1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]),
            vec![],
        );
        let (qx, qy) = (1.1, 0.0); // projects onto the duplicated corner (1,0)
        let a_clean = nearest_boundary_angle(qx, qy, &clean);
        let a_dup = nearest_boundary_angle(qx, qy, &dup);
        assert_eq!(
            a_dup, a_clean,
            "a zero-length boundary segment must not change the nearest-boundary angle (clean {a_clean}, dup {a_dup})"
        );
    }

    /// KR2 (#1): the Blend mode's per-edge penalty must be CONTINUOUS across a
    /// boundary-orientation change (a nearest-segment pick jumps ~the full range).
    #[test]
    fn blend_edge_penalty_is_continuous_across_a_corner() {
        let b = Polygon::new(
            LineString::from(vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (1.0, 4.0), (0.0, 3.0)]),
            vec![],
        );
        let (dx, dy) = (0.2_f32, 0.2_f32); // a fixed 45° edge vector
        let n = 120;
        let mut prev: Option<f64> = None;
        let (mut lo, mut hi, mut max_jump) = (f64::INFINITY, f64::NEG_INFINITY, 0.0_f64);
        for i in 0..=n {
            let x = 0.55 + (2.5 - 0.55) * (i as f64) / (n as f64);
            let p = edge_alignment_penalty(x, 3.4, dx, dy, &b) as f64;
            lo = lo.min(p);
            hi = hi.max(p);
            if let Some(pp) = prev {
                max_jump = max_jump.max((p - pp).abs());
            }
            prev = Some(p);
        }
        let range = hi - lo;
        assert!(range > 0.1, "sweep must cross a real orientation change (range {range} too small)");
        assert!(
            max_jump < 0.25 * range,
            "Blend penalty must be continuous; max step {max_jump} ≥ 0.25×range {range} (a jump)"
        );
    }
}
