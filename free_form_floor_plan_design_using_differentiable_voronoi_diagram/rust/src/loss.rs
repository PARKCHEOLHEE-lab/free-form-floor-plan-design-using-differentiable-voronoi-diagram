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

/// The wall loss: every room-boundary edge is measured as a *rotated* taxicab
/// length in the frame of the nearest domain-boundary segment (its orientation
/// φ). An edge parallel or perpendicular to the local boundary costs its
/// Euclidean length; a 45°-skewed edge costs √2×. On an axis-aligned boundary
/// every φ ≡ 0 (mod 90°), so this reduces bit-exactly to the plain taxicab wall
/// length (which is why the axis-aligned goldens are unchanged by the switch).
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

/// Unit direction `(cx, cy)` (= cos φ, sin φ) of the boundary segment nearest to
/// `(mx, my)`, as f32. Computing the unit vector by division — not atan2/cos/sin —
/// keeps the rotated-L1 below portable f32-exact across platforms (libm trig is
/// not IEEE-mandated; +,−,×,÷,√ are). At an axis-aligned segment (cx,cy) is exactly
/// (±1,0)/(0,±1), so the local sum reduces bit-identically to the global wall sum.
fn nearest_boundary_direction(mx: f64, my: f64, boundary: &Polygon<f64>) -> (f32, f32) {
    let mut best_d2 = f64::INFINITY;
    let (mut cx, mut cy) = (1.0f32, 0.0f32); // φ = 0 default (no usable segment)
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
            let l = len2.sqrt();
            cx = (ex / l) as f32;
            cy = (ey / l) as f32;
        }
    }
    (cx, cy)
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
        let (c, sn) = nearest_boundary_direction(mx, my, boundary);
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

#[derive(Default)]
pub struct LossWeights {
    pub w_wall: f64,
    pub w_area: f64,
    pub w_lloyd: f64,
    pub w_topo: f64,
    pub w_bb: f64,
    pub w_cell: f64,
}

#[derive(Clone, Copy)]
pub struct LossBreakdown {
    pub total: f32,
    pub wall: f32,
    pub area: f32,
    pub lloyd: f32,
    pub topo: f32,
    pub bb: f32,
    pub cell: f32,
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
    let unions: Vec<MultiPolygon<f64>> = if w.w_wall > 0.0 || w.w_topo > 0.0 || w.w_bb > 0.0 {
        groups.iter().map(|g| union_group(g)).collect()
    } else {
        Vec::new()
    };

    // The wall term is the nearest-boundary alignment loss (rotated-L1). On an
    // axis-aligned boundary it reduces bit-exactly to the plain taxicab length.
    let wall = if w.w_wall > 0.0 {
        compute_wall_local_loss(&unions, boundary, w.w_wall)
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
    // loss = loss_wall + loss_area + loss_lloyd + loss_topo + loss_bb + loss_cell
    let total = wall + area + lloyd + topo + bb + cell;
    LossBreakdown {
        total,
        wall,
        area,
        lloyd,
        topo,
        bb,
        cell,
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
        // taxicab reference (the pre-merge global wall loss), computed inline
        // since that path was removed — proves the alignment loss reduces to it.
        let mut tx = 0.0f64;
        for poly in &room {
            let pts = &poly.exterior().0;
            let n = pts.len().saturating_sub(1);
            let t: Vec<[f32; 2]> = pts[..n].iter().map(|c| [c.x as f32, c.y as f32]).collect();
            let mut s = 0.0f32;
            for i in 0..n {
                let j = (i + 1) % n;
                s += (t[i][0] - t[j][0]).abs();
                s += (t[i][1] - t[j][1]).abs();
            }
            tx += s as f64;
        }
        let taxicab = (tx as f32) * 2.5f32;
        let local = compute_wall_local_loss(&[room], &b, 2.5);
        assert_eq!(
            local, taxicab,
            "the alignment wall loss must reduce to the plain taxicab length on an axis-aligned boundary"
        );
    }

    fn duck_geom() -> (Vec<[f32; 2]>, Polygon<f64>, Vec<f64>, Vec<usize>) {
        // duck has 40 diagonal boundary edges, so the rotated-L1 alignment loss
        // differs from the global taxicab loss — the shape that distinguishes them.
        let boundary = crate::shapes::by_name("shape_duck").unwrap().polygon();
        let barea = boundary.unsigned_area();
        let ratios = [0.2, 0.2, 0.2, 0.2, 0.2];
        let target_areas: Vec<f64> = ratios.iter().map(|r| barea * r).collect();
        let sites = crate::init::initialize_sites(&boundary, 40, 777);
        let room_indices = crate::init::kmeans_labels(&sites, ratios.len(), 777);
        (sites, boundary, target_areas, room_indices)
    }

    #[test]
    fn wall_term_computes_nearest_boundary_alignment_on_diagonal_shape() {
        // KR3: the single `wall` term (weighted by w_wall) must be the
        // nearest-boundary ALIGNMENT loss, not the global taxicab loss. On the
        // duck (diagonal edges) the two differ, so this distinguishes them.
        let (sites, boundary, ta, ri) = duck_geom();
        let w = LossWeights { w_wall: 2.5, ..Default::default() };
        let b = floor_plan_loss(&sites, &boundary, &ta, &ri, &w, None);
        // reference alignment from the same unions floor_plan_loss builds internally
        let geom = crate::voronoi::compute_cells(&sites, &boundary, None);
        let groups = rooms_group(&geom.cells_sorted, &ri);
        let unions: Vec<MultiPolygon<f64>> = groups.iter().map(|g| union_group(g)).collect();
        let alignment = compute_wall_local_loss(&unions, &boundary, 2.5);
        assert_eq!(
            b.wall, alignment,
            "the wall term must be the nearest-boundary ALIGNMENT loss, not the global taxicab loss"
        );
    }

    /// #7: a boundary with a DUPLICATED vertex has a zero-length segment with no
    /// direction. A query point nearest that corner must still get the real
    /// adjacent edges' local frame, not a corrupted (cx,cy).
    #[test]
    fn degenerate_boundary_segment_does_not_corrupt_local_angle() {
        let clean = diamond_boundary();
        let dup = Polygon::new(
            LineString::from(vec![(1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]),
            vec![],
        );
        let (qx, qy) = (1.1, 0.0); // projects onto the duplicated corner (1,0)
        let a_clean = nearest_boundary_direction(qx, qy, &clean);
        let a_dup = nearest_boundary_direction(qx, qy, &dup);
        assert_eq!(
            a_dup, a_clean,
            "a zero-length boundary segment must not change the nearest-boundary direction (clean {a_clean:?}, dup {a_dup:?})"
        );
    }

}
