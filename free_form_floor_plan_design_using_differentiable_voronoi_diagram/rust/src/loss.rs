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

pub struct LossWeights {
    pub w_wall: f64,
    pub w_area: f64,
    pub w_lloyd: f64,
    pub w_topo: f64,
    pub w_bb: f64,
    pub w_cell: f64,
}

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
