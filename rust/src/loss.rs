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

pub fn compute_wall_loss(rooms_group: &[Vec<&Polygon<f64>>], w_wall: f64) -> f32 {
    // Python accumulates the per-ring f32 sums into a Python float (f64) ...
    let mut loss_wall: f64 = 0.0;
    for group in rooms_group {
        let room_union = union_group(group);
        for room in &room_union {
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
