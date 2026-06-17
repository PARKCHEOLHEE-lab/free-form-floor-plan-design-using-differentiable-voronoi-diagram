//! Wall-band geometry for the B&W floor-plan visualization (demo only).
//!
//! Each room's boundary ring is treated as a wall *centerline*; buffering it by
//! `d` to both sides yields a band of width `2d`, and unioning every room's
//! bands welds shared walls (where two rooms meet) into one. Pure geometry,
//! wasm-safe; reuses geo's i_overlay `unary_union` for the weld (the same
//! engine `grad_local::room_union` uses). geo 0.30 has no buffer, so the band
//! is built from per-segment rectangles plus per-vertex disks.

use geo::{Coord, LineString, MultiPolygon, Polygon};

/// Vertices per buffer disk (round joins/caps). 16 keeps the joins smooth enough
/// that sharp outer corners fill cleanly, without bloating the union input.
const DISK_SEGS: usize = 16;

/// Buffer every room-boundary ring by `d` — a band of width `2d` centered on
/// the ring — and union all bands into the wall network. Shared walls dissolve
/// because both rooms emit the same strip and the union welds them.
pub fn room_walls(room_unions: &[MultiPolygon<f64>], d: f64) -> MultiPolygon<f64> {
    if d <= 0.0 {
        return MultiPolygon::new(vec![]);
    }
    let mut parts: Vec<Polygon<f64>> = Vec::new();
    for mp in room_unions {
        for poly in mp {
            for ring in std::iter::once(poly.exterior()).chain(poly.interiors().iter()) {
                let pts = &ring.0;
                // one rectangle per segment (the straight wall run)...
                for w in pts.windows(2) {
                    if w[0] != w[1] {
                        parts.push(segment_quad(w[0], w[1], d));
                    }
                }
                // ...and one disk per vertex to fill the joins (skip the closing dup)
                let m = pts.len().saturating_sub(1);
                for &c in &pts[..m] {
                    parts.push(disk(c, d, DISK_SEGS));
                }
            }
        }
    }
    geo::algorithm::unary_union(parts.iter())
}

/// Rectangle of width `2d` centered on segment `a`-`b` (the straight wall run).
fn segment_quad(a: Coord<f64>, b: Coord<f64>, d: f64) -> Polygon<f64> {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let len = (dx * dx + dy * dy).sqrt();
    // perpendicular unit normal scaled by d (a != b is guaranteed by the caller)
    let (nx, ny) = (-dy / len * d, dx / len * d);
    Polygon::new(
        LineString::from(vec![
            (a.x + nx, a.y + ny),
            (b.x + nx, b.y + ny),
            (b.x - nx, b.y - ny),
            (a.x - nx, a.y - ny),
            (a.x + nx, a.y + ny),
        ]),
        vec![],
    )
}

/// Regular `n`-gon of radius `d` centered at `c` — fills a ring vertex so the
/// wall keeps constant width through the join (round join).
fn disk(c: Coord<f64>, d: f64, n: usize) -> Polygon<f64> {
    use std::f64::consts::TAU;
    let ring: Vec<(f64, f64)> = (0..=n)
        .map(|k| {
            let t = TAU * (k as f64) / (n as f64);
            (c.x + d * t.cos(), c.y + d * t.sin())
        })
        .collect();
    Polygon::new(LineString::from(ring), vec![])
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{Area, Contains, Point};

    fn unit_square_room() -> Vec<MultiPolygon<f64>> {
        let sq = Polygon::new(
            LineString::from(vec![(0., 0.), (1., 0.), (1., 1.), (0., 1.), (0., 0.)]),
            vec![],
        );
        vec![MultiPolygon::new(vec![sq])]
    }

    #[test]
    fn room_walls_is_band_centered_on_boundary() {
        // KR1: room_walls buffers the room boundary ring into a band of width 2d
        // centered on it — a point ON the boundary is inside the band, the room
        // CENTER is outside (a band, not a filled polygon), and the band grows
        // with d.
        let unions = unit_square_room();
        let d = 0.05;
        let walls = room_walls(&unions, d);

        // midpoint of the bottom edge sits on the centerline -> inside the band
        let on_boundary = Point::new(0.5, 0.0);
        assert!(walls.contains(&on_boundary), "boundary point not inside wall band");

        // the room center is 0.5 >> d from every edge -> OUTSIDE (band, not fill)
        let center = Point::new(0.5, 0.5);
        assert!(
            !walls.contains(&center),
            "room center inside walls — the band is filling the room, not bordering it"
        );

        // a point well outside the room is outside the band
        let outside = Point::new(0.5, -0.5);
        assert!(!walls.contains(&outside), "far exterior point inside wall band");

        // the band grows with d
        let area_thin = walls.unsigned_area();
        let area_thick = room_walls(&unions, 0.10).unsigned_area();
        assert!(
            area_thick > area_thin,
            "wall area did not grow with d ({area_thin} -> {area_thick})"
        );
    }
}
