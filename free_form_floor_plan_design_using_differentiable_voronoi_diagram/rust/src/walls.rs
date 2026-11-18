//! Wall-band geometry for the B&W floor-plan visualization (demo only).
//!
//! Each room's boundary ring is treated as a wall *centerline*; buffering it by
//! `d` to both sides yields a band of width `2d`, and unioning every room's
//! bands welds shared walls (where two rooms meet) into one. Pure geometry,
//! wasm-safe; reuses geo's i_overlay `unary_union` for the weld (the same
//! engine `grad_local::room_union` uses). geo 0.30 has no buffer, so the band
//! is built from per-segment rectangles plus per-vertex disks.

use geo::{Coord, Line, LineString, MultiPolygon, Polygon, Simplify};

pub(crate) fn boundary_segments(room:&MultiPolygon<f64>)->Vec<Line<f64>>{
    use std::collections::HashMap;
    let mut pending=HashMap::new();
    let mut edges:Vec<Option<Line<f64>>>=Vec::new();
    for ring in room.iter().flat_map(|p|std::iter::once(p.exterior()).chain(p.interiors())){
        for line in ring.simplify(&f64::MIN_POSITIVE).lines(){
            let a=(line.start.x.to_bits(),line.start.y.to_bits());
            let b=(line.end.x.to_bits(),line.end.y.to_bits());
            let key=if a<b {(a,b)}else{(b,a)};
            // Snapped twin edges within one room are cell seams. Keep source
            // order for surviving edges so gradients remain deterministic.
            if let Some(index)=pending.remove(&key){edges[index]=None;}
            else {pending.insert(key,edges.len());edges.push(Some(line));}
        }
    }
    edges.into_iter().flatten().collect()
}

/// Actual positive-length overlap, shared by door widths and wall shape losses.
pub(crate) fn shared_wall_segments(a:&[Line<f64>],b:&[Line<f64>])->Vec<Line<f64>>{
    use geo::line_intersection::{line_intersection,LineIntersection};
    let mut shared=Vec::new();
    for &x in a {for &y in b {
        if let Some(LineIntersection::Collinear{intersection})=line_intersection(x,y){
            if intersection.start!=intersection.end {shared.push(intersection);}
        }
    }}
    shared
}

pub(crate) fn wall_chains(segments:&[Line<f64>],tol:f64)->Vec<Vec<[f64;2]>>{
    fn vertex(nodes:&mut Vec<[f64;2]>,p:[f64;2],tol:f64)->usize{
        if let Some(i)=nodes.iter().position(|q|(p[0]-q[0]).abs()<=tol&&(p[1]-q[1]).abs()<=tol){return i;}
        nodes.push(p);nodes.len()-1
    }
    let mut nodes=Vec::new();
    let mut edges=Vec::new();
    for segment in segments{
        let a=vertex(&mut nodes,[segment.start.x,segment.start.y],tol);
        let b=vertex(&mut nodes,[segment.end.x,segment.end.y],tol);
        let edge=if a<b {(a,b)}else{(b,a)};
        if a!=b&&!edges.contains(&edge){edges.push(edge);}
    }
    let mut adjacent=vec![Vec::new();nodes.len()];
    for (i,&(a,b)) in edges.iter().enumerate(){adjacent[a].push(i);adjacent[b].push(i);}
    let mut starts:Vec<_>=adjacent.iter().enumerate().filter(|(_,a)|a.len()!=2).flat_map(|(v,a)|a.iter().map(move|&e|(v,e))).collect();
    starts.extend(edges.iter().enumerate().map(|(e,&(a,_))|(a,e)));
    let mut visited=vec![false;edges.len()];
    let mut chains=Vec::new();
    for (mut at,mut edge) in starts{
        if visited[edge]{continue;}
        let mut points=vec![nodes[at]];
        loop{
            visited[edge]=true;
            let (a,b)=edges[edge];let next=if a==at {b}else{a};
            points.push(nodes[next]);
            if adjacent[next].len()!=2{break;}
            let Some(&following)=adjacent[next].iter().find(|&&e|!visited[e])else{break;};
            at=next;edge=following;
        }
        chains.push(points);
    }
    chains
}

fn collinear(a:[f64;2],b:[f64;2],c:[f64;2],tol:f64)->bool{
    let ab=[b[0]-a[0],b[1]-a[1]];
    let bc=[c[0]-b[0],c[1]-b[1]];
    let ac=[c[0]-a[0],c[1]-a[1]];
    let length=ac[0].hypot(ac[1]);
    length>0. && ab[0]*bc[0]+ab[1]*bc[1]>=0. && (ab[0]*ac[1]-ab[1]*ac[0]).abs()<=tol*length
}

// Canonical physical segments make the nearest-frame sample independent of
// arbitrary collinear Voronoi subdivisions. This only removes roundoff bends.
pub(crate) fn canonical_chain(raw:&[[f64;2]],tol:f64)->Vec<[f64;2]>{
    let mut points:Vec<[f64;2]>=Vec::new();
    for &p in raw{
        if points.last().is_some_and(|q|(p[0]-q[0]).hypot(p[1]-q[1])<=tol){continue;}
        points.push(p);
        while points.len()>=3{
            let n=points.len();
            if !collinear(points[n-3],points[n-2],points[n-1],tol){break;}
            points.remove(n-2);
        }
    }
    if points.len()>2 && (points[0][0]-points.last().unwrap()[0]).hypot(points[0][1]-points.last().unwrap()[1])<=tol{
        points.pop();
        while points.len()>2 && collinear(*points.last().unwrap(),points[0],points[1],tol){points.remove(0);}
        while points.len()>2 && collinear(points[points.len()-2],*points.last().unwrap(),points[0],tol){points.pop();}
        points.push(points[0]);
    }
    points
}

/// Unit direction of the nearest exterior segment. Preserve the original
/// f32 projection arithmetic for the native numerical-reference path.
pub(crate) fn nearest_boundary_direction(mx: f64, my: f64, boundary: &Polygon<f64>) -> (f32, f32) {
    let mut best_d2 = f64::INFINITY;
    let (mut cx, mut cy) = (1.0f32, 0.0f32);
    for line in boundary.exterior().lines() {
        let (ax, ay) = (line.start.x, line.start.y);
        let (ex, ey) = (line.end.x - ax, line.end.y - ay);
        let len2 = ex * ex + ey * ey;
        if len2 == 0.0 { continue; }
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
    #[test]
    fn cell_seams_do_not_become_room_boundary_walls(){
        let square=|x:f64|Polygon::new(LineString::from(vec![(x,0.),(x+1.,0.),(x+1.,1.),(x,1.),(x,0.)]),vec![]);
        let room=MultiPolygon(vec![square(0.),square(1.)]);
        let edges=boundary_segments(&room);
        assert!(edges.iter().all(|e|e.start.x!=1. || e.end.x!=1.),"the common cell seam lies inside one room and is not a wall");
        let perimeter:f64=edges.iter().map(|e|(e.end.x-e.start.x).hypot(e.end.y-e.start.y)).sum();
        assert_eq!(perimeter,6.);
        assert_eq!(boundary_segments(&room),edges,"preserve deterministic source order for finite differences");
    }

}
