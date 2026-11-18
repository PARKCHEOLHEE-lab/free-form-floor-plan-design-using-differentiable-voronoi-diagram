//! Wall-band geometry for the B&W floor-plan visualization (demo only).
//!
//! Each room's boundary ring is treated as a wall *centerline*; buffering it by
//! `d` to both sides yields a band of width `2d`, and unioning every room's
//! bands welds shared walls (where two rooms meet) into one. Pure geometry,
//! wasm-safe; reuses geo's i_overlay `unary_union` for the weld (the same
//! engine `grad_local::room_union` uses). geo 0.30 has no buffer, so the band
//! is built from per-segment rectangles plus per-vertex disks.

use geo::{Coord, Line, LineString, MultiPolygon, Polygon, Simplify};

type CellEdge=((u64,u64),(u64,u64));
const NO_CELL:usize=usize::MAX;

/// Shared physical walls, in stable room/edge order for numerical evaluation.
pub(crate) struct SharedWalls {
    rooms:usize,
    pairs:std::collections::BTreeMap<(usize,usize),Vec<Line<f64>>>,
}
impl SharedWalls {
    pub(crate) fn from_cells(groups:&[Vec<&Polygon<f64>>])->Self {
        SharedWallCells::from_iter(groups.len(),groups.iter().enumerate().flat_map(|(room,polys)|polys.iter().map(move|p|(*p,room)))).snapshot()
    }
    pub(crate) fn between(&self,a:usize,b:usize)->&[Line<f64>]{
        let pair=if a<b {(a,b)}else{(b,a)};
        self.pairs.get(&pair).map(Vec::as_slice).unwrap_or(&[])
    }
    pub(crate) fn pairs(&self)->impl Iterator<Item=&[Line<f64>]> {self.pairs.values().map(Vec::as_slice)}
    pub(crate) fn room_count(&self)->usize{self.rooms}
}

/// Immutable base-cell ownership. Each perturbation changes only its cell faces.
pub(crate) struct SharedWallCells {
    rooms:usize,
    cell_rooms:Vec<usize>,
    cell_edges:Vec<Vec<CellEdge>>,
    owners:std::collections::HashMap<CellEdge,[usize;2]>,
}
impl SharedWallCells {
    pub(crate) fn new(cells:&[Polygon<f64>],labels:&[usize])->Self{
        Self::from_iter(labels.iter().copied().max().map_or(0,|n|n+1),cells.iter().zip(labels.iter().copied()))
    }
    fn from_iter<'a>(rooms:usize,cells:impl Iterator<Item=(&'a Polygon<f64>,usize)>)->Self{
        let mut out=Self{rooms,cell_rooms:Vec::new(),cell_edges:Vec::new(),owners:std::collections::HashMap::new()};
        for (cell,room) in cells {
            let id=out.cell_rooms.len();let edges=Self::edges(cell);
            for &edge in &edges {Self::insert(&mut out.owners,edge,id);}
            out.cell_rooms.push(room);out.cell_edges.push(edges);
        }
        out
    }
    fn edges(cell:&Polygon<f64>)->Vec<CellEdge>{
        let mut edges=Vec::new();
        // Cell ownership precedes collinear simplification.
        for ring in std::iter::once(cell.exterior()).chain(cell.interiors()){for line in ring.lines(){
            if line.start==line.end {continue;}
            let a=(line.start.x.to_bits(),line.start.y.to_bits());let b=(line.end.x.to_bits(),line.end.y.to_bits());
            edges.push(if a<b{(a,b)}else{(b,a)});
        }}
        edges.sort_unstable();edges.dedup();edges
    }
    fn insert(map:&mut std::collections::HashMap<CellEdge,[usize;2]>,edge:CellEdge,cell:usize){
        let owners=map.entry(edge).or_insert([NO_CELL;2]);
        if owners.contains(&cell){return;}
        let slot=owners.iter_mut().find(|n|**n==NO_CELL).expect("a Voronoi cell face has at most two owners");*slot=cell;
    }
    fn collect(&self,map:&std::collections::HashMap<CellEdge,[usize;2]>)->SharedWalls{
        let mut pairs:std::collections::BTreeMap<_,Vec<Line<f64>>>=std::collections::BTreeMap::new();
        for (&(a,b),owners) in map {
            if owners[0]==NO_CELL||owners[1]==NO_CELL {continue;}
            let (r,t)=(self.cell_rooms[owners[0]],self.cell_rooms[owners[1]]);
            if r==t{continue;}
            let pair=if r<t{(r,t)}else{(t,r)};
            pairs.entry(pair).or_default().push(Line{start:Coord{x:f64::from_bits(a.0),y:f64::from_bits(a.1)},end:Coord{x:f64::from_bits(b.0),y:f64::from_bits(b.1)}});
        }
        for edges in pairs.values_mut(){edges.sort_unstable_by_key(|e|((e.start.x.to_bits(),e.start.y.to_bits()),(e.end.x.to_bits(),e.end.y.to_bits())));}
        SharedWalls{rooms:self.rooms,pairs}
    }
    pub(crate) fn snapshot(&self)->SharedWalls{self.collect(&self.owners)}
    pub(crate) fn updated(&self,cells:&[Polygon<f64>],changed:&[usize])->SharedWalls{
        let mut owners=self.owners.clone();
        for &cell in changed {for edge in &self.cell_edges[cell]{
            let slots=owners.get_mut(edge).expect("base cell ownership exists");
            *slots.iter_mut().find(|n|**n==cell).expect("base face contains its cell")=NO_CELL;
            if slots.iter().all(|n|*n==NO_CELL){owners.remove(edge);}
        }}
        for &cell in changed {for edge in Self::edges(&cells[cell]){Self::insert(&mut owners,edge,cell);}}
        self.collect(&owners)
    }
}

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

    #[test]
    fn shared_cell_faces_match_generic_geometry_including_local_perturbations(){
        use crate::{init,shapes,voronoi,loss::rooms_group};
        let compare=|cells:&[Polygon<f64>],labels:&[usize]|{
            let groups=rooms_group(cells,labels);
            let fast=SharedWalls::from_cells(&groups);
            let indexed=SharedWallCells::new(cells,labels).snapshot();
            assert_eq!(fast.room_count(),groups.len());
            let rooms:Vec<_>=groups.iter().map(|g|MultiPolygon(g.iter().map(|p|(*p).clone()).collect())).collect();
            let edges:Vec<_>=rooms.iter().map(boundary_segments).collect();
            let length=|lines:&[Line<f64>]|lines.iter().map(|l|(l.end.x-l.start.x).hypot(l.end.y-l.start.y)).sum::<f64>();
            for r in 0..groups.len(){for t in r+1..groups.len(){
                let expected=shared_wall_segments(&edges[r],&edges[t]);let actual=fast.between(r,t);
                assert_eq!(indexed.between(r,t),actual,"stable owner ordering must not depend on cell labels");
                assert!((length(&expected)-length(actual)).abs()<1e-8,"shared physical length must match for rooms{r},{t}");
                for edge in actual {assert!((length(&shared_wall_segments(&[*edge],&expected))-length(&[*edge])).abs()<1e-8);}
                for edge in &expected {assert!((length(&shared_wall_segments(&[*edge],actual))-length(&[*edge])).abs()<1e-8);}
            }}
        };
        for name in ["shape_a","shape_b","shape_c","shape_d","shape_e","shape_duck"]{
            let b=shapes::by_name(name).unwrap().polygon();
            for seed in [0,777,4971]{let s=init::initialize_sites(&b,80,seed);let labels=init::kmeans_labels(&s,6,seed);compare(&voronoi::compute_cells(&s,&b,None).cells_sorted,&labels);}
        }
        let data:serde_json::Value=serde_json::from_str(include_str!("../../web-tests/fixtures/room-program-six.json")).unwrap();
        let circles:Vec<[f64;3]>=data["rooms"].as_array().unwrap().iter().map(|r|[r["center"][0].as_f64().unwrap(),r["center"][1].as_f64().unwrap(),r["radius"].as_f64().unwrap()]).collect();
        let b=shapes::by_name("shape_e").unwrap().polygon();let p=crate::program::RoomProgram::new(&b,&circles).unwrap();let (s,labels)=p.initialize(&b,80,4971).unwrap();
        let base=voronoi::compute_cells(&s,&b,None).cells_sorted;let targets=vec![0.2;6];let w=crate::loss::LossWeights::default();let ctx=crate::grad_local::LocalGradContext::new(&s,&b,&targets,&labels,&w);
        for i in 0..s.len(){for axis in 0..2{for sign in [-1.,1.]{let mut moved=s.clone();moved[i][axis]+=sign*1e-6;compare(&voronoi::compute_cells(&moved,&b,None).cells_sorted,&labels);let mut patched=base.clone();for(j,c)in ctx.affected_cells(&moved,i){patched[j]=c;}voronoi::snap_cells(&mut patched);compare(&patched,&labels);}}}
        // A known multipart-cell clipping case, with each site owning a room.
        let b=shapes::by_name("shape_b").unwrap().polygon();let sites=init::initialize_sites(&b,16,5);let labels:Vec<_>=(0..16).collect();compare(&voronoi::compute_cells(&sites,&b,None).cells_sorted,&labels);
        // Compare the actual engine's clipped-cell representation for a hole
        // and extra collinear boundary vertices, without claiming hole support.
        let b=Polygon::new(LineString::from(vec![(-1.,-1.),(0.,-1.),(1.,-1.),(1.,1.),(-1.,1.),(-1.,-1.)]),vec![LineString::from(vec![(-0.2,-0.2),(-0.2,0.2),(0.2,0.2),(0.2,-0.2),(-0.2,-0.2)])]);
        let sites=init::initialize_sites(&b,40,777);let labels=init::kmeans_labels(&sites,4,777);compare(&voronoi::compute_cells(&sites,&b,None).cells_sorted,&labels);
        let b=Polygon::new(LineString::from(vec![(-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.),(-1.,-1.)]),vec![]);
        let s:Vec<_>=[-0.75_f32,-0.25,0.25,0.75].into_iter().flat_map(|y|[-0.12,0.,0.12].into_iter().map(move|x|[x,y])).collect();
        let labels:Vec<_>=s.iter().map(|p|if p[0]<0. ||(p[0]==0. && p[1].abs()<0.5){0}else{1}).collect();
        compare(&voronoi::compute_cells(&s,&b,None).cells_sorted,&labels);
    }

}
