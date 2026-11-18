//! Direction and straightness of physical shared-wall chains.
use geo::{Area,Line,Polygon,MultiPolygon};

pub(crate) fn evaluate(rooms:&[MultiPolygon<f64>],boundary:&Polygon<f64>)->(f64,f64){
    if rooms.len()<2 {return (0.,0.);}
    let edges:Vec<_>=rooms.iter().map(crate::walls::boundary_segments).collect();
    let mut chains=Vec::new();
    let tol=boundary.unsigned_area().sqrt()*1e-12;
    for r in 0..rooms.len(){for s in r+1..rooms.len(){
        let shared=crate::walls::shared_wall_segments(&edges[r],&edges[s]);
        chains.extend(wall_chains(&shared,tol));
    }}
    let (direction,jog)=losses(&chains,boundary);
    // Room count is fixed during an optimization. Adding a perfect chain must
    // not lower another wall's error by changing the averaging denominator.
    (direction/rooms.len() as f64,jog/rooms.len() as f64)
}

fn wall_chains(segments:&[Line<f64>],tol:f64)->Vec<Vec<[f64;2]>>{
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
fn canonical_chain(raw:&[[f64;2]],tol:f64)->Vec<[f64;2]>{
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

fn losses(chains: &[Vec<[f64;2]>], boundary: &Polygon<f64>) -> (f64,f64) {
    let tol=boundary.unsigned_area().sqrt()*1e-12;
    let (mut direction,mut jog)=(0.,0.);
    for raw in chains{
        let points=canonical_chain(raw,tol);
        let (mut weighted,mut length)=(0.,0.);
        for pair in points.windows(2){
            let (a,b)=(pair[0],pair[1]);
            let (dx,dy)=(b[0]-a[0],b[1]-a[1]);
            let l=dx.hypot(dy);if l==0. {continue;}
            let (c,s)=crate::walls::nearest_boundary_direction((a[0]+b[0])/2.,(a[1]+b[1])/2.,boundary);
            let (du,dv)=(dx*c as f64+dy*s as f64,-dx*s as f64+dy*c as f64);
            let q=du*du/(du*du+dv*dv);
            weighted+=l*4.*q*(1.-q);length+=l;
        }
        if length!=0. {direction+=weighted/length;}
        jog+=chain_jog(&points,boundary);
    }
    (direction,jog)
}

fn chain_jog(points:&[[f64;2]],boundary:&Polygon<f64>)->f64{
    let (mut x,mut y,mut total)=(0.,0.,0.);
    for pair in points.windows(2){
        let (a,b)=(pair[0],pair[1]);let length=(b[0]-a[0]).hypot(b[1]-a[1]);
        x+=length*(a[0]+b[0])/2.;y+=length*(a[1]+b[1])/2.;total+=length;
    }
    if total==0. {return 0.;}
    let (x,y)=(x/total,y/total);
    let (c,s)=crate::walls::nearest_boundary_direction(x,y,boundary);
    let norm=(c as f64).hypot(s as f64);let (c,s)=(c as f64/norm,s as f64/norm);
    let (u0,v0)=(x*c+y*s,-x*s+y*c);
    let mut groups=[[0.;3];4];
    for pair in points.windows(2){
        let (a,b)=(pair[0],pair[1]);let (dx,dy)=(b[0]-a[0],b[1]-a[1]);
        let length=dx.hypot(dy);if length==0. {continue;}
        let (du,dv)=(dx*c+dy*s,-dx*s+dy*c);
        let (mx,my)=((a[0]+b[0])/2.,(a[1]+b[1])/2.);
        let (u,v)=(mx*c+my*s,-mx*s+my*c);
        let (group,offset)=if du.abs()>=dv.abs(){
            (if du>0. {0}else{1},(v-v0)-(dv/du)*(u-u0))
        }else{
            (if dv>0. {2}else{3},(u-u0)-(du/dv)*(v-v0))
        };
        groups[group][0]+=length;groups[group][1]+=length*offset;groups[group][2]+=length*offset*offset;
    }
    // Two percent of the boundary's area-based linear scale represents a small
    // offset. The monotone response cannot reward enlarging a short connector.
    let scale2=0.02_f64.powi(2)*boundary.unsigned_area();
    groups.iter().filter(|g|g[0]>0.).map(|g|{
        let variance=(g[2]/g[0]-(g[1]/g[0]).powi(2)).max(0.);
        variance/(variance+scale2)
    }).sum::<f64>()/4.
}

#[cfg(test)]
#[path = "wall_shape_tests.rs"]
mod tests;
