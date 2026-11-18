//! Direction and straightness of physical shared-wall chains.
use geo::{Area,Polygon};

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
    let mut direction=0.;
    for raw in chains{
        let points=canonical_chain(raw,tol);
        let (mut weighted,mut length)=(0.,0.);
        for pair in points.windows(2){
            let (a,b)=(pair[0],pair[1]);
            let (dx,dy)=(b[0]-a[0],b[1]-a[1]);
            let l=dx.hypot(dy);if l==0. {continue;}
            let (c,s)=crate::loss::nearest_boundary_direction((a[0]+b[0])/2.,(a[1]+b[1])/2.,boundary);
            let (du,dv)=(dx*c as f64+dy*s as f64,-dx*s as f64+dy*c as f64);
            let q=du*du/(du*du+dv*dv);
            weighted+=l*4.*q*(1.-q);length+=l;
        }
        if length!=0. {direction+=weighted/length;}
    }
    (direction,0.)
}

#[cfg(test)]
#[path = "wall_shape_tests.rs"]
mod tests;
