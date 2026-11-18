//! Direction and straightness of physical shared-wall chains.
use geo::{Area,Polygon,MultiPolygon};
use crate::walls::{wall_chains,canonical_chain};

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
    let mut groups=[[0.;4];4];
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
        groups[group][3]+=du.abs().max(dv.abs());
    }
    // Two percent of the boundary's area-based linear scale represents a small
    // offset. The monotone response cannot reward enlarging a short connector.
    let scale2=0.02_f64.powi(2)*boundary.unsigned_area();
    groups.iter().filter(|g|g[0]>0.).map(|g|{
        let variance=(g[2]/g[0]-(g[1]/g[0]).powi(2)).max(0.);
        // Vanishing transverse runs must vanish from the objective too. Use
        // projected extent so shearing parallel runs preserves their penalty.
        let activity=g[3]/(g[3]+scale2.sqrt());
        variance/(variance+scale2)*activity
    }).sum::<f64>()/4.
}

#[cfg(test)]
#[path = "wall_shape_tests.rs"]
mod tests;
