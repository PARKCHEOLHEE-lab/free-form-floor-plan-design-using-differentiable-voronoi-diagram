//! Direction and straightness of physical shared-wall chains.
use geo::{Area,Polygon};
use crate::walls::{wall_chains,canonical_chain};

pub(crate) fn evaluate_shared(shared:&crate::walls::SharedWalls,boundary:&Polygon<f64>)->(f64,f64){
    if shared.room_count()<2{return (0.,0.);}
    let tol=boundary.unsigned_area().sqrt()*1e-12;
    let chains:Vec<_>=shared.pairs().flat_map(|edges|wall_chains(edges,tol)).collect();
    let (direction,jog)=losses(&chains,boundary);
    (direction/shared.room_count() as f64,jog/shared.room_count() as f64)
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
    let scale2=0.02_f64.powi(2)*boundary.unsigned_area();
    let softness=scale2.sqrt();
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
        // A physical run has bounded influence once it is visible. Weighting
        // directly by its full length lets a growing step hide in a short run.
        let extent=du.abs().max(dv.abs());
        let weight=extent/(extent+softness);
        groups[group][0]+=weight;
        groups[group][1]+=weight*offset;
        groups[group][2]+=weight*offset*offset;
        groups[group][3]+=1.;
    }
    groups.iter().filter(|g|g[3]>1.).map(|g|{
        // Sum of weighted pairwise squared line offsets. There is no changing
        // length/count denominator that can dilute a remaining step.
        let energy=(g[0]*g[2]-g[1]*g[1]).max(0.);
        let z=energy/(4.*scale2);
        // Smooth soft-L1 keeps a useful force on large offsets. This rational
        // form is stable when z is tiny, unlike sqrt(1+z)-1.
        2.*z/((1.+z).sqrt()+1.)
    }).sum::<f64>()/4.

}

// The former geometric search remains an independent test oracle.
#[cfg(test)]
pub(crate) fn evaluate(rooms:&[geo::MultiPolygon<f64>],boundary:&Polygon<f64>)->(f64,f64){
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

#[cfg(test)]
#[path = "wall_shape_tests.rs"]
mod tests;
