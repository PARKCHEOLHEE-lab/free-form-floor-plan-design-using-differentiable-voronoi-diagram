//! Direction and straightness of physical shared-wall chains.
use crate::walls::{canonical_chain, stitch_room_chains, wall_chains, SharedWalls};
use geo::{Area, Line, Polygon};
use std::{borrow::Cow, collections::BTreeMap};

#[cfg(test)]
thread_local! {
    static SCORE_WORK: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
}

#[cfg(test)]
fn count_score_work(kind: usize) {
    SCORE_WORK.with(|work| {
        let mut counts = work.get();
        counts[kind] += 1;
        work.set(counts);
    });
}

#[derive(Clone)]
struct ChainScore {
    points: Vec<[f64; 2]>,
    direction: f64,
    jog: f64,
}

struct PairScores {
    edges: Vec<Line<f64>>,
    chains: Vec<ChainScore>,
}

/// Scores for the base diagram of a single gradient evaluation. A pair's
/// ordered edge bits identify its inputs; a room's stitched score depends
/// on all of its incident pairs. Keep individual terms in fold order.
pub(crate) struct WallShapeCache<'a> {
    boundary: &'a Polygon<f64>,
    direction: bool,
    jog: bool,
    pairs: BTreeMap<[usize; 2], PairScores>,
    junctions: Vec<Vec<f64>>,
}

impl<'a> WallShapeCache<'a> {
    pub(crate) fn new(shared: &SharedWalls, boundary: &'a Polygon<f64>, direction: bool, jog: bool) -> Self {
        let scale = boundary.unsigned_area().sqrt();
        let pairs: BTreeMap<_, _> = shared.indexed_pairs().map(|(rooms, edges)| {
            let chains = score_pair(edges, boundary, scale, direction, jog);
            (rooms, PairScores { edges: edges.to_vec(), chains })
        }).collect();
        let junctions = if jog {
            let mut by_room = vec![Vec::new(); shared.room_count()];
            for (rooms, pair) in &pairs {
                for &room in rooms {
                    by_room[room].extend(pair.chains.iter().map(|chain| chain.points.as_slice()));
                }
            }
            by_room.iter().map(|chains| score_junctions(chains, boundary, scale)).collect()
        } else {
            Vec::new()
        };
        Self { boundary, direction, jog, pairs, junctions }
    }

    pub(crate) fn evaluate(&self, shared: &SharedWalls) -> (f64, f64) {
        score_shared(shared, self.boundary, self.direction, self.jog, Some(self))
    }
}

fn same_edges(a: &[Line<f64>], b: &[Line<f64>]) -> bool {
    fn bits(line: &Line<f64>) -> [u64; 4] {
        [line.start.x.to_bits(), line.start.y.to_bits(), line.end.x.to_bits(), line.end.y.to_bits()]
    }
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| bits(a) == bits(b))
}

fn score_pair(edges: &[Line<f64>], boundary: &Polygon<f64>, scale: f64, direction: bool, jog: bool) -> Vec<ChainScore> {
    #[cfg(test)]
    count_score_work(0);
    let tol = scale * 1e-12;
    wall_chains(edges, tol).into_iter().map(|chain| {
        let points = canonical_chain(&chain, tol);
        let direction = if direction {
            #[cfg(test)]
            count_score_work(1);
            chain_direction(&points, boundary)
        } else {
            0.
        };
        let jog = if jog { chain_jog(&points, boundary) } else { 0. };
        ChainScore { points, direction, jog }
    }).collect()
}

fn score_junctions(chains: &[&[[f64; 2]]], boundary: &Polygon<f64>, scale: f64) -> Vec<f64> {
    if chains.len() <= 1 {
        return Vec::new();
    }
    let tol = scale * 1e-12;
    stitch_room_chains(chains, tol).into_iter().map(|(chain, joins)| {
        room_jog(&canonical_chain(&chain, tol), boundary, scale, &joins)
    }).collect()
}

#[cfg(test)]
pub(crate) fn evaluate_shared(shared: &SharedWalls, boundary: &Polygon<f64>) -> (f64, f64) {
    evaluate_shared_terms(shared, boundary, true, true)
}

pub(crate) fn evaluate_shared_terms(shared: &SharedWalls, boundary: &Polygon<f64>, direction: bool, jog: bool) -> (f64, f64) {
    score_shared(shared, boundary, direction, jog, None)
}

fn score_shared(shared: &SharedWalls, boundary: &Polygon<f64>, direction: bool, jog: bool, cache: Option<&WallShapeCache<'_>>) -> (f64, f64) {
    if shared.room_count() < 2 {
        return (0., 0.);
    }
    let scale = boundary.unsigned_area().sqrt();
    let mut dirty = vec![cache.is_none(); shared.room_count()];
    if let Some(base) = cache {
        // A pair that disappeared touches both rooms even though the new
        // graph has no entry for it.
        for &[a, b] in base.pairs.keys() {
            if shared.between(a, b).is_empty() {
                dirty[a] = true;
                dirty[b] = true;
            }
        }
    }
    let traced: Vec<_> = shared.indexed_pairs().map(|(rooms, edges)| {
        let saved = cache.and_then(|base| base.pairs.get(&rooms))
            .filter(|pair| same_edges(&pair.edges, edges));
        let chains = if let Some(pair) = saved {
            Cow::Borrowed(pair.chains.as_slice())
        } else {
            for r in rooms { dirty[r] = true; }
            Cow::Owned(score_pair(edges, boundary, scale, direction, jog))
        };
        (rooms, chains)
    }).collect();
    let (mut direction_score, mut pair_jog) = (0., 0.);
    for (_, chains) in &traced {
        for score in chains.iter() {
            direction_score += score.direction;
            pair_jog += score.jog;
        }
    }
    let mut junction_jog = 0.;
    if jog {
        let mut by_room = vec![Vec::new(); shared.room_count()];
        for (rooms, chains) in &traced {
            for &r in rooms {
                if dirty[r] {
                    by_room[r].extend(chains.iter().map(|chain| chain.points.as_slice()));
                }
            }
        }
        // Add individual stitch scores, not per-room sums. Changing the
        // floating-point grouping can change finite-difference gradients.
        for (r, chains) in by_room.iter().enumerate() {
            if let Some(base) = cache.filter(|_| !dirty[r]) {
                for &score in &base.junctions[r] {
                    junction_jog += score;
                }
            } else {
                for score in score_junctions(chains, boundary, scale) {
                    junction_jog += score;
                }
            }
        }
    }
    let count = shared.room_count() as f64;
    (direction_score / count, (pair_jog + 0.5 * junction_jog) / count)
}

// Score the transverse offset along a room's internal walls. Different
// neighbors may supply the arms of a step; distant supporting lines cannot
// pull on each other unless the connecting wall makes opposing turns.
fn room_jog(points: &[[f64; 2]], boundary: &Polygon<f64>, scale: f64, joins: &[[f64; 2]]) -> f64 {
    let dot = |a: [f64; 2], b: [f64; 2]| a[0] * b[0] + a[1] * b[1];
    let cross = |a: [f64; 2], b: [f64; 2]| a[0] * b[1] - a[1] * b[0];
    let sub = |a: [f64; 2], b: [f64; 2]| [a[0] - b[0], a[1] - b[1]];
    let fade = |x: f64| {
        let x = x.clamp(0., 1.);
        x * x * (3. - 2. * x)
    };
    let edges: Vec<_> = points
        .windows(2)
        .map(|p| {
            let d = sub(p[1], p[0]);
            let length = d[0].hypot(d[1]);
            let heading = [d[0] / length, d[1] / length];
            let join = joins.iter().any(|&j| {
                if j == p[0] || j == p[1] {
                    return true;
                }
                let delta = sub(j, p[0]);
                let along = dot(delta, heading);
                let eps = 1e-10 * scale;
                cross(delta, heading).abs() <= eps && along >= -eps && along <= length + eps
            });
            let (c, s) = crate::walls::nearest_boundary_direction(
                (p[0][0] + p[1][0]) / 2.,
                (p[0][1] + p[1][1]) / 2.,
                boundary,
            );
            let mut axis = [c as f64, s as f64];
            if cross(heading, axis).abs() > dot(heading, axis).abs() {
                axis = [-axis[1], axis[0]];
            }
            if dot(heading, axis) < 0. {
                axis = [-axis[0], -axis[1]];
            }
            (heading, length, join, axis)
        })
        .collect();
    if edges.len() < 3 {
        return 0.;
    }
    let closed = points.first() == points.last();
    let mut energy = 0.;
    for i in 0..edges.len() {
        if !closed && i + 2 >= edges.len() {
            break;
        }
        let n = edges.len();
        let (_, al, _, a) = edges[i];
        let turn_in = cross(a, edges[(i + 1) % n].0);
        let into = fade((turn_in.abs() - 0.25) / 0.45);
        if into == 0. {
            continue;
        }
        let mut best: f64 = 0.;
        // Limit the connector to a local excursion. A broad sequence that
        // runs back along the first arm is a distinct room shape.
        let count = if closed { n - 1 } else { n - i - 1 };
        let mut travel = 0.;
        let mut crosses_join = false;
        for gap in 2..=count {
            let last = (i + gap - 1) % n;
            travel += edges[last].1;
            crosses_join |= edges[last].2;
            let end = (i + gap) % n;
            if travel > 0.65 * scale {
                break;
            }
            if !crosses_join {
                continue;
            }
            let (_, bl, _, b) = edges[end];
            if dot(a, b) < 1. - 1e-6 {
                continue;
            }
            let turn_out = -turn_in.signum() * cross(edges[last].0, b);
            let out = fade((turn_out - 0.25) / 0.45);
            if out == 0. {
                continue;
            }
            let shift = sub(points[end], points[i + 1]);
            let offset = cross(a, shift).abs();
            let transverse = fade((offset / (travel + f64::MIN_POSITIVE) - 0.5) / 0.35);
            let arms = fade(al / (0.03 * scale)) * fade(bl / (0.03 * scale));
            // Fit both visible runs to the same supporting line. Uniform
            // samples along each arm have the squared center separation
            // plus the two linear transverse variances (span²/12).
            let first_span = cross(a, sub(points[i + 1], points[i]));
            let second_span = cross(a, sub(points[end + 1], points[end]));
            let center_delta = cross(a, shift) + 0.5 * (first_span + second_span);
            let squared_shift = center_delta * center_delta
                + (first_span * first_span + second_span * second_span) / 12.;
            let z = squared_shift / (0.084 * scale).powi(2);
            let cost = (2. * z / ((1. + z).sqrt() + 1.)) / 4.;
            best = best.max(cost * into * out * transverse * arms);
        }
        energy += best;
    }
    energy
}

#[cfg(test)]
fn losses(chains: &[Vec<[f64; 2]>], boundary: &Polygon<f64>) -> (f64, f64) {
    let tol = boundary.unsigned_area().sqrt() * 1e-12;
    let (mut direction, mut jog) = (0., 0.);
    for raw in chains {
        let points = canonical_chain(raw, tol);
        direction += chain_direction(&points, boundary);
        jog += chain_jog(&points, boundary);
    }
    (direction, jog)
}

fn chain_direction(points: &[[f64; 2]], boundary: &Polygon<f64>) -> f64 {
    let (mut weighted, mut length) = (0., 0.);
    for pair in points.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l = dx.hypot(dy);
        if l == 0. {
            continue;
        }
        let (ex, ey) = crate::walls::nearest_boundary_direction(
            (a[0] + b[0]) / 2.,
            (a[1] + b[1]) / 2.,
            boundary,
        );
        let (ex, ey) = (ex as f64, ey as f64);
        let cosine = (dx * ex + dy * ey) / (l * ex.hypot(ey));
        let c2 = (cosine * cosine).clamp(0., 1.);
        weighted += l * (c2 * (1. - c2));
        length += l;
    }
    if length != 0. {
        weighted / length
    } else {
        0.
    }
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
    // Pair energy has different units from the former bounded variance.
    // Match the cost of a 0.12-unit step in the two-room reference case
    // so the existing step weight remains comparable.
    let scale2=0.042_f64.powi(2)*boundary.unsigned_area();
    let softness=scale2.sqrt();
    let mut groups=[[0.;4];4];
    let mut runs=Vec::new();
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
        let extent=du.abs().max(dv.abs());
        runs.push((group,extent,offset));
    }
    for &(group,extent,offset) in &runs{
        // Nearly collinear pieces share one visibility budget continuously.
        // For identical supports their extents add before saturation, so an
        // arbitrarily small subdivision bend cannot multiply another step.
        // Supports farther apart than one quarter of the step scale do
        // not share weight. Longer distinct runs cannot dilute the step.
        let density: f64 = runs.iter().filter(|&&(heading, _, _)| heading == group)
            .map(|&(_, other_extent, other_offset)| {
                let distance_squared = (offset - other_offset).powi(2);
                let similarity = (1. - 16. * distance_squared / scale2).max(0.);
                other_extent * similarity.powi(2)
        }).sum();
        let weight = extent / (density + softness);
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
