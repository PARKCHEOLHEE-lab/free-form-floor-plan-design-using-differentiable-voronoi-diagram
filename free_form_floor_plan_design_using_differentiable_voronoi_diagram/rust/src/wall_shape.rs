//! Direction of physical shared-wall chains.
use crate::walls::{canonical_chain, wall_chains, SharedWalls};
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

struct PairScores {
    edges: Vec<Line<f64>>,
    chains: Vec<f64>,
}

/// Direction scores for the base diagram of a single gradient evaluation. A
/// pair's ordered edge bits identify its inputs. Keep individual terms in
/// fold order.
pub(crate) struct WallShapeCache<'a> {
    boundary: &'a Polygon<f64>,
    pairs: BTreeMap<[usize; 2], PairScores>,
}

impl<'a> WallShapeCache<'a> {
    pub(crate) fn new(shared: &SharedWalls, boundary: &'a Polygon<f64>) -> Self {
        let scale = boundary.unsigned_area().sqrt();
        let pairs = shared.indexed_pairs().map(|(rooms, edges)| {
            (rooms, PairScores { edges: edges.to_vec(), chains: score_pair(edges, boundary, scale) })
        }).collect();
        Self { boundary, pairs }
    }

    pub(crate) fn evaluate(&self, shared: &SharedWalls) -> f64 {
        score_shared(shared, self.boundary, Some(self))
    }
}

fn same_edges(a: &[Line<f64>], b: &[Line<f64>]) -> bool {
    fn bits(line: &Line<f64>) -> [u64; 4] {
        [line.start.x.to_bits(), line.start.y.to_bits(), line.end.x.to_bits(), line.end.y.to_bits()]
    }
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| bits(a) == bits(b))
}

fn score_pair(edges: &[Line<f64>], boundary: &Polygon<f64>, scale: f64) -> Vec<f64> {
    #[cfg(test)]
    count_score_work(0);
    let tol = scale * 1e-12;
    wall_chains(edges, tol).into_iter().map(|chain| {
        #[cfg(test)]
        count_score_work(1);
        chain_direction(&canonical_chain(&chain, tol), boundary)
    }).collect()
}

pub(crate) fn evaluate_shared(shared: &SharedWalls, boundary: &Polygon<f64>) -> f64 {
    score_shared(shared, boundary, None)
}

fn score_shared(shared: &SharedWalls, boundary: &Polygon<f64>, cache: Option<&WallShapeCache<'_>>) -> f64 {
    if shared.room_count() < 2 {
        return 0.;
    }
    let scale = boundary.unsigned_area().sqrt();
    let mut direction = 0.;
    for (rooms, edges) in shared.indexed_pairs() {
        let saved = cache.and_then(|base| base.pairs.get(&rooms))
            .filter(|pair| same_edges(&pair.edges, edges));
        let chains = match saved {
            Some(pair) => Cow::Borrowed(pair.chains.as_slice()),
            None => Cow::Owned(score_pair(edges, boundary, scale)),
        };
        for score in chains.iter() {
            direction += score;
        }
    }
    // Room count is fixed during an optimization. Adding a perfect chain must
    // not lower another wall's error by changing the averaging denominator.
    direction / shared.room_count() as f64
}

#[cfg(test)]
fn losses(chains: &[Vec<[f64; 2]>], boundary: &Polygon<f64>) -> f64 {
    let tol = boundary.unsigned_area().sqrt() * 1e-12;
    let mut direction = 0.;
    for raw in chains {
        direction += chain_direction(&canonical_chain(raw, tol), boundary);
    }
    direction
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

#[cfg(test)]
#[path = "wall_shape_tests.rs"]
mod tests;
