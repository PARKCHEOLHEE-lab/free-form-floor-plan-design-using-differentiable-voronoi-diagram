//! Central finite-difference gradients, ported from `FloorPlanLoss.backward`:
//! each site coordinate is perturbed by +-1e-6 *in f32* (torch adds the
//! scalar to a float32 tensor), the full forward loss is evaluated at both
//! perturbations, and the f32 difference is divided by 2e-6. Parallelized
//! with rayon instead of multiprocessing; per-coordinate results are
//! identical because each evaluation is independent.

use crate::loss::LossWeights;
use crate::voronoi::GeosOrderHint;
use geo::Polygon;

pub fn finite_difference_grads(
    sites: &[[f32; 2]],
    boundary: &Polygon<f64>,
    target_areas: &[f64],
    room_indices: &[usize],
    w: &LossWeights,
    hint: Option<&GeosOrderHint>,
) -> Vec<[f32; 2]> {
    use rayon::prelude::*;

    // epsilon = 1e-6 (a Python float); torch adds it to a float32 tensor, so
    // the perturbation itself rounds in f32
    const EPS: f32 = 1e-6;

    let n = sites.len();
    let entries: Vec<f32> = (0..n * 2)
        .into_par_iter()
        .map(|idx| {
            let (i, j) = (idx / 2, idx % 2);

            let mut pos = sites.to_vec();
            pos[i][j] += EPS;
            let loss_pos =
                crate::loss::floor_plan_loss(&pos, boundary, target_areas, room_indices, w, hint)
                    .total;

            let mut neg = sites.to_vec();
            neg[i][j] -= EPS;
            let loss_neg =
                crate::loss::floor_plan_loss(&neg, boundary, target_areas, room_indices, w, hint)
                    .total;

            // (loss_pos - loss_neg) / (2 * epsilon): f32 subtraction, then
            // division by the f64 scalar 2e-6 coerced to f32
            (loss_pos - loss_neg) / (2.0 * EPS)
        })
        .collect();

    entries.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
}
