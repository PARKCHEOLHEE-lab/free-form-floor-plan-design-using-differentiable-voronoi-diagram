//! Central finite-difference gradients, ported from `FloorPlanLoss.backward`:
//! each site coordinate is perturbed by +-1e-6 *in f32* (torch adds the
//! scalar to a float32 tensor), the full forward loss is evaluated at both
//! perturbations, and the f32 difference is divided by 2e-6. Parallelized with
//! rayon instead of multiprocessing on native; the wasm build (no threads) runs
//! the evaluations serially. Per-coordinate results are identical either way
//! because each evaluation is independent.

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
    // epsilon = 1e-6 (a Python float); torch adds it to a float32 tensor, so
    // the perturbation itself rounds in f32
    const EPS: f32 = 1e-6;

    let n = sites.len();

    // One forward-difference evaluation per (site, axis); each is independent,
    // so the collection order does not affect the result.
    let eval = |idx: usize| -> f32 {
        let (i, j) = (idx / 2, idx % 2);
        let orig = sites[i][j];

        // One scratch copy reused for both perturbations. Each coordinate is
        // written from the *original* value (not the +EPS result), so the f32
        // perturbations are bitwise identical to perturbing two fresh clones.
        let mut buf = sites.to_vec();

        buf[i][j] = orig + EPS;
        let loss_pos =
            crate::loss::floor_plan_loss(&buf, boundary, target_areas, room_indices, w, hint).total;

        buf[i][j] = orig - EPS;
        let loss_neg =
            crate::loss::floor_plan_loss(&buf, boundary, target_areas, room_indices, w, hint).total;

        // (loss_pos - loss_neg) / (2 * epsilon): f32 subtraction, then division
        // by the f64 scalar 2e-6 coerced to f32
        (loss_pos - loss_neg) / (2.0 * EPS)
    };

    // With the `parallel` feature (default on native, and on the threaded wasm
    // build via wasm-bindgen-rayon) the 2N evaluations fan across rayon threads;
    // otherwise they run serially. The entries are bitwise identical either way.
    #[cfg(feature = "parallel")]
    let entries: Vec<f32> = {
        use rayon::prelude::*;
        (0..n * 2).into_par_iter().map(eval).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let entries: Vec<f32> = (0..n * 2).map(eval).collect();

    entries.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
}
