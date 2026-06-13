//! AdamW matching `torch.optim.AdamW` (torch 2.1, single-tensor, f32
//! parameters) with the example defaults: betas (0.9, 0.999), eps 1e-8,
//! weight_decay 0.01. Scalar quantities (lr * wd, bias corrections,
//! step_size) are computed in f64 exactly as Python computes them, and cast
//! to f32 only where torch applies them to tensors.

pub struct AdamW {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    step: i32,
    exp_avg: Vec<[f32; 2]>,
    exp_avg_sq: Vec<[f32; 2]>,
}

impl AdamW {
    pub fn new(n_params: usize, lr: f64) -> Self {
        AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step: 0,
            exp_avg: vec![[0.0; 2]; n_params],
            exp_avg_sq: vec![[0.0; 2]; n_params],
        }
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// One `optimizer.step()`, mirroring torch 2.1's `_single_tensor_adamw`:
    ///   param *= 1 - lr*wd                       (decoupled weight decay)
    ///   exp_avg.lerp_(grad, 1-beta1)
    ///   exp_avg_sq = exp_avg_sq*beta2 + (1-beta2)*grad^2
    ///   denom = sqrt(exp_avg_sq)/sqrt(bc2) + eps
    ///   param += -lr/bc1 * exp_avg/denom
    /// Scalars are f64 (Python floats) cast to f32 at the tensor boundary;
    /// elementwise tensor math is f32.
    pub fn step(&mut self, params: &mut [[f32; 2]], grads: &[[f32; 2]]) {
        self.step += 1;
        let step = self.step as f64;

        let decay = (1.0 - self.lr * self.weight_decay) as f32;
        let lerp_w = (1.0 - self.beta1) as f32;
        let beta2 = self.beta2 as f32;
        let one_minus_beta2 = (1.0 - self.beta2) as f32;
        let bias_correction1 = 1.0 - self.beta1.powf(step);
        let bias_correction2 = 1.0 - self.beta2.powf(step);
        let step_size = (self.lr / bias_correction1) as f32;
        let bc2_sqrt = bias_correction2.sqrt() as f32;
        let eps = self.eps as f32;

        for ((p, g), (m, v)) in params
            .iter_mut()
            .zip(grads)
            .zip(self.exp_avg.iter_mut().zip(self.exp_avg_sq.iter_mut()))
        {
            for j in 0..2 {
                p[j] *= decay;
                m[j] += lerp_w * (g[j] - m[j]);
                v[j] = v[j] * beta2 + one_minus_beta2 * g[j] * g[j];
                let denom = v[j].sqrt() / bc2_sqrt + eps;
                p[j] += -step_size * (m[j] / denom);
            }
        }
    }
}
