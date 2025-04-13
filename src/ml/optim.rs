use std::collections::HashMap;

use super::{Optimizer, Tensor};

pub struct SGD {
    alpha: f32,
}

impl SGD {
    pub fn new(alpha: f32) -> Self {
        return Self { alpha: alpha };
    }
}

impl Optimizer for SGD {
    fn optimize(&mut self, _: usize, grads: Vec<&Tensor>) -> Vec<Tensor> {
        let mut ans_vec = Vec::new();

        for grad in grads {
            let mut ans = Tensor::zeros_like(grad);
            for i in 0..ans.data.len() {
                ans.data[i] = -grad.data[i] * self.alpha;
            }
            ans_vec.push(ans);
        }

        return ans_vec;
    }
}

pub struct MomentumSGD {
    alpha: f32,
    beta: f32,
    logs: HashMap<usize, Vec<Tensor>>,
}

impl MomentumSGD {
    // TODO
    pub fn new(alpha: f32, beta: f32) -> Self {
        return MomentumSGD {
            alpha: alpha,
            beta: beta,
            logs: HashMap::new(),
        };
    }
}

impl Optimizer for MomentumSGD {
    fn optimize(&mut self, tar_id: usize, grads: Vec<&Tensor>) -> Vec<Tensor> {
        let log = self.logs.get(&tar_id);
        if let Some(vn) = log {
            let mut w = Vec::new();
            let mut vn_ = Vec::new();
            for i in 0..grads.len() {
                let vni = &vn[i];
                let grad = grads[i];
                let mut vni_ = Tensor::zeros_like(grad);
                let mut wi = Tensor::zeros_like(grad);
                for j in 0..grad.data.len() {
                    vni_.data[j] = self.beta * vni.data[j] + (1.0 - self.beta) * grad.data[j];
                    wi.data[j] = -self.alpha * vni_.data[j];
                }

                w.push(wi);
                vn_.push(vni_);
            }
            self.logs.entry(tar_id).and_modify(|e| *e = vn_);
            return w;
        } else {
            let mut w = Vec::new();
            let mut vn: Vec<Tensor> = Vec::new();
            for i in 0..grads.len() {
                let grad = grads[i];
                let mut vni = Tensor::zeros_like(grad);
                let mut wi = Tensor::zeros_like(grad);
                for j in 0..grad.data.len() {
                    vni.data[j] = grad.data[j];
                    wi.data[j] = -self.alpha * vni.data[j];
                }
                vn.push(vni);
                w.push(wi);
            }
            self.logs.insert(tar_id, vn);
            return w;
        }
    }
}
