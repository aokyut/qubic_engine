use std::fmt::Binary;

use super::Node;
use super::Tensor;

pub trait SingleShoot {
    fn single_forward(&self, x: f32) -> f32;
    fn single_backward(&self, x: f32, y: f32) -> f32;
    fn no_grad(&self) -> bool {
        false
    }
}
trait SingleShootLoss {
    fn single_forward_loss(&self, x1: f32, x2: f32) -> f32;
    fn single_backward_loss(&self, x1: f32, x2: f32, y: f32) -> (f32, f32);
    fn no_grad(&self) -> bool {
        false
    }
}

impl<F: SingleShoot> Node for F {
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, output: &Tensor) -> Vec<Tensor> {
        let mut igrad = Tensor::zeros_like(inputs[0]);

        for i in 0..grad.data.len() {
            igrad.data[i] = grad.data[i] * self.single_backward(inputs[0].data[i], output.data[i]);
        }

        return vec![igrad];
    }
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 1);
        let input = &input[0];
        let mut output = Tensor::zeros_like(&input);
        for i in 0..input.data.len() {
            output.data[i] = self.single_forward(input.data[i]);
        }

        return output;
    }
}

// impl<F: SingleShootLoss> Node for F {
//     fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, output: &Tensor) -> Vec<Tensor> {
//         let mut igrad1 = Tensor::zeros_like(inputs[0]);
//         let mut igrad2 = Tensor::zeros_like(inputs[1]);
//         let scale = 1.0 / inputs[0].data.len() as f32;
//         let loss = output.data[0];

//         for i in 0..grad.data.len() {
//             let (x1_, x2_) = self.single_backward_loss(inputs[0].data[i], inputs[1].data[i], loss);
//             igrad1.data[i] = grad.data[i] * scale * x1_;
//             igrad2.data[i] = grad.data[i] * scale * x2_;
//         }

//         return vec![igrad1, igrad2];
//     }
//     fn call(&self, input: Vec<Tensor>) -> Tensor {
//         assert_eq!(input.len(), 1);
//         let xs1 = &input[0];
//         let xs2 = &input[1];
//         let size = xs1.data.len() as f32;
//         let mut loss = 0.0;
//         for i in 0..input.data.len() {
//             loss += self.single_forward_loss(xs1.data[i], xs2.data[i]);
//         }

//         return Tensor::new(vec![loss / size], vec![1]);
//     }
// }

pub struct ReLU {
    ignore_grad: bool,
}

impl ReLU {
    pub fn new() -> Self {
        return ReLU { ignore_grad: false };
    }
}

impl Node for ReLU {
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let mut igrad = Tensor::zeros_like(inputs[0]);

        for i in 0..grad.data.len() {
            igrad.data[i] = grad.data[i] * inputs[0].data[i].signum().max(0.0);
        }

        return vec![igrad];
    }
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 1);
        let input = &input[0];
        let mut output = Tensor::zeros_like(&input);
        for i in 0..input.data.len() {
            output.data[i] = input.data[i].max(0.0);
        }

        return output;
    }
    fn no_grad(&self) -> bool {
        self.ignore_grad
    }
}

pub struct LeaklyReLU {
    ignore_grad: bool,
    alpha: f32,
}

impl LeaklyReLU {
    pub fn new(alpha: f32) -> Self {
        return LeaklyReLU {
            ignore_grad: false,
            alpha: alpha,
        };
    }

    pub fn default() -> Self {
        return LeaklyReLU {
            ignore_grad: false,
            alpha: 0.01,
        };
    }
}

impl Node for LeaklyReLU {
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 1);
        let input = &input[0];
        let mut output = Tensor::zeros_like(&input);
        for i in 0..input.data.len() {
            let x = input.data[i];
            if x > 0.0 {
                output.data[i] = x;
            } else {
                output.data[i] = self.alpha * x;
            }
        }

        return output;
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let mut igrad = Tensor::zeros_like(inputs[0]);

        for i in 0..grad.data.len() {
            let x = inputs[0].data[i];
            if x < 0.0 {
                igrad.data[i] = self.alpha * grad.data[i];
            } else {
                igrad.data[i] = grad.data[i];
            }
        }

        return vec![igrad];
    }
    fn no_grad(&self) -> bool {
        self.ignore_grad
    }
}

pub struct ClippedReLU {
    ignore_grad: bool,
    ceil: f32,
}

impl ClippedReLU {
    pub fn new(ceil: f32) -> Self {
        assert!(ceil > 0.0);
        return ClippedReLU {
            ignore_grad: false,
            ceil: ceil,
        };
    }

    pub fn default() -> Self {
        return ClippedReLU::new(1.0);
    }
}

impl Node for ClippedReLU {
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 1);
        let input = &input[0];
        let mut output = Tensor::zeros_like(&input);
        for i in 0..input.data.len() {
            output.data[i] = input.data[i].max(0.0).min(self.ceil);
        }

        return output;
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let mut igrad = Tensor::zeros_like(inputs[0]);

        for i in 0..grad.data.len() {
            let x = inputs[0].data[i];
            if x < 0.0 {
                igrad.data[i] = 0.0
            } else if x > self.ceil {
                igrad.data[i] = 0.0
            } else {
                igrad.data[i] = grad.data[i];
            }
        }

        return vec![igrad];
    }
    fn no_grad(&self) -> bool {
        return self.ignore_grad;
    }
}

pub struct Tanh {}

impl Tanh {
    pub fn new() -> Self {
        return Tanh {};
    }
}

impl Node for Tanh {
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 1);
        let input = &input[0];
        let mut output = Tensor::zeros_like(&input);
        for i in 0..input.data.len() {
            output.data[i] = input.data[i].tanh();
        }

        return output;
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, output: &Tensor) -> Vec<Tensor> {
        let mut igrad = Tensor::zeros_like(inputs[0]);

        for i in 0..grad.data.len() {
            let x = inputs[0].data[i];
            igrad.data[i] = 1.0 - output.data[i].powi(2);
        }

        return vec![igrad];
    }
    fn no_grad(&self) -> bool {
        return false;
    }
}

pub struct Sigmoid {
    alpha: f32,
}

impl Sigmoid {
    pub fn new(alpha: f32) -> Self {
        return Sigmoid { alpha: alpha };
    }
}

impl SingleShoot for Sigmoid {
    fn single_backward(&self, _: f32, y: f32) -> f32 {
        self.alpha * y * (1.0 - y)
    }
    fn single_forward(&self, x: f32) -> f32 {
        let y = 1.0 / (1.0 + (-x * self.alpha).exp());
        return y;
    }
}

pub struct Softmax {}

impl Softmax {
    pub fn new() -> Self {
        Softmax {}
    }
}

impl Node for Softmax {
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 1);

        let &width = input[0].shape.last().unwrap();
        let batch = input[0].data.len() / width;
        let input = &input[0];
        let mut output = Tensor::zeros_like(&input);

        for i in 0..batch {
            let offset = i * width;
            let mut sum = 0.0;
            for j in 0..width {
                output.data[offset + j] = input.data[offset + j].exp();
                sum += output.data[offset + j]
            }
            for j in 0..width {
                output.data[offset + j] /= sum;
            }
        }

        return output;
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, output: &Tensor) -> Vec<Tensor> {
        let mut igrad = Tensor::zeros_like(inputs[0]);
        let &width = inputs[0].shape.last().unwrap();
        let batch = inputs[0].data.len() / width;

        for i in 0..batch {
            let offset = i * width;
            for j in 0..width {
                for k in j..width {
                    igrad.data[offset + j] -=
                        output.data[offset + j] * output.data[offset + k] * grad.data[offset + k];
                    igrad.data[offset + k] -=
                        output.data[offset + k] * output.data[offset + j] * grad.data[offset + j];
                }
                igrad.data[offset + j] += output.data[offset + j]
                    * (1.0 - output.data[offset + j])
                    * grad.data[offset + j];
            }
        }

        return vec![igrad];
    }
    fn no_grad(&self) -> bool {
        return false;
    }
}

pub struct MSE {}

impl MSE {
    pub fn new() -> Self {
        MSE {}
    }
}

impl Node for MSE {
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 2);
        // let loss = Tensor::zeros(vec![1]);
        let left = &input[0].data;
        let right = &input[1].data;
        assert_eq!(left.len(), right.len());
        let mut loss = 0.0;

        for i in 0..left.len() {
            loss += (left[i] - right[i]).powi(2);
        }

        return Tensor::new(vec![loss / left.len() as f32], vec![1]);
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let mut left = Tensor::zeros_like(inputs[0]);
        let mut right = Tensor::zeros_like(inputs[1]);

        for i in 0..left.data.len() {
            left.data[i] = 2.0 * (inputs[0].data[i] - inputs[1].data[i]) * grad.data[0]
                / left.data.len() as f32;
            right.data[i] = 2.0 * (inputs[1].data[i] - inputs[0].data[i]) * grad.data[0]
                / left.data.len() as f32;
        }

        return vec![left, right];
    }
    fn no_grad(&self) -> bool {
        false
    }
}

pub struct BinaryCrossEntropy {
    eps: f32,
}

impl BinaryCrossEntropy {
    pub fn default() -> Self {
        return BinaryCrossEntropy { eps: 0.001 };
    }
}

impl Node for BinaryCrossEntropy {
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 2);
        // let loss = Tensor::zeros(vec![1]);
        let left = &input[0].data;
        let right = &input[1].data;
        assert_eq!(left.len(), right.len());
        let mut loss = 0.0;
        let scale = 1.0 / (left.len()) as f32;

        // left: src
        // right: tar
        for i in 0..left.len() {
            loss += -left[i].ln() * right[i] - (1.0 - left[i]).ln() * (1.0 - right[i]);
        }

        return Tensor::new(vec![loss * scale], vec![1]);
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let mut left = Tensor::zeros_like(inputs[0]);
        let mut right = Tensor::zeros_like(inputs[1]);
        let scale = 1.0 / (left.data.len() as f32);

        for i in 0..left.data.len() {
            left.data[i] = (-right.data[i] / (left.data[i] + self.eps)
                + (1.0 - right.data[i]) / (1.0 - left.data[i] + self.eps))
                * grad.data[0];
            right.data[i] = -left.data[i].ln() + (1.0 - left.data[i]).ln();
        }

        return vec![left, right];
    }
    fn no_grad(&self) -> bool {
        false
    }
}
