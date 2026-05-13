use minimum_ml::ml::{Node, Tensor, TensorData, xiver_vec, Result};

pub enum TrainMode{
    Param,
    Quantize
}

struct QuantizedWeight{

}

pub struct FakeQuantizedClippedReLULinear {
    pub w: Tensor,
    pub b: Tensor,
    pub quantized_w: Option<Tensor>,
    pub quantized_b: Option<Tensor>,
    pub w_grad: Option<Tensor>,
    pub b_grad: Option<Tensor>,
    pub in_n: usize,
    pub out_n: usize,
    // pub quantized_w: QuantizedWeight,
    mode: TrainMode,
    is_inference: bool
}

impl FakeQuantizedClippedReLULinear {
    // 8bitでの行列積を行い16bit整数を作り、0-1でのclippedReLU処理を行う
    pub fn auto(input: usize, output: usize) -> Self{
        return Self { in_n: input, out_n: output, w_grad: None, b_grad: None, quantized_w: None, quantized_b: None, w: Tensor::new(xiver_vec(input, input * output), vec![input, output]), b: Tensor::zeros(vec![output]), mode: TrainMode::Param, is_inference: false};
    }

    pub fn prepare_inference_with_calib(&mut self, calibration_data: Option<&Tensor>){
        
    }
}

impl Node for FakeQuantizedClippedReLULinear{
    fn call(&self, input_vec: Vec<Tensor>) -> Tensor {
        assert_eq!(input_vec.len(), 1);
        let input = &input_vec[0];

        if self.is_inference {
            match &input.data {
                TensorData::I8 { data, scales } => {
                    let batch_size = scales.len();
                    let mut out_data = Vec::with_capacity(batch_size * self.out_n);
                    let mut out_scales = Vec::with_capacity(batch_size);
                    let b_f32 = self.b.as_f32_slice();

                    return Tensor::new_i8(out_data, out_scales, vec![batch_size, self.out_n]);
                }
                TensorData::F32(data) => {
                    panic!("Input for FakeQuantizedClippedReLULinear-inference must be I8");
                }
            }
        }

        // Fallback to F32 if IntMM is not prepared
        let input_f32 = input.as_f32_slice();
        let in_features = *input.shape.last().unwrap();
        let out_features = self.w.shape[0];

        // Calculate batch size by flattening all dimensions except the last
        let batch = input_f32.len() / in_features;

        let mut ans_shape = input.shape.clone();
        *ans_shape.last_mut().unwrap() = out_features;
        let mut ans_data = vec![0.0; batch * out_features];
        let w_f32 = self.w.as_f32_slice();
        let b_f32 = self.b.as_f32_slice();

        for b in 0..batch {
            let offset_input = b * in_features;
            let offset_ans = b * out_features;
            for i in 0..out_features {
                let mut sum = b_f32[i];
                for j in 0..in_features {
                    sum += input_f32[offset_input + j] * w_f32[i * in_features + j];
                }
                ans_data[offset_ans + i] = sum;
            }
        }
        Tensor::new(ans_data, ans_shape)
    }

    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let input = inputs[0];
        let in_features = *input.shape.last().unwrap();
        let out_features = self.out_n;

        // Calculate batch size by flattening all dimensions except the last
        let input_data = input.as_f32_slice();
        let batch = input_data.len() / in_features;

        let mut w_grad = Tensor::zeros_like(&self.w);
        let mut b_grad = Tensor::zeros_like(&self.b);
        let mut input_grad = Tensor::zeros_like(input);

        let grad_data = grad.as_f32_slice();
        let input_data = input.as_f32_slice();
        let w_data = self.w.as_f32_slice();

        let w_grad_f32 = w_grad.f32_data_mut();
        let b_grad_f32 = b_grad.f32_data_mut();
        let input_grad_f32 = input_grad.f32_data_mut();

        for b in 0..batch {
            let offset_input = b * in_features;
            let offset_grad = b * out_features;
            for i in 0..out_features {
                let gi = grad_data[offset_grad + i];
                b_grad_f32[i] += gi;
                for j in 0..in_features {
                    w_grad_f32[i * in_features + j] += gi * input_data[offset_input + j];
                    input_grad_f32[offset_input + j] += gi * w_data[i * in_features + j];
                }
            }
        }

        if let Some(_w_grad) = self.w_grad.as_mut() {
            *_w_grad += w_grad;
        } else {
            self.w_grad = Some(w_grad);
        }
        if let Some(_b_grad) = self.b_grad.as_mut() {
            *_b_grad += b_grad;
        } else {
            self.b_grad = Some(b_grad);
        }

        vec![input_grad]
    }

    fn prepare_inference(&mut self) {
        self.is_inference = true;
    }
    fn prepare_train(&mut self) {
        self.is_inference = false;
    }
    fn no_grad(&self) -> bool {
        false
    }
    fn has_params(&self) -> bool {
        true
    }
    fn pull_grad(&self) -> Option<Vec<&Tensor>> {
        None
    }
    fn save_param(&self, path: std::path::PathBuf) -> Result<()> {
        // use minimum_ml::ml::binary_io::*;
        // use std::fs::File;
        // use std::io::BufWriter;

        // let file = File::create(path)?;
        // let mut writer = BufWriter::new(file);

        // write_header(&mut writer, TYPE_QUANTIZED_LINEAR)?;
        // write_tensor_data(&mut writer, self.w.as_f32_slice().as_ref(), &self.w.shape)?;
        // write_tensor_data(&mut writer, self.b.as_f32_slice().as_ref(), &self.b.shape)?;

        Ok(())
    }

    fn load_param(&mut self, path: std::path::PathBuf) -> Result<()> {
        // use minimum_ml::ml::binary_io::*;
        // use std::fs::File;
        // use std::io::BufReader;

        // let file = File::open(path)?;
        // let mut reader = BufReader::new(file);

        // read_header(&mut reader, TYPE_QUANTIZED_LINEAR)?;

        // let (w_data, w_shape) = read_tensor_data(&mut reader)?;
        // let (b_data, b_shape) = read_tensor_data(&mut reader)?;

        // self.w = crate::ml::Tensor {
        //     data: crate::ml::TensorData::F32(w_data),
        //     shape: w_shape,
        // };
        // self.b = crate::ml::Tensor {
        //     data: crate::ml::TensorData::F32(b_data),
        //     shape: b_shape,
        // };

        Ok(())
    }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[target_feature(enable = "avx2")]
pub unsafe fn mul_8x4ni8_4nu8_i32(a: &[i8], b: &[u8], output: &mut[i32]){ unsafe {
    let a_ptr = a.as_ptr() as (*const i8);
    let b_ptr = b.as_ptr() as (*const i32);

    let mut acum_vec = _mm256_setzero_si256();

    for idx in 0..(a.len() >> 1){
        let a_vec = _mm256_loadu_epi8(a_ptr.add(idx * 32));
        let b_vec = _mm256_set1_epi32(*(b_ptr.add(idx)));
        let c_vec = _mm256_maddubs_epi16(b_vec, a_vec);
        let c_low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(c_vec));
        let c_high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(c_vec));
        acum_vec = _mm256_add_epi32(acum_vec, 
            _mm256_add_epi32(c_low, c_high)
        );
    }

    _mm256_storeu_epi32(output.as_mut_ptr(), acum_vec);
}}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[target_feature(enable = "avx2")]
pub unsafe fn mul_16x16nu8_16ni8_sr4(a: &[u8], b: &[i8], output: &mut[i16]){ unsafe {
    let a_ptr = a.as_ptr() as (*const i8);
    let b_ptr = b.as_ptr() as (*const i16);

    let mut c_vec = _mm256_setzero_si256();

    for idx in 0..(a.len()/2){  
        let a_vec = _mm256_loadu_epi8(a_ptr.add(idx * 32));
        let b_vec = _mm256_set1_epi16(*(b_ptr.add(idx)));
        c_vec = _mm256_adds_epi16(c_vec, 
            _mm256_srai_epi16::<4>(_mm256_maddubs_epi16(a_vec, b_vec)));
    }

    _mm256_storeu_epi16(output.as_mut_ptr(), c_vec);
}}

#[cfg(target_arch="x86_64")]
use std::arch::x86_64::*;
#[target_feature(enable = "avx2")]
pub unsafe fn mul_16x2nu8_2ni8(a: &[u8], b: &[i8], output: &mut[i16]){ unsafe {
    // assert!(a.len() == b.len())
    // assert!(a.len() % 32 == 0)

    let mut result = _mm256_setzero_si256();
    let a_ptr = a.as_ptr() as (*const i8);
    let b_ptr = b.as_ptr() as (*const i16);

    let a_vec = _mm256_loadu_epi8(a_ptr);
    let b_vec = _mm256_set1_epi16(*(b_ptr));
    let mut c_vec = _mm256_maddubs_epi16(a_vec, b_vec);

    for idx in 1..(a.len()/2){  
        let a_vec = _mm256_loadu_epi8(a_ptr.add(idx * 32));
        let b_vec = _mm256_set1_epi16(*(b_ptr.add(idx)));
        c_vec = _mm256_adds_epi16(c_vec, _mm256_maddubs_epi16(a_vec, b_vec));        
    }

    _mm256_storeu_epi16(output.as_mut_ptr(), c_vec);
}}