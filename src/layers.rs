use crate::constants::FloatPrecision;

use crate::math::addm;
use crate::math::addm_assign;
use crate::math::linm;
use crate::math::mulm;
use crate::math::naive_mulm;
use crate::math::naive_mulm_assign;
use crate::math::scale;
use crate::math::smmulmt;
use crate::math::DMatrix;

use crate::activations::Activation;
use crate::activations::mwrap;
use rand::Rng;
use std::fmt;
use std::process::exit;

// A layer that is densly connected with the previous one
pub struct Layer {
    input_size: usize,
    output_size: usize,
    pub weights: DMatrix,
    pub bias: DMatrix,
    pub activation: Activation,
    pub net: DMatrix,
    pub out: DMatrix,
    pub delta: DMatrix,
    fdnet: DMatrix,
    pub dw: DMatrix,
    pub db: DMatrix,
    rate: FloatPrecision
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation, rate: FloatPrecision) -> Self {
        let mut rng = rand::thread_rng();
        let weights_data:Vec<FloatPrecision> = (0..output_size*input_size).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let bias_data:Vec<FloatPrecision> = (0..output_size).map(|_| rng.gen_range(-0.5..0.5)).collect();

        let weights = DMatrix { data: weights_data, shape: (output_size, input_size)};
        let bias = DMatrix { data: bias_data, shape: (output_size, 1)};
        Self {
            input_size,
            output_size,
            weights,
            bias,
            activation,
            net: DMatrix::new(vec![0.;output_size], (output_size, 1)),
            out: DMatrix::new(vec![0.;output_size], (output_size, 1)),
            fdnet: DMatrix::new(vec![0.;output_size], (output_size, 1)),
            delta: DMatrix::new(vec![0.;output_size], (output_size, 1)),
            dw: DMatrix::new(vec![0.;output_size*input_size], (output_size, input_size)),
            db: DMatrix::new(vec![0.;output_size], (output_size, 1)),
            rate
        }
    }

    pub fn forward(&mut self, input: &DMatrix) {
        linm(&self.weights, input, &self.bias, &mut self.net); // Wx+b
        mwrap(self.activation.f, &self.net, &mut self.out); // s(Wx+b)
    }

    pub fn backward(&mut self, input: &DMatrix) {
        mwrap(self.activation.fd, &self.net, &mut self.fdnet); // f'(net)
        naive_mulm_assign(&mut self.delta, &self.fdnet); // dE * f'(net)
        smmulmt(self.rate, &self.delta, input, &mut self.dw); // dW = rate * delta * inputT
        scale(self.rate, &self.delta, &mut self.db); // db = rate * delta

        addm_assign(&mut self.weights, &self.dw);
        addm_assign(&mut self.bias, &self.db);
    }
}