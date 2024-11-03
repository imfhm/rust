use crate::constants::FloatPrecision;
use crate::math::Matrix;
use crate::math::dot;
use crate::math::add;
use crate::activations::Activation;
use crate::activations::mwrap;
use rand::Rng;
use std::fmt;
pub trait Layer {
    fn predict(&self, input: Matrix) -> Matrix;
    fn backprop(&mut self, prev_net: Matrix, error: &Matrix) -> (Matrix, Matrix);
}


// A layer that is densly connected with the previous one
pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    pub weights: Matrix,
    pub bias: Matrix,
    pub activation: Activation,
    rate: FloatPrecision
}

pub struct FlattenLayer {}
impl FlattenLayer {pub fn new()->Self{ Self {}}}
impl Layer for FlattenLayer {
    fn predict(&self, input: Matrix) -> Matrix {
        Matrix {data: input.data.clone(), shape: input.shape}
    }

    fn backprop(&mut self, _prev_net: Matrix, _error: &Matrix) -> (Matrix, Matrix) {
        (Matrix::zeros(0, 0), Matrix::zeros(0,0))
    }
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation, rate: FloatPrecision) -> Self {
        let mut rng = rand::thread_rng();
        let weights_data:Vec<FloatPrecision> = (0..output_size*input_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias_data:Vec<FloatPrecision> = (0..output_size*1).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let weights = Matrix { data: weights_data, shape: (output_size, input_size)};
        let bias = Matrix { data: bias_data, shape: (output_size, 1)};
        Self {
            input_size,
            output_size,
            weights,
            bias,
            activation,
            rate
        }
    }
}

impl Layer for DenseLayer {
    fn predict(&self, input: Matrix) -> Matrix {
        let sum = dot(self.weights.clone(), input);
        let net = add(sum, self.bias.clone());
        mwrap(self.activation.f, net)
    }

    fn backprop(&mut self, prev_net: Matrix, error: &Matrix) -> (Matrix, Matrix) {
        let fd = self.activation.fd;
        let (n,m) = self.weights.shape;
        let mut dw = Matrix::zeros(n,m);
        let mut delta = Matrix::zeros(n, 1);
        for i in 0..n {
            delta.set(i, 0, error.as_column(i) * fd(prev_net.as_column(i)));
            for j in 0..m {
                dw.set(i, j, self.rate * delta.as_column(i) * prev_net.as_column(j));
            }
        }
        (dw, delta)
    }
}

impl fmt::Display for DenseLayer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Weights: {:?}", self.weights);
        writeln!(f, "Biases: {:?}", self.bias)
    }
}
