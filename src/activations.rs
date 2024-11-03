use crate::{constants::FloatPrecision, math::Matrix};

pub struct Activation {
    pub f: fn(FloatPrecision) -> FloatPrecision,
    pub fd: fn(FloatPrecision) -> FloatPrecision
}

pub fn mwrap(f: fn(FloatPrecision) -> FloatPrecision, m: Matrix) -> Matrix {
    Matrix {
        data: m.data
        .iter()
        .map(|&x| f(x)).collect(),
        shape: m.shape
    }
}
/// Sigmoid activation function
fn sigmoid(x: FloatPrecision) -> FloatPrecision {
    1. / (1. + (-x).exp())
}

/// Derivative of the sigmoid function for backpropagation
fn sigmoid_derivative(x: FloatPrecision) -> FloatPrecision {
    let s = sigmoid(x);
    s * (1.0 - s)
}

pub const SIGMOID:Activation = Activation {f: sigmoid, fd: sigmoid_derivative};