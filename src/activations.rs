use crate::{constants::FloatPrecision, math::DMatrix};

pub struct Activation {
    pub f: fn(FloatPrecision) -> FloatPrecision,
    pub fd: fn(FloatPrecision) -> FloatPrecision
}

pub fn mwrap(f: fn(FloatPrecision) -> FloatPrecision, m: &DMatrix, result: &mut DMatrix) {
    for i in 0..m.data.len(){
        result.data[i] = f(m.data[i])
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