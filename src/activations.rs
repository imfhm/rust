use std::f64::MAX;

use plotters::data::float::FloatPrettyPrinter;

use crate::{constants::FloatPrecision, math::DMatrix};

pub struct Activation {
    pub f: fn(FloatPrecision) -> FloatPrecision,
    pub fd: fn(FloatPrecision) -> FloatPrecision,
}

pub fn mwrap(f: fn(FloatPrecision) -> FloatPrecision, m: &DMatrix, result: &mut DMatrix) {
    for i in 0..m.data.len() {
        result.data[i] = f(m.data[i])
    }
}


fn linear(x: FloatPrecision) -> FloatPrecision {
    x
}

fn linear_derivative(x: FloatPrecision) -> FloatPrecision {
    1.
}
fn relu(x: FloatPrecision) -> FloatPrecision {
    if x > 0. {
        x
    } else {
        0.
    }
}

fn relu_derivative(x: FloatPrecision) -> FloatPrecision {
    if x > 0. {
        1.
    } else {
        0.
    }
}

const ALPHA: FloatPrecision = 0.3;
fn leakyrelu(x: FloatPrecision) -> FloatPrecision {
    if x >= 0. {
        x
    } else {
        ALPHA * x
    }
}

fn leakyrelu_derivative(x: FloatPrecision) -> FloatPrecision {
    if x >= 0. {
        1.
    } else {
        ALPHA
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

pub const SIGMOID: Activation = Activation {
    f: sigmoid,
    fd: sigmoid_derivative,
};
pub const RELU: Activation = Activation {
    f: relu,
    fd: relu_derivative,
};
pub const LEAKYRELU: Activation = Activation {
    f: leakyrelu,
    fd: leakyrelu_derivative,
};

pub const LINEAR: Activation = Activation {
    f: linear,
    fd: linear_derivative,
};