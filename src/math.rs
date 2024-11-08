use std::fmt;
use std::ops;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;

use plotters::data::float::FloatPrettyPrinter;

use crate::constants::FloatPrecision;

pub fn argmax<T: PartialOrd>(slice: &[T]) -> Option<usize> {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
}

pub fn min(values: &Vec<FloatPrecision>) -> FloatPrecision {
    let mut min = values[0];
    for i in 1..values.len() {
        if min > values[i] {
            min = values[i]
        }
    }
    min
}
pub fn max(values: &Vec<FloatPrecision>) -> FloatPrecision {
    let mut max = values[0];
    for i in 1..values.len() {
        if max < values[i] {
            max = values[i]
        }
    }
    max
}

pub fn scale(s: FloatPrecision, rhs: &DMatrix, result: &mut DMatrix) {
    let (n, m) = rhs.shape;
    for i in 0..n {
        for j in 0..m {
            let index = i * m + j;
            result.data[index] = s * rhs.data[index];
        }
    }
}
pub fn mulm(lhs: &DMatrix, rhs: &DMatrix, result: &mut DMatrix) {
    let n = lhs.shape.0;
    let K = lhs.shape.1;
    let m = rhs.shape.1;

    for i in 0..n {
        let rowr = i * m;
        let rowlhs = i * K;
        for j in 0..m {
            result.data[rowr + j] = 0.;
            for k in 0..K {
                result.data[rowr + j] += lhs.data[rowlhs + k] * rhs.data[k * m + j];
            }
        }
    }
}

pub fn mtmulm(lhs: &DMatrix, rhs: &DMatrix, result: &mut DMatrix) {
    // e.g. (10, 32), (10, 1)
    let n = lhs.shape.1; // number of columns of transpose, e.g. 32
    let K = lhs.shape.0; // number of rows of transpose, e.g. 10
    let m = rhs.shape.1; // e.g. 1

    for i in 0..n {
        for j in 0..m {
            result.data[i * m + j] = 0.;
            for k in 0..K {
                result.data[i * m + j] += lhs.data[k * n + i] * rhs.data[k * m + j];
            }
        }
    }
}
pub fn naive_mulm(lhs: &DMatrix, rhs: &DMatrix, result: &mut DMatrix) {
    let n = lhs.shape.0;
    let m = rhs.shape.1;

    for i in 0..n {
        for j in 0..m {
            let index = i * m + j;
            result.data[index] = lhs.data[index] * rhs.data[index];
        }
    }
}
pub fn naive_mulm_assign(lhs: &mut DMatrix, rhs: &DMatrix) {
    let n = lhs.shape.0;
    let m = rhs.shape.1;

    for i in 0..n {
        for j in 0..m {
            let index = i * m + j;
            lhs.data[index] = lhs.data[index] * rhs.data[index];
        }
    }
}

pub fn linm(lhs: &DMatrix, rhs: &DMatrix, q: &DMatrix, result: &mut DMatrix) {
    let n = lhs.shape.0;
    let K = lhs.shape.1;
    let m = rhs.shape.1;

    for i in 0..n {
        let rowr = i * m;
        let rowlhs = i * K;
        for j in 0..m {
            let index = rowr + j;
            result.data[index] = 0.;
            for k in 0..K {
                result.data[index] += lhs.data[rowlhs + k] * rhs.data[k * m + j];
            }
            result.data[index] += q.data[index];
        }
    }
}

pub fn smmulmt(s: FloatPrecision, lhs: &DMatrix, rhs: &DMatrix, result: &mut DMatrix) {
    let n = lhs.shape.0;
    let K = rhs.shape.1; // number of rows of transpose
    let m = rhs.shape.0; // number of columns of transpose

    for i in 0..n {
        for j in 0..m {
            result.data[i * m + j] = 0.;
            for k in 0..K {
                result.data[i * m + j] += lhs.data[i * K + k] * rhs.data[j * K + k];
            }
            result.data[i * m + j] *= s;
        }
    }
}

pub fn addm(lhs: &DMatrix, rhs: &DMatrix, result: &mut DMatrix) {
    let (n, m) = lhs.shape;
    for i in 0..n {
        let row = i * m;
        for j in 0..m {
            let index = row + j;
            result.data[index] = lhs.data[index] + rhs.data[index];
        }
    }
}

pub fn addm_assign(lhs: &mut DMatrix, rhs: &DMatrix) {
    let (n, m) = lhs.shape;
    for i in 0..n {
        let row = i * m;
        for j in 0..m {
            let index = row + j;
            lhs.data[index] = lhs.data[index] + rhs.data[index];
        }
    }
}

pub fn subm(lhs: &DMatrix, rhs: &DMatrix, result: &mut DMatrix) {
    let (n, m) = lhs.shape;
    for i in 0..n {
        let row = i * m;
        for j in 0..m {
            let index = row + j;
            result.data[index] = (lhs.data[index] - rhs.data[index]);
        }
    }
}

pub fn ssubm(s: FloatPrecision, lhs: &DMatrix, rhs: &DMatrix, result: &mut DMatrix) {
    let (n, m) = lhs.shape;
    for i in 0..n {
        let row = i * m;
        for j in 0..m {
            let index = row + j;
            result.data[index] = s * (lhs.data[index] - rhs.data[index]);
        }
    }
}

// A dynamically sized Matrix implementation.
#[derive(Debug, Clone)]
pub struct DMatrix {
    pub data: Vec<FloatPrecision>,
    pub shape: (usize, usize),
}

impl DMatrix {
    pub fn new(data: Vec<FloatPrecision>, shape: (usize, usize)) -> Self {
        let d = data.len();
        let (n, m) = shape;
        if d != n * m {
            panic!("Data does not fit dimensions, got {d} elements but shape {n} x {m}.");
        }
        Self {
            data,
            shape: (n, m),
        }
    }

    pub fn abs(&self) -> FloatPrecision {
        self.data.iter().map(|&x| x * x).sum::<FloatPrecision>().sqrt()
    }
}

impl fmt::Display for DMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (rows, cols) = self.shape;
        println!("Shape: {:?}", self.shape);
        for i in 0..std::cmp::min(rows, 5) {
            for j in 0..std::cmp::min(cols, 5) {
                if i == 3 && j == 3 && (rows > 5 || cols > 5) {
                    print!("...");
                } else if i == 3 && (rows > 5) {
                    print!("...");
                } else if j == 3 && (cols > 5) {
                    print!("...");
                } else if i == 4 && j == 4 && (rows > 5 || cols > 5) {
                    print!("{:>5}", self.data[(rows - 1) * cols + cols - 1]);
                } else if i == 4 && rows > 5 {
                    print!("{:>5}", self.data[(rows - 1) * cols + j]);
                } else if j == 4 && cols > 5 {
                    print!("{:>5}", self.data[i * cols + cols - 1]);
                } else {
                    print!("{:>5}", self.data[i * cols + j]);
                }
            }
            println!();
        }
        Ok(())
    }
}
