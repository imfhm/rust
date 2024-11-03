use std::fmt;
use std::ops;
use std::ops::Add;

use crate::constants::FloatPrecision;

// A statically sized matrix implementation.
// Chose 1D-data storage for increased cache performance.
// Copy is implementable as properties are of known size.
#[derive(Clone, Copy)]
pub struct SMatrix<'a, const RC: usize> {
    pub data: &'a [FloatPrecision; RC],
    pub shape: (usize, usize),
}

impl<'a, const RC: usize> SMatrix<'a, RC> {
    pub fn new(data: &'a [FloatPrecision; RC], shape: (usize, usize)) -> Self {
        let (n, m) = shape;
        if RC != n * m {
            panic!("Data does not fit dimensions, got {RC} elements but shape is ({n}, {m}).");
        }

        Self { data, shape }
    }
}

impl<'a, const RC: usize> Add for SMatrix<'a, RC> {
    type Output = [FloatPrecision; RC];

    fn add(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            panic!(
                "For addition, matrices must be of same shape, got {:?} and {:?}.",
                self.shape, rhs.shape
            );
        }
        let (n, m) = self.shape;
        let mut data = [0.; RC];
        for i in 0..n {
            for j in 0..m {
                let index = i * m + j;
                data[index] = self.data[index] + rhs.data[index];
            }
        }
        data
    }
}

// Maybe in the future swap to unsafe operations
// Speed may approximately double

#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<FloatPrecision>,
    pub shape: (usize, usize),
}

impl Matrix {
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

    pub fn zeros(n: usize, m: usize) -> Self {
        Matrix {
            data: vec![0.0; n * m],
            shape: (n, m),
        }
    }
    pub fn get(&self, i: usize, j: usize) -> FloatPrecision {
        self.data[i * self.shape.1 + j]
    }
    pub fn set(&mut self, i: usize, j: usize, value: FloatPrecision) {
        self.data[i * self.shape.1 + j] = value;
    }

    pub fn as_column(&self, i: usize) -> FloatPrecision {
        self.data[i * self.shape.1]
    }

    pub fn as_row(&self, i: usize) -> FloatPrecision {
        self.data[i]
    }
    pub fn get_row(&self, i: usize) -> Vec<FloatPrecision> {
        self.data[i * self.shape.1..(i + 1) * self.shape.1].to_vec()
    }
}

pub fn scale(scalar: FloatPrecision, rhs: Matrix) -> Matrix {
    Matrix {
        data: rhs.data.iter().map(|x| scalar * x).collect(),
        shape: rhs.shape,
    }
}

pub fn add(lhs: Matrix, rhs: Matrix) -> Matrix {
    let (n0, m0) = lhs.shape;
    let (n, m) = rhs.shape;
    if n != n0 || m != m0 {
        panic!("Matrices must be of same shape, got ({n0},{m0}) and ({n}, {m}).");
    }

    let mut sum = Matrix {
        data: vec![0.0; n0 * m0],
        shape: (n0, m0),
    };
    for i in 0..n0 {
        for j in 0..m0 {
            sum.set(i, j, lhs.get(i, j) + rhs.get(i, j));
        }
    }
    sum
}

pub fn sub(lhs: Matrix, rhs: Matrix) -> Matrix {
    let (n0, m0) = lhs.shape;
    let (n, m) = rhs.shape;
    if n != n0 || m != m0 {
        panic!("Matrices must be of same shape, got ({n0},{m0}) and ({n}, {m}).");
    }

    let mut diff = Matrix {
        data: vec![0.0; n0 * m0],
        shape: (n0, m0),
    };
    for i in 0..n0 {
        for j in 0..m0 {
            diff.set(i, j, lhs.get(i, j) - rhs.get(i, j));
        }
    }
    diff
}

pub fn dot(lhs: Matrix, rhs: Matrix) -> Matrix {
    let (n0, m0) = lhs.shape;
    let (n, m) = rhs.shape;
    if m0 != n {
        panic!("Matrices must be of compatible shape, got ({n0},{m0}) and ({n}, {m}).");
    }

    let mut data = Matrix {
        data: vec![0.0; m * n0],
        shape: (n0, m),
    };
    for i in 0..n0 {
        for j in 0..m {
            let mut d = 0.0;
            for k in 0..m0 {
                d += lhs.get(i, k) * rhs.get(k, j);
            }
            data.set(i, j, d);
        }
    }
    data
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{:p}", self.data.as_ptr());
        for i in 0..self.shape.0 {
            // Join the elements of each row with a space and print the row
            let row_str: Vec<String> = self
                .get_row(i)
                .iter()
                .map(|value| value.to_string())
                .collect();
            writeln!(f, "{}", row_str.join(" "))?; // Add a newline after each row
        }
        Ok(())
    }
}
