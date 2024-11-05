use std::fmt;
use std::ops;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;

use crate::constants::FloatPrecision;

pub fn argmax<T: PartialOrd>(slice: &[T]) -> Option<usize> {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
}

pub fn smul(s: FloatPrecision, rhs: &DMatrix, result: &mut DMatrix) {
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
            for k in 0..K {
                result.data[rowr + j] += lhs.data[rowlhs + k] * rhs.data[k * m + j];
            }
        }
    }
}

pub fn tmulm(lhs: &DMatrix, rhs: &DMatrix, result: &mut DMatrix) { // e.g. (10, 32), (10, 1)
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

pub fn smulmt(s: FloatPrecision, lhs: &DMatrix, rhs: &DMatrix, result: &mut DMatrix) {
    let n = lhs.shape.0;
    let K = rhs.shape.1; // number of rows of transpose
    let m = rhs.shape.0; // number of columns of transpose

    for i in 0..n {
        let rowr = i * m;
        let rowlhs = i * K;
        for j in 0..m {
            let index = rowr + j;
            result.data[index] = 0.;
            for k in 0..K {
                result.data[index] += lhs.data[rowlhs + k] * rhs.data[j * K + k];
            }
            result.data[index] *= s;
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
            result.data[index] = lhs.data[index] - rhs.data[index];
        }
    }
}

// A statically sized self.data implementation.
// Chose 1D-data storage for increased cache performance.
// Copy is implementable as properties are of known size.
#[derive(Clone, Copy)]
pub struct SMatrix<const RC: usize> {
    pub data: [FloatPrecision; RC],
    pub shape: (usize, usize),
}

impl<const RC: usize> SMatrix<RC> {
    pub fn new(data: [FloatPrecision; RC], shape: (usize, usize)) -> Self {
        let (n, m) = shape;
        if RC != n * m {
            panic!("Data does not fit dimensions, got {RC} elements but shape is ({n}, {m}).");
        }

        Self { data, shape }
    }
}

// Passing pointers avoids copying the self.data data into the add function.
impl<const RC: usize> Add for &SMatrix<RC> {
    type Output = SMatrix<RC>;

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
        // Here the sum is returned by copying the data, which seems slow considering
        // the sum may be a large self.data. But as the function call opens up a new block
        // on the stack (which is dropped at the end of the function) we can't return
        // a pointer to a variable inside the function. Only alternative would be to
        // allocate memory on the heap and return a pointer to that, but this is slow.
        SMatrix {
            data,
            shape: (n, m),
        }
    }
}

impl<const RC: usize> fmt::Display for SMatrix<RC> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.shape.0 {
            let l = self.shape.1;
            let row_str: Vec<String> = self.data[i * l..(i + 1) * l]
                .iter()
                .map(|value| value.to_string())
                .collect();
            writeln!(f, "{}", row_str.join(" "))?; // Add a newline after each row
        }
        Ok(())
    }
}

// Maybe in the future swap to unsafe operations
// Speed may approximately double

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

    pub fn mul_elementwise(&self, rhs: &DMatrix) -> DMatrix {
        if self.shape != rhs.shape {
            panic!("For elementwise multiplication, matrices must be of same shape, got {:?} and {:?}.", self.shape, rhs.shape);
        }
        let (n, m) = self.shape;
        let mut data = vec![0.; n * m];
        for i in 0..n {
            for j in 0..m {
                let index = i * m + j;
                data[index] = self.data[index] * rhs.data[index];
            }
        }
        DMatrix {
            data,
            shape: (n, m),
        }
    }

    pub fn T(&self) -> DMatrix {
        let (n, m) = self.shape;
        let mut data = vec![0.; m * n];
        for i in 0..n {
            for j in 0..m {
                data[j * n + i] = self.data[i * m + j];
            }
        }
        DMatrix {
            data,
            shape: (m, n),
        }
    }

    pub fn get_column(&self, col: usize) -> DMatrix {
        let (n, m) = self.shape;
        let mut data = vec![0.; n];
        for i in 0..n {
            data[i] = self.data[col + i * m];
        }
        DMatrix {
            data,
            shape: (n, m),
        }
    }

    pub fn dot(&self, rhs: &DMatrix) -> FloatPrecision {
        if self.shape.0 != 1 || rhs.shape.1 != 1 {
            panic!(
                "For dot product, both matrices need to be one dimensional, got {:?} and {:?}.",
                self.shape, rhs.shape
            );
        }
        if self.shape.1 != rhs.shape.0 {
            panic!(
                "For dot product, matrices need to be of compatible shape, got {:?} and {:?}.",
                self.shape, rhs.shape
            );
        }
        let mut sum = 0.;
        for i in 0..self.shape.1 {
            sum += self.data[i] * rhs.data[i];
        }
        sum
    }

    pub fn mul_diag(&self, rhs: &DMatrix) -> DMatrix {
        if rhs.shape.1 != 1 {
            panic!("For multiplying with a diagonal self.data from the right, use a column vector, got shape {:?}", rhs.shape);
        }
        if self.shape.1 != rhs.shape.0 {
            panic!("For multiplying with a diagonal self.data from the right, matrices must be of compatible shape, got {:?} and {:?}.", self.shape, rhs.shape);
        }
        let (n, m) = self.shape;
        let mut data = vec![0.; n * m];
        for i in 0..n {
            for j in 0..m {
                let index = i * m + j;
                data[index] = rhs.data[j] * self.data[index]
            }
        }
        DMatrix {
            data,
            shape: self.shape,
        }
    }
    pub fn diag_mul(&self, rhs: &DMatrix) -> DMatrix {
        if self.shape.1 != 1 {
            panic!("For multiplying with a diagonal self.data from the left, use a column vector, got shape {:?}", self.shape);
        }
        if self.shape.0 != rhs.shape.0 {
            panic!("For multiplying with a diagonal self.data from the right, matrices must be of compatible shape, got {:?} and {:?}.", self.shape, rhs.shape);
        }
        let (n, m) = rhs.shape;
        let mut data = vec![0.; n * m];
        for i in 0..n {
            for j in 0..m {
                let index = i * m + j;
                data[index] = self.data[i] * rhs.data[index]
            }
        }
        DMatrix {
            data,
            shape: rhs.shape,
        }
    }

    pub fn abs(&self) -> FloatPrecision {
        let sum: FloatPrecision = self.data.iter().map(|&x| x * x).sum();
        sum.sqrt()
    }
}

impl Add for &DMatrix {
    type Output = DMatrix;

    fn add(self, rhs: &DMatrix) -> DMatrix {
        if self.shape != rhs.shape {
            panic!(
                "For addition, matrices must be of same shape, got {:?} and {:?}.",
                self.shape, rhs.shape
            );
        }

        let (n0, m0) = self.shape;
        let (n, m) = rhs.shape;
        let mut sum = DMatrix {
            data: vec![0.0; n0 * m0],
            shape: (n0, m0),
        };
        for i in 0..n0 {
            for j in 0..m0 {
                let index = i * m + j;
                sum.data[index] = self.data[index] + rhs.data[index];
            }
        }
        sum
    }
}

impl Sub for &DMatrix {
    type Output = DMatrix;

    fn sub(self, rhs: &DMatrix) -> DMatrix {
        if self.shape != rhs.shape {
            panic!(
                "For subtraction, matrices must be of same shape, got {:?} and {:?}.",
                self.shape, rhs.shape
            );
        }

        let (n0, m0) = self.shape;
        let mut diff = vec![0.0; n0 * m0];
        for i in 0..n0 {
            for j in 0..m0 {
                let index = i * m0 + j;
                diff[index] = self.data[index] - rhs.data[index];
            }
        }
        DMatrix {
            data: diff,
            shape: self.shape,
        }
    }
}

impl Mul for &DMatrix {
    type Output = DMatrix;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.shape.1 != rhs.shape.0 {
            panic!(
                "For multiplication, matrices must be of compatible shape, got {:?} and {:?}.",
                self.shape, rhs.shape
            );
        }

        let (n0, m0) = self.shape;
        let (n, m) = rhs.shape;
        let mut dot = DMatrix {
            data: vec![0.0; n0 * m],
            shape: (n0, m),
        };
        for i in 0..n0 {
            for j in 0..m {
                let index = i * m + j;
                for k in 0..m0 {
                    dot.data[index] += self.data[i * m0 + k] * rhs.data[k * m + j];
                }
            }
        }
        dot
    }
}

impl Mul<DMatrix> for FloatPrecision {
    type Output = DMatrix;

    fn mul(self, rhs: DMatrix) -> Self::Output {
        DMatrix {
            data: rhs.data.iter().map(|x| self * x).collect(),
            shape: rhs.shape,
        }
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
                    print!("{:>5}", self.data[i  *cols + cols - 1]);
                } else {
                    print!("{:>5}", self.data[i * cols + j]);
                }
            }
            println!();
        }
        Ok(())
    }
}
