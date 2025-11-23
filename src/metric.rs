use ndarray::{Array2, ArrayView2};
use num_traits::Float;
use std::fmt::Debug;

pub trait Metric<T>: Sync + Send {
    fn distance(&self, p1: &[T], p2: &[T]) -> T;
    fn distance_sq(&self, p1: &[T], p2: &[T]) -> T;

    /// Optional upper bound on the metric distance.
    ///
    /// If provided, callers can bypass expensive distance calculations when
    /// the current radius is larger than this bound (e.g., residual
    /// correlation is capped by √2). Default is `None`.
    fn max_distance_hint(&self) -> Option<T> {
        None
    }
}

#[derive(Copy, Clone)]
pub struct Euclidean;

impl<T> Metric<T> for Euclidean
where
    T: Float + Debug + Send + Sync + std::iter::Sum,
{
    fn distance(&self, p1: &[T], p2: &[T]) -> T {
        self.distance_sq(p1, p2).sqrt()
    }

    fn distance_sq(&self, p1: &[T], p2: &[T]) -> T {
        p1.iter()
            .zip(p2.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum()
    }
}

// Residual Metric operates on INDICES into the V-Matrix and Coords
pub struct ResidualMetric<'a, T> {
    pub v_matrix: ArrayView2<'a, T>,
    pub p_diag: &'a [T],
    pub rbf_var: T,
    pub scaled_coords: Array2<T>,
    pub scaled_norms: Vec<T>,
    pub neg_half: T,
}

impl<'a, T> ResidualMetric<'a, T>
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'a,
{
    pub fn new(
        v_matrix: ArrayView2<'a, T>,
        p_diag: &'a [T],
        coords: ArrayView2<'a, T>,
        rbf_var: T,
        rbf_ls: &'a [T],
    ) -> Self {
        let dim = coords.ncols();
        let eps = T::from(1e-6).unwrap();
        let fallback_ls = *rbf_ls.get(0).unwrap_or(&T::one());
        let mut inv_ls: Vec<T> = Vec::with_capacity(dim);
        for i in 0..dim {
            let raw = *rbf_ls.get(i).unwrap_or(&fallback_ls);
            let safe = if raw.abs() < eps { eps } else { raw };
            inv_ls.push(T::one() / safe);
        }

        let mut scaled_coords = coords.as_standard_layout().to_owned();
        for mut row in scaled_coords.outer_iter_mut() {
            for (val, inv) in row.iter_mut().zip(inv_ls.iter()) {
                *val = *val * *inv;
            }
        }

        let mut scaled_norms: Vec<T> = Vec::with_capacity(scaled_coords.nrows());
        for row in scaled_coords.outer_iter() {
            let norm: T = row.iter().map(|v| *v * *v).sum();
            scaled_norms.push(norm);
        }

        let neg_half = T::from(-0.5).unwrap();

        ResidualMetric {
            v_matrix,
            p_diag,
            rbf_var,
            scaled_coords,
            scaled_norms,
            neg_half,
        }
    }

    #[inline(always)]
    pub fn distance_sq_idx(&self, idx_1: usize, idx_2: usize) -> T {
        // Coords dot product (small dimension, typically 2-10)
        let x_view = self.scaled_coords.row(idx_1);
        let y_view = self.scaled_coords.row(idx_2);
        
        // Safety: We rely on the caller to provide valid indices. 
        // In the context of Cover Tree, these are verified at construction or insertion.
        let x = match x_view.as_slice() {
            Some(s) => s,
            None => panic!("Scaled coords must be row-contiguous"),
        };
        let y = match y_view.as_slice() {
            Some(s) => s,
            None => panic!("Scaled coords must be row-contiguous"),
        };

        let mut dot_scaled = T::zero();
        for i in 0..x.len() {
            dot_scaled = dot_scaled + x[i] * y[i];
        }

        let two = T::from(2.0).unwrap();
        let mut d2 = self.scaled_norms[idx_1] + self.scaled_norms[idx_2] - two * dot_scaled;
        if d2 < T::zero() {
            d2 = T::zero();
        }

        let k_val = self.rbf_var * (self.neg_half * d2).exp();

        // V-Matrix dot product (hot loop)
        let v1_view = self.v_matrix.row(idx_1);
        let v2_view = self.v_matrix.row(idx_2);
        let len = v1_view.len();
        let dot = dot_product_unrolled(v1_view, v2_view, len);

        let denom = (self.p_diag[idx_1] * self.p_diag[idx_2]).sqrt();
        let eps = T::from(1e-9).unwrap();

        if denom < eps {
            return T::one();
        }

        let rho = (k_val - dot) / denom;
        let one = T::one();
        let neg_one = -one;
        let rho_clamped = rho.max(neg_one).min(one);
        one - rho_clamped.abs()
    }

    pub fn distances_sq_batch_idx(&self, q_idx: usize, p_indices: &[usize]) -> Vec<T> {
        let mut results = Vec::with_capacity(p_indices.len());
        self.distances_sq_batch_idx_into(q_idx, p_indices, &mut results);
        results
    }

    pub fn distances_sq_batch_idx_into(&self, q_idx: usize, p_indices: &[usize], out: &mut Vec<T>) {
        out.clear();
        out.reserve(p_indices.len());
        // Pre-fetch query data
        let x_view = self.scaled_coords.row(q_idx);
        let x = x_view.as_slice().unwrap_or_else(|| panic!("Query coords must be contiguous"));
        let q_norm = self.scaled_norms[q_idx];
        let v1_view = self.v_matrix.row(q_idx);
        let v1_len = v1_view.len();
        let q_diag = self.p_diag[q_idx];
        let two = T::from(2.0).unwrap();
        let one = T::one();
        let neg_one = -one;
        let eps = T::from(1e-9).unwrap();

        for &idx_2 in p_indices {
            // 1. Coords Part
            let y_view = self.scaled_coords.row(idx_2);
            let y = y_view.as_slice().unwrap(); // Assume contiguous from construction
            
            let mut dot_scaled = T::zero();
            for i in 0..x.len() {
                dot_scaled = dot_scaled + x[i] * y[i];
            }
            
            let mut d2 = q_norm + self.scaled_norms[idx_2] - two * dot_scaled;
            if d2 < T::zero() { d2 = T::zero(); }
            
            let k_val = self.rbf_var * (self.neg_half * d2).exp();

            // 2. V-Matrix Part
            let v2_view = self.v_matrix.row(idx_2);
            let dot = dot_product_unrolled(v1_view, v2_view, v1_len);

            let denom = (q_diag * self.p_diag[idx_2]).sqrt();
            
            if denom < eps {
                out.push(T::one());
            } else {
                let rho = (k_val - dot) / denom;
                let rho_clamped = rho.max(neg_one).min(one);
                out.push(one - rho_clamped.abs());
            }
        }
    }

    pub fn distance_idx(&self, idx_1: usize, idx_2: usize) -> T {
        self.distance_sq_idx(idx_1, idx_2).sqrt()
    }
}

impl<'a, T> Metric<T> for ResidualMetric<'a, T>
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'a,
{
    fn distance(&self, p1: &[T], p2: &[T]) -> T {
        // Assume points are 1D arrays containing a single value which is the index
        let idx1 = p1[0].to_usize().unwrap();
        let idx2 = p2[0].to_usize().unwrap();
        self.distance_idx(idx1, idx2)
    }

    fn distance_sq(&self, p1: &[T], p2: &[T]) -> T {
        let idx1 = p1[0].to_usize().unwrap();
        let idx2 = p2[0].to_usize().unwrap();
        self.distance_sq_idx(idx1, idx2)
    }

    fn max_distance_hint(&self) -> Option<T> {
        // Residual correlation distance is bounded by sqrt(2).
        Some(T::from(2.0).unwrap().sqrt())
    }
}

#[inline(always)]
fn dot_product_unrolled<T>(a: ndarray::ArrayView1<T>, b: ndarray::ArrayView1<T>, len: usize) -> T
where
    T: Float + Debug + Send + Sync + std::iter::Sum,
{
    let mut acc0 = T::zero();
    let mut acc1 = T::zero();
    let mut acc2 = T::zero();
    let mut acc3 = T::zero();

    let chunks = len / 4;
    let rem = len % 4;

    for i in 0..chunks {
        let base = i * 4;
        unsafe {
            acc0 = acc0 + *a.uget(base) * *b.uget(base);
            acc1 = acc1 + *a.uget(base + 1) * *b.uget(base + 1);
            acc2 = acc2 + *a.uget(base + 2) * *b.uget(base + 2);
            acc3 = acc3 + *a.uget(base + 3) * *b.uget(base + 3);
        }
    }

    let mut tail = T::zero();
    for i in (len - rem)..len {
        unsafe {
            tail = tail + *a.uget(i) * *b.uget(i);
        }
    }

    acc0 + acc1 + acc2 + acc3 + tail
}
