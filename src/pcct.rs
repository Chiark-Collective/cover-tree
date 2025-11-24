use ndarray::ArrayView2;

/// Morton (Z-order) code for 3D points scaled to 21 bits per axis (fits in u64).
/// This is a lightweight stand-in for the Hilbert ordering used by the Python PCCT path.
fn morton3(x: u32, y: u32, z: u32) -> u64 {
    interleave3(x as u64, y as u64, z as u64)
}

#[inline(always)]
fn part1by2(mut n: u64) -> u64 {
    n &= 0x1f_ffff; // 21 bits
    n = (n | (n << 32)) & 0x1f00_0000_00ff_ffff;
    n = (n | (n << 16)) & 0x1f00_00ff_0000_ff00;
    n = (n | (n << 8)) & 0x100f_00f0_0f00_f00f;
    n = (n | (n << 4)) & 0x10c3_0c30_c30c_30c3;
    n = (n | (n << 2)) & 0x1249_2492_4924_9249;
    n
}

#[inline(always)]
fn interleave3(x: u64, y: u64, z: u64) -> u64 {
    part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)
}

/// Compute a Morton-ordered permutation of point indices (3D only). Falls back to
/// natural order for other dimensions.
pub fn hilbert_like_order(coords: ArrayView2<'_, f32>) -> Vec<usize> {
    let n = coords.nrows();
    let dim = coords.ncols();
    if n == 0 {
        return Vec::new();
    }
    if dim != 3 {
        return (0..n).collect();
    }

    let mut mins = [f32::MAX; 3];
    let mut maxs = [f32::MIN; 3];
    for row in coords.outer_iter() {
        for i in 0..3 {
            let v = row[i];
            if v < mins[i] {
                mins[i] = v;
            }
            if v > maxs[i] {
                maxs[i] = v;
            }
        }
    }
    let eps = 1e-9f32;
    let scale = |v: f32, idx: usize| -> u32 {
        let denom = (maxs[idx] - mins[idx]).abs().max(eps);
        let norm = ((v - mins[idx]) / denom).clamp(0.0, 1.0);
        (norm * ((1u32 << 21) as f32 - 1.0)) as u32
    };

    let mut codes: Vec<(u64, usize)> = Vec::with_capacity(n);
    for (i, row) in coords.outer_iter().enumerate() {
        let x = scale(row[0], 0);
        let y = scale(row[1], 1);
        let z = scale(row[2], 2);
        let code = morton3(x, y, z);
        codes.push((code, i));
    }
    codes.sort_by_key(|&(c, _)| c);
    codes.into_iter().map(|(_, i)| i).collect()
}
