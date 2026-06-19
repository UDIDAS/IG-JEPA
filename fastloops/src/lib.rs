// fastloops: graph-minor pooling on grayscale or RGB images.
//
// Output:
//   adj_array (2H-1, 2W-1) u8 bitflags
//   features  (N, 16) u64, columns:
//     0 AREA  1 SUM_X  2 SUM_Y
//     3 SUM_XX  4 SUM_YY  5 SUM_XY
//     6 SUM_R  7 SUM_G  8 SUM_B           (replicated for grayscale input)
//     9 MIN_X 10 MAX_X 11 MIN_Y 12 MAX_Y
//    13 BOUNDARY_LEN
//    14 CANON_Y 15 CANON_X

use ndarray::{s, Array2, Ix2, Ix3};
use numpy::{PyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use std::mem;

const VISITED:                u8 = 0b0000_0001;
const NODE_MERGED:            u8 = 0b0000_0010;
const NODE_DELETED:           u8 = 0b0000_0100;
const NODE_ADJACENT_BOUNDARY: u8 = 0b0000_1000;
const EDGE_MERGED:            u8 = 0b0001_0000;
const EDGE_DELETED:           u8 = 0b0010_0000;
const IN_CURRENT_REGION:      u8 = 0b0100_0000;

const AREA: usize = 0;
const SUM_X: usize = 1;
const SUM_Y: usize = 2;
const SUM_XX: usize = 3;
const SUM_YY: usize = 4;
const SUM_XY: usize = 5;
const SUM_R: usize = 6;
const SUM_G: usize = 7;
const SUM_B: usize = 8;
const MIN_X: usize = 9;
const MAX_X: usize = 10;
const MIN_Y: usize = 11;
const MAX_Y: usize = 12;
const BOUNDARY_LEN: usize = 13;
const CANON_Y: usize = 14;
const CANON_X: usize = 15;
const N_FEATURES: usize = 16;

const CHUNK: usize = 8192;
const DELTA_R: [isize; 4] = [-2, 2, 0, 0];
const DELTA_C: [isize; 4] = [0, 0, -2, 2];

const WR: u32 = 2126;
const WG: u32 = 7152;
const WB: u32 = 722;

#[inline(always)]
fn luma_u8(rgb: [u8; 3]) -> u8 {
    ((rgb[0] as u32 * WR + rgb[1] as u32 * WG + rgb[2] as u32 * WB) / 10000) as u8
}

#[inline(always)]
fn pixel_distance(a: [u8; 3], b: [u8; 3], method: u8, weighted_luma: bool) -> u16 {
    if weighted_luma {
        let la = luma_u8(a) as i16;
        let lb = luma_u8(b) as i16;
        return (la - lb).unsigned_abs();
    }
    let dr = (a[0] as i16 - b[0] as i16).unsigned_abs();
    let dg = (a[1] as i16 - b[1] as i16).unsigned_abs();
    let db = (a[2] as i16 - b[2] as i16).unsigned_abs();
    match method {
        0 => dr + dg + db,
        1 => {
            let s = (dr as u32) * (dr as u32)
                  + (dg as u32) * (dg as u32)
                  + (db as u32) * (db as u32);
            (s as f64).sqrt().round() as u16
        }
        2 => dr.max(dg).max(db),
        _ => unreachable!(),
    }
}

fn euclidean_gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

fn coprime_step(val: usize, divisor: f64) -> usize {
    if val <= 1 { return 1; }
    let split = (((val as f64) / divisor.max(1.0)).floor() as usize)
        .clamp(1, val.saturating_sub(1));
    for i in split..val { if euclidean_gcd(val, i) == 1 { return i; } }
    for i in (1..split).rev() { if euclidean_gcd(val, i) == 1 { return i; } }
    1
}

#[pyfunction]
#[pyo3(signature = (
    image,
    merge_distance = None,
    cut_distance = None,
    delete_small_node_max_size = None,
    delete_large_node_min_size = None,
    iterator_divisor = None,
    distance_method = None,
    use_weighted_luminance = None,
))]
#[allow(clippy::too_many_arguments)]
fn merge_and_cut<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'_, u8>,
    merge_distance: Option<u16>,
    cut_distance: Option<u16>,
    delete_small_node_max_size: Option<usize>,
    delete_large_node_min_size: Option<usize>,
    iterator_divisor: Option<f64>,
    distance_method: Option<&str>,
    use_weighted_luminance: Option<bool>,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<u64>>)> {

    let view = image.as_array();
    let (n_rows, n_cols, is_color) = match view.ndim() {
        2 => (view.shape()[0], view.shape()[1], false),
        3 => {
            if view.shape()[2] != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "3D image must have exactly 3 channels (H, W, 3)",
                ));
            }
            (view.shape()[0], view.shape()[1], true)
        }
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            "image must be 2D (H, W) u8 or 3D (H, W, 3) u8",
        )),
    };
    if n_rows < 2 || n_cols < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "image must be at least 2x2",
        ));
    }
    let n_pixels = n_rows * n_cols;

    let method_code: u8 = match distance_method.unwrap_or("chebyshev") {
        "manhattan" => 0,
        "euclidean" => 1,
        "chebyshev" => 2,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            "distance_method must be one of: manhattan, euclidean, chebyshev",
        )),
    };
    let weighted_luma = use_weighted_luminance.unwrap_or(false);
    let divisor = iterator_divisor.unwrap_or(4.12).max(1.0);
    let merge_distance = merge_distance.unwrap_or(10);
    let cut_distance = cut_distance.unwrap_or(100);
    let small_default = (n_pixels as f64).log2() as usize;
    let large_default = (n_pixels as f64).powf(0.8) as usize;
    let delete_small = delete_small_node_max_size.unwrap_or(small_default);
    let delete_large = delete_large_node_min_size.unwrap_or(large_default);

    let v2 = if !is_color {
        Some(view.view().into_dimensionality::<Ix2>().unwrap())
    } else { None };
    let v3 = if is_color {
        Some(view.view().into_dimensionality::<Ix3>().unwrap())
    } else { None };

    let get_pixel = move |r: usize, c: usize| -> [u8; 3] {
        if is_color {
            let v = v3.as_ref().unwrap();
            [v[[r, c, 0]], v[[r, c, 1]], v[[r, c, 2]]]
        } else {
            let v = v2.as_ref().unwrap();
            let p = v[[r, c]];
            [p, p, p]
        }
    };

    let row_step = coprime_step(n_rows, divisor);
    let col_step = coprime_step(n_cols, divisor);
    let row_start = row_step % n_rows;
    let col_start = col_step % n_cols;

    let (adj, features) = py.allow_threads(move || {
        let h2 = n_rows * 2 - 1;
        let w2 = n_cols * 2 - 1;
        let mut adj: Array2<u8> = Array2::zeros((h2, w2));
        let mut buf: Array2<u64> = Array2::zeros((CHUNK, N_FEATURES));
        let mut chunks: Vec<Array2<u64>> = Vec::new();
        let mut buf_idx: usize = 0;

        for i in 0..n_rows {
            let r = (row_start + i * row_step) % n_rows;
            let ar = r * 2;
            for j in 0..n_cols {
                let c = (col_start + j * col_step) % n_cols;
                let ac = c * 2;

                if adj[[ar, ac]] & VISITED != 0 { continue; }

                if buf_idx >= CHUNK {
                    chunks.push(mem::replace(&mut buf, Array2::zeros((CHUNK, N_FEATURES))));
                    buf_idx = 0;
                }
                {
                    let mut row = buf.row_mut(buf_idx);
                    for k in 0..N_FEATURES { row[k] = 0; }
                    row[MIN_X] = u64::MAX;
                    row[MIN_Y] = u64::MAX;
                    row[CANON_X] = u64::MAX;
                    row[CANON_Y] = u64::MAX;
                }

                let seed_color = get_pixel(r, c);
                adj[[ar, ac]] |= NODE_MERGED | IN_CURRENT_REGION;
                let mut stack: Vec<(usize, usize)> = vec![(ar, ac)];
                let mut delete_stack: Vec<(usize, usize)> = Vec::new();

                while let Some((ar2, ac2)) = stack.pop() {
                    if adj[[ar2, ac2]] & VISITED != 0 { continue; }
                    adj[[ar2, ac2]] |= VISITED;
                    delete_stack.push((ar2, ac2));

                    let pr = ar2 / 2;
                    let pc = ac2 / 2;
                    let here = get_pixel(pr, pc);
                    let x = pc as u64;
                    let y = pr as u64;

                    {
                        let mut row = buf.row_mut(buf_idx);
                        row[AREA]  += 1;
                        row[SUM_X] += x;
                        row[SUM_Y] += y;
                        row[SUM_XX] += x * x;
                        row[SUM_YY] += y * y;
                        row[SUM_XY] += x * y;
                        row[SUM_R] += here[0] as u64;
                        row[SUM_G] += here[1] as u64;
                        row[SUM_B] += here[2] as u64;
                        if x < row[MIN_X] { row[MIN_X] = x; }
                        if x > row[MAX_X] { row[MAX_X] = x; }
                        if y < row[MIN_Y] { row[MIN_Y] = y; }
                        if y > row[MAX_Y] { row[MAX_Y] = y; }
                        if y < row[CANON_Y] || (y == row[CANON_Y] && x < row[CANON_X]) {
                            row[CANON_Y] = y;
                            row[CANON_X] = x;
                        }
                    }

                    let mut on_boundary = false;
                    for d in 0..4 {
                        let nr_i = ar2 as isize + DELTA_R[d];
                        let nc_i = ac2 as isize + DELTA_C[d];
                        if nr_i < 0 || nc_i < 0
                            || nr_i >= h2 as isize
                            || nc_i >= w2 as isize { continue; }
                        let nar = nr_i as usize;
                        let nac = nc_i as usize;
                        let er = (ar2 as isize + DELTA_R[d] / 2) as usize;
                        let ec = (ac2 as isize + DELTA_C[d] / 2) as usize;
                        let n_flags = adj[[nar, nac]];

                        if n_flags & IN_CURRENT_REGION != 0 {
                            adj[[er, ec]] |= EDGE_MERGED;
                            continue;
                        }
                        if n_flags & (VISITED | NODE_MERGED) != 0 {
                            on_boundary = true;
                            continue;
                        }

                        let neighbor = get_pixel(nar / 2, nac / 2);
                        let diff = pixel_distance(seed_color, neighbor, method_code, weighted_luma);

                        if diff <= merge_distance {
                            adj[[nar, nac]] |= NODE_MERGED | IN_CURRENT_REGION;
                            adj[[er, ec]] |= EDGE_MERGED;
                            stack.push((nar, nac));
                        } else {
                            on_boundary = true;
                            if diff >= cut_distance {
                                adj[[er, ec]] |= EDGE_DELETED;
                            }
                        }
                    }

                    if on_boundary {
                        adj[[ar2, ac2]] |= NODE_ADJACENT_BOUNDARY;
                        buf.row_mut(buf_idx)[BOUNDARY_LEN] += 1;
                    }
                }

                let area = buf.row(buf_idx)[AREA] as usize;
                let pass_size = (area > delete_small && area < delete_large)
                    || (delete_small >= delete_large);
                if pass_size {
                    for (dr, dc) in &delete_stack {
                        adj[[*dr, *dc]] &= !IN_CURRENT_REGION;
                    }
                    buf_idx += 1;
                } else {
                    for (dr, dc) in delete_stack.drain(..) {
                        adj[[dr, dc]] |= NODE_DELETED;
                        adj[[dr, dc]] &= !(NODE_MERGED | IN_CURRENT_REGION);
                    }
                }
            }
        }

        if buf_idx > 0 {
            chunks.push(buf.slice(s![0..buf_idx, ..]).to_owned());
        }
        let total: usize = chunks.iter().map(|c| c.nrows()).sum();
        let mut features: Array2<u64> = Array2::zeros((total, N_FEATURES));
        let mut off = 0;
        for c in &chunks {
            features.slice_mut(s![off..off + c.nrows(), ..]).assign(&c.view());
            off += c.nrows();
        }
        (adj, features)
    });

    Ok((
        PyArray2::from_owned_array_bound(py, adj),
        PyArray2::from_owned_array_bound(py, features),
    ))
}

/// BFS-based subgraph masking for JEPA training.
///
/// Given edge_index (2, E) and num_nodes N, grows a connected subgraph from a
/// random seed until mask_count nodes are selected. Returns:
///   - masked_indices: (mask_count,) i64 — the masked node indices
///   - ctx_src, ctx_dst: (E',) i64 each — edges NOT touching any masked node
#[pyfunction]
fn subgraph_mask<'py>(
    py: Python<'py>,
    src_arr: PyReadonlyArrayDyn<'_, i64>,
    dst_arr: PyReadonlyArrayDyn<'_, i64>,
    num_nodes: usize,
    mask_count: usize,
    seed: usize,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<i64>>,
    Bound<'py, numpy::PyArray1<i64>>,
    Bound<'py, numpy::PyArray1<i64>>,
)> {
    let src = src_arr.as_array();
    let dst = dst_arr.as_array();
    let e = src.len();

    let (masked_vec, ctx_s, ctx_d) = py.allow_threads(move || {
        // Build adjacency list
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for i in 0..e {
            let s = src[i] as usize;
            let d = dst[i] as usize;
            adj[s].push(d);
        }

        // BFS from seed
        let mut is_masked = vec![false; num_nodes];
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        let start = seed % num_nodes;
        queue.push_back(start);
        is_masked[start] = true;
        let mut count = 1usize;

        while count < mask_count {
            let node = match queue.pop_front() {
                Some(n) => n,
                None => break,
            };
            for &nbr in &adj[node] {
                if !is_masked[nbr] {
                    is_masked[nbr] = true;
                    count += 1;
                    queue.push_back(nbr);
                    if count >= mask_count { break; }
                }
            }
        }

        // If disconnected, fill randomly (deterministic sweep from seed)
        if count < mask_count {
            let mut idx = (start + 1) % num_nodes;
            while count < mask_count {
                if !is_masked[idx] {
                    is_masked[idx] = true;
                    count += 1;
                }
                idx = (idx + 1) % num_nodes;
            }
        }

        // Collect masked indices
        let mut masked_vec: Vec<i64> = Vec::with_capacity(mask_count);
        for i in 0..num_nodes {
            if is_masked[i] { masked_vec.push(i as i64); }
        }

        // Filter edges: keep only edges where neither endpoint is masked
        let mut ctx_s: Vec<i64> = Vec::with_capacity(e);
        let mut ctx_d: Vec<i64> = Vec::with_capacity(e);
        for i in 0..e {
            let s = src[i] as usize;
            let d = dst[i] as usize;
            if !is_masked[s] && !is_masked[d] {
                ctx_s.push(s as i64);
                ctx_d.push(d as i64);
            }
        }

        (masked_vec, ctx_s, ctx_d)
    });

    Ok((
        numpy::PyArray1::from_vec_bound(py, masked_vec),
        numpy::PyArray1::from_vec_bound(py, ctx_s),
        numpy::PyArray1::from_vec_bound(py, ctx_d),
    ))
}

#[pymodule]
fn fastloops(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(merge_and_cut, m)?)?;
    m.add_function(wrap_pyfunction!(subgraph_mask, m)?)?;
    Ok(())
}
