use super::types::Conv1d;

pub(super) fn conv1d_same_reflect(x: &[f32], len: usize, conv: &Conv1d) -> Vec<f32> {
    let pad = ((conv.kernel - 1) * conv.dilation) / 2;
    let mut y = vec![0.0; conv.out_ch * len];
    for oc in 0..conv.out_ch {
        for t in 0..len {
            let mut sum = conv.bias[oc];
            for ic in 0..conv.in_ch {
                for kk in 0..conv.kernel {
                    let src = reflect_index(
                        t as isize + (kk * conv.dilation) as isize - pad as isize,
                        len,
                    );
                    let wi = (oc * conv.in_ch + ic) * conv.kernel + kk;
                    sum += conv.weight[wi] * x[ic * len + src];
                }
            }
            y[oc * len + t] = sum;
        }
    }
    y
}

pub(super) fn relu(mut x: Vec<f32>) -> Vec<f32> {
    for value in &mut x {
        *value = value.max(0.0);
    }
    x
}

fn reflect_index(mut i: isize, n: usize) -> usize {
    let n = n as isize;
    while i < 0 || i >= n {
        if i < 0 {
            i = -i;
        }
        if i >= n {
            i = 2 * n - 2 - i;
        }
    }
    i as usize
}
