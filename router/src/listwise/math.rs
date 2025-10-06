//! Vector math utilities for listwise reranking
//!
//! Pure functions for cosine similarity, normalization, and weighted averaging.

use anyhow::{anyhow, Result};

/// Compute cosine similarity between two vectors
///
/// Formula: cos(a, b) = (a · b) / (||a||_2 * ||b||_2)
///
/// Note: This function performs L2 normalization internally before computing the dot product.
/// Backend projector outputs are intentionally unnormalized - normalization happens here.
/// This matches modeling.py where normalize() is called inside compute_scores().
///
/// # Arguments
/// * `a` - First vector (will be normalized internally)
/// * `b` - Second vector (will be normalized internally, must be same length as `a`)
///
/// # Returns
/// Cosine similarity in range [-1, 1]
///
/// # Errors
/// Returns error if vectors have different lengths
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(anyhow!(
            "Vector length mismatch: {} vs {}",
            a.len(),
            b.len()
        ));
    }

    if a.is_empty() {
        return Err(anyhow!("Cannot compute cosine of empty vectors"));
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    const EPS: f32 = 1e-8;
    let norm_a = norm_a + EPS;
    let norm_b = norm_b + EPS;

    let similarity = dot_product / (norm_a * norm_b);

    // Clamp to valid range (numerical stability)
    Ok(similarity.clamp(-1.0, 1.0))
}

/// Normalize a vector in-place with L2 norm
///
/// Formula: x := x / (||x||_2 + eps)
///
/// # Arguments
/// * `vec` - Vector to normalize (modified in-place)
///
/// # Returns
/// The original L2 norm of the vector
pub fn normalize(vec: &mut [f32]) -> f32 {
    const EPS: f32 = 1e-8;

    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_with_eps = norm + EPS;

    for x in vec.iter_mut() {
        *x /= norm_with_eps;
    }

    norm
}

/// Normalize a vector to a new vector (non-mutating)
pub fn normalize_new(vec: &[f32]) -> Vec<f32> {
    let mut result = vec.to_vec();
    normalize(&mut result);
    result
}

/// Compute weighted average of vectors
///
/// Formula: result = Σ(weight_i * vec_i) / Σ(weight_i)
///
/// # Arguments
/// * `vectors` - Slice of vectors (all must have same length)
/// * `weights` - Weight for each vector (length must equal vectors.len())
///
/// # Returns
/// Weighted average vector
///
/// # Errors
/// Returns error if:
/// - `vectors` is empty
/// - `weights.len() != vectors.len()`
/// - Vectors have inconsistent lengths
/// - Sum of weights is too small (< 1e-8)
pub fn weighted_average(vectors: &[Vec<f32>], weights: &[f32]) -> Result<Vec<f32>> {
    if vectors.is_empty() {
        return Err(anyhow!(
            "Cannot compute weighted average of empty vector set"
        ));
    }

    if vectors.len() != weights.len() {
        return Err(anyhow!(
            "Mismatch: {} vectors but {} weights",
            vectors.len(),
            weights.len()
        ));
    }

    let dim = vectors[0].len();
    if dim == 0 {
        return Err(anyhow!("Vectors must have non-zero dimension"));
    }

    // Verify all vectors have same dimension
    for (i, vec) in vectors.iter().enumerate() {
        if vec.len() != dim {
            return Err(anyhow!(
                "Vector {} has length {}, expected {}",
                i,
                vec.len(),
                dim
            ));
        }
    }

    // Compute weighted sum
    let mut result = vec![0.0f32; dim];
    for (vec, &weight) in vectors.iter().zip(weights.iter()) {
        for (r, &v) in result.iter_mut().zip(vec.iter()) {
            *r += weight * v;
        }
    }

    // Normalize by weight sum
    let weight_sum: f32 = weights.iter().sum();
    const EPS: f32 = 1e-8;
    if weight_sum < EPS {
        return Err(anyhow!("Sum of weights too small: {}", weight_sum));
    }

    for r in result.iter_mut() {
        *r /= weight_sum;
    }

    Ok(result)
}

/// Scaled vector addition: a := a + scale * b
///
/// # Arguments
/// * `a` - Target vector (modified in-place)
/// * `b` - Source vector
/// * `scale` - Scaling factor
///
/// # Errors
/// Returns error if vectors have different lengths
pub fn add_scaled(a: &mut [f32], b: &[f32], scale: f32) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!(
            "Vector length mismatch: {} vs {}",
            a.len(),
            b.len()
        ));
    }

    for (a_i, &b_i) in a.iter_mut().zip(b.iter()) {
        *a_i += scale * b_i;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_parallel() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0]; // Parallel to a
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_antiparallel() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut vec = vec![3.0, 4.0];
        let norm = normalize(&mut vec);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((vec[0] - 0.6).abs() < 1e-6);
        assert!((vec[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_average() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let weights = vec![0.3, 0.7];
        let result = weighted_average(&vectors, &weights).unwrap();
        assert!((result[0] - 0.3).abs() < 1e-6);
        assert!((result[1] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_average_equal_weights() {
        let vectors = vec![vec![2.0, 4.0], vec![4.0, 6.0]];
        let weights = vec![1.0, 1.0];
        let result = weighted_average(&vectors, &weights).unwrap();
        // Average: (2+4)/2=3, (4+6)/2=5
        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_scaled() {
        let mut a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        add_scaled(&mut a, &b, 0.5).unwrap();
        // a + 0.5*b = [1+1.5, 2+2] = [2.5, 4.0]
        assert!((a[0] - 2.5).abs() < 1e-6);
        assert!((a[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(cosine_similarity(&a, &b).is_err());
    }

    #[test]
    fn test_weighted_average_length_mismatch() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let weights = vec![1.0]; // Wrong length
        assert!(weighted_average(&vectors, &weights).is_err());
    }

    #[test]
    fn test_cosine_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(cosine_similarity(&a, &b).is_err());
    }

    #[test]
    fn test_weighted_average_empty_vectors() {
        let vectors: Vec<Vec<f32>> = vec![];
        let weights: Vec<f32> = vec![];
        assert!(weighted_average(&vectors, &weights).is_err());
    }

    #[test]
    fn test_weighted_average_zero_weights() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let weights = vec![0.0, 0.0];
        assert!(weighted_average(&vectors, &weights).is_err());
    }
}
