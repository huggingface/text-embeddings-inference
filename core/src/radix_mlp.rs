// SPDX-License-Identifier: MIT
// Published under RadixMLP by Michael Feil
// Copyright (c) 2025 michaelfeil

/// Computes indices for RadixMLP-style folding and scattering to enable prefix-based computation sharing.
///
/// This function identifies shared prefixes among sequences in a batch. For a batch of token
/// sequences, it produces a "compacted" representation containing only the unique subsequences
/// encountered. It also generates index maps to "scatter" (unfold) results from the compact
/// representation back to the original batch structure and to "gather" (fold) the original
/// inputs into the compact form.
///
/// The core idea is to build a prefix tree (trie) over the sequences, where each node represents
/// a unique `(token_id, position_id)` pair in a specific path. This allows deduplication of
/// identical sub-sequences across the batch.
///
/// # Arguments
///
/// * `input_ids`: A flattened vector of token IDs for all sequences in the batch.
/// * `position_ids`: A flattened vector of position IDs corresponding to each token in `input_ids`.
/// * `cu_seq_lengths`: Cumulative sequence lengths, e.g., `[0, len_seq1, len_seq1 + len_seq2, ...]`.
///   This defines the boundaries of each sequence in the flattened `input_ids` and `position_ids`.
/// * `pad_multiple_of`: If `true`, the output compact vectors are padded to a multiple of 8 (for
///   small tensors) or 64 (for larger ones) to improve performance on certain hardware (e.g., cuBLAS).
///
/// # Returns
///
/// A tuple containing four vectors:
///
/// 1. `compact_input_ids`: A vector of the unique token IDs, representing the compacted data.
///    Each unique prefix path from the input sequences appears only once.
/// 2. `compact_position_ids`: The corresponding position IDs for `compact_input_ids`.
/// 3. `scatter_indices`: An index map to unfold data from the compact space to the original
///    batch space. It has the same length as the original `input_ids`.
///    `unfolded[i] = compact[scatter_indices[i]]`.
/// 4. `fold_gather`: An index map to gather data from the original batch space to the compact
///    space. It has the same length as the `compact_input_ids`. Each index points to the
///    *first occurrence* of that unique `(token, position)` pair in the original `input_ids`.
///    `compact[j] = original[fold_gather[j]]`.
pub fn compute_fold_and_scatter(
    input_ids: &[u32],
    position_ids: &[u32],
    cu_seq_lengths: &[u32],
    pad_multiple_of: bool,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    // Empty fast-path
    if input_ids.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    // Single-sequence fast-path: identity
    if cu_seq_lengths.len() == 2 {
        let mut compact_input_ids = input_ids.to_vec();
        let mut compact_position_ids = position_ids.to_vec();
        let mut fold_gather: Vec<u32> = (0..input_ids.len() as u32).collect();
        let scatter_indices = fold_gather.clone();

        if pad_multiple_of {
            pad_to_multiple(
                &mut compact_input_ids,
                &mut compact_position_ids,
                &mut fold_gather,
            );
        }

        return (
            compact_input_ids,
            compact_position_ids,
            scatter_indices,
            fold_gather,
        );
    }

    #[inline]
    fn make_key(token: u32, pos: u32) -> u64 {
        ((pos as u64) << 32) | (token as u64)
    }

    // Pad to a multiple of 8 or 64 for performance if requested.
    #[inline]
    fn pad_to_multiple(
        compact_input_ids: &mut Vec<u32>,
        compact_position_ids: &mut Vec<u32>,
        fold_gather: &mut Vec<u32>,
    ) {
        let current_len = compact_input_ids.len();
        if current_len == 0 {
            return;
        }

        let multiple = if current_len < 1024 { 8 } else { 64 };
        let remainder = current_len % multiple;

        if remainder != 0 {
            let padding_needed = multiple - remainder;
            compact_input_ids.reserve(padding_needed);
            compact_position_ids.reserve(padding_needed);
            fold_gather.reserve(padding_needed);
            for _ in 0..padding_needed {
                compact_input_ids.push(0); // Pad with token 0
                compact_position_ids.push(0); // Pad with position 0
                fold_gather.push(0); // Pad with index 0
            }
        }
    }

    #[derive(Debug)]
    struct Node {
        compact: u32,                // u32::MAX => not assigned yet
        children: Vec<(u64, usize)>, // sorted by key
    }

    let n = input_ids.len();

    // Arena of nodes; index 0 is a synthetic root.
    let mut nodes: Vec<Node> = Vec::with_capacity(n + 1);
    nodes.push(Node {
        compact: u32::MAX,
        children: Vec::new(),
    });

    // Outputs (pre-reserve generously to avoid reallocs)
    let mut compact_input_ids: Vec<u32> = Vec::with_capacity(n);
    let mut compact_position_ids: Vec<u32> = Vec::with_capacity(n);
    let mut fold_gather: Vec<u32> = Vec::with_capacity(n);
    let mut scatter_indices: Vec<u32> = Vec::with_capacity(n);

    let mut next_compact: u32 = 0;

    // -------- Single pass: build trie + produce all mappings --------
    for s in 0..cu_seq_lengths.len().saturating_sub(1) {
        let start = cu_seq_lengths[s] as usize;
        let end = cu_seq_lengths[s + 1] as usize;

        let mut parent = 0usize; // start from root
        for i in start..end {
            let t = input_ids[i];
            let p = position_ids[i];
            let k = make_key(t, p);

            // immutable lookup to find child or insertion point
            let (exists, val) = {
                let children = &nodes[parent].children;
                match children.binary_search_by_key(&k, |&(key, _)| key) {
                    Ok(pos) => (true, children[pos].1),
                    Err(pos) => (false, pos),
                }
            };

            let child_idx = if exists {
                val
            } else {
                // create new node
                let insert_pos = val;
                let idx = nodes.len();
                nodes.push(Node {
                    compact: next_compact, // assign compact immediately
                    children: Vec::new(),
                });
                // insert into parent's sorted children
                nodes[parent].children.insert(insert_pos, (k, idx));

                // record compact stream + first occurrence position
                compact_input_ids.push(t);
                compact_position_ids.push(p);
                fold_gather.push(i as u32);

                next_compact += 1;
                idx
            };

            // scatter: original position -> compact index
            scatter_indices.push(nodes[child_idx].compact);

            parent = child_idx;
        }
    }

    // If no reduction happened, the streams equal identity (creation order == input order).
    // That already satisfies your tests, so just return what we built.

    // Pad to a multiple of 8 for cublas performance if requested.
    if pad_multiple_of {
        pad_to_multiple(
            &mut compact_input_ids,
            &mut compact_position_ids,
            &mut fold_gather,
        );
    }

    (
        compact_input_ids,
        compact_position_ids,
        scatter_indices,
        fold_gather,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_fold_and_scatter_empty() {
        let input_ids: Vec<u32> = vec![];
        let position_ids: Vec<u32> = vec![];
        let cu_seq_lengths: Vec<u32> = vec![];

        let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) =
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths, false);

        assert_eq!(compact_input_ids, vec![] as Vec<u32>);
        assert_eq!(compact_position_ids, vec![] as Vec<u32>);
        assert_eq!(scatter_indices, vec![] as Vec<u32>);
        assert_eq!(fold_gather, vec![] as Vec<u32>);
    }

    #[test]
    fn test_compute_fold_and_scatter_single_sequence() {
        // Single sequence: [a, b, c]
        let input_ids = vec![1, 2, 3];
        let position_ids = vec![0, 1, 2];
        let cu_seq_lengths = vec![0, 3];

        let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) =
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths, false);

        // No deduplication possible with single sequence
        assert_eq!(compact_input_ids, vec![1, 2, 3]);
        assert_eq!(compact_position_ids, vec![0, 1, 2]);
        assert_eq!(scatter_indices, vec![0, 1, 2]);
        assert_eq!(fold_gather, vec![0, 1, 2]);
    }

    #[test]
    fn test_compute_fold_and_scatter_example_from_comments() {
        // Example from comments:
        // tokens    = [a,b,c,d,e,f,g, a,b,c, e,f,g,h,i]
        // pos       = [0,1,2,3,4,5,6, 0,1,2, 3,4,5,6,7]
        // cu_seqlen = [0,7,10,15]
        // Expected folded:
        // tokens    = [a,b,c, d,e,f,g, e,f,g,h,i]
        // pos       = [0,1,2, 3,4,5,6, 3,4,5,6,7]

        let input_ids = vec![1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 5, 6, 7, 8, 9];
        let position_ids = vec![0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7];
        let cu_seq_lengths = vec![0, 7, 10, 15];

        let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) =
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths, false);

        // Should deduplicate shared prefix [a,b,c] at positions [0,1,2]
        // and shared subsequence [e,f,g] at positions [3,4,5]
        assert_eq!(compact_input_ids.len(), 12); // Reduced from 15 to 12
        assert_eq!(compact_position_ids.len(), 12);
        assert_eq!(scatter_indices.len(), 15); // Original length preserved
        assert_eq!(fold_gather.len(), 12); // Same as compact length

        // Verify that we can reconstruct original sequences using scatter indices
        for i in 0..input_ids.len() {
            let compact_idx = scatter_indices[i] as usize;
            assert_eq!(input_ids[i], compact_input_ids[compact_idx]);
            assert_eq!(position_ids[i], compact_position_ids[compact_idx]);
        }
    }

    #[test]
    fn test_compute_fold_and_scatter_identical_sequences() {
        // Two identical sequences: [a,b,c] and [a,b,c]
        let input_ids = vec![1, 2, 3, 1, 2, 3];
        let position_ids = vec![0, 1, 2, 0, 1, 2];
        let cu_seq_lengths = vec![0, 3, 6];

        let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) =
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths, false);

        // Should completely deduplicate to single sequence
        assert_eq!(compact_input_ids, vec![1, 2, 3]);
        assert_eq!(compact_position_ids, vec![0, 1, 2]);
        assert_eq!(scatter_indices, vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(fold_gather, vec![0, 1, 2]);
    }

    #[test]
    fn test_fold_gather_points_to_first_occurrence() {
        // Two sequences with overlapping prefixes/suffixes
        // S1: a b c d
        // S2: a b e f
        let input_ids = vec![1, 2, 3, 4, 1, 2, 5, 6];
        let position_ids = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let cu = vec![0, 4, 8];

        let (compact_ids, compact_pos, scatter, fold_gather) =
            compute_fold_and_scatter(&input_ids, &position_ids, &cu, false);

        // For each compact index, compute the minimal original position that maps to it.
        let mut mins = vec![u32::MAX; compact_ids.len()];
        for (orig_idx, &cidx) in scatter.iter().enumerate() {
            mins[cidx as usize] = mins[cidx as usize].min(orig_idx as u32);
        }

        assert_eq!(mins.len(), fold_gather.len());
        for (i, (&m, &fg)) in mins.iter().zip(fold_gather.iter()).enumerate() {
            assert_eq!(m, fg, "fold_gather[{}] should be first occurrence index", i);
            // sanity: the pair at fold_gather matches compact pair at i
            let fi = fg as usize;
            assert_eq!(input_ids[fi], compact_ids[i]);
            assert_eq!(position_ids[fi], compact_pos[i]);
        }
    }

    #[test]
    fn test_compute_fold_and_scatter_no_overlap() {
        // Two sequences with no overlap: [a,b] and [c,d]
        let input_ids = vec![1, 2, 3, 4];
        let position_ids = vec![0, 1, 0, 1];
        let cu_seq_lengths = vec![0, 2, 4];

        let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) =
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths, false);

        // No deduplication possible
        assert_eq!(compact_input_ids, vec![1, 2, 3, 4]);
        assert_eq!(compact_position_ids, vec![0, 1, 0, 1]);
        assert_eq!(scatter_indices, vec![0, 1, 2, 3]);
        assert_eq!(fold_gather, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_compute_fold_and_scatter_partial_overlap() {
        // Sequences: [a,b,c] and [a,b,d]
        let input_ids = vec![1, 2, 3, 1, 2, 4];
        let position_ids = vec![0, 1, 2, 0, 1, 2];
        let cu_seq_lengths = vec![0, 3, 6];

        let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) =
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths, false);

        // Should deduplicate shared prefix [a,b] at positions [0,1]
        assert_eq!(compact_input_ids.len(), 4); // [a,b,c,d] in some order
        assert_eq!(compact_position_ids.len(), 4);
        assert_eq!(scatter_indices.len(), 6);
        assert_eq!(fold_gather.len(), 4);

        // Verify reconstruction
        for i in 0..input_ids.len() {
            let compact_idx = scatter_indices[i] as usize;
            assert_eq!(input_ids[i], compact_input_ids[compact_idx]);
            assert_eq!(position_ids[i], compact_position_ids[compact_idx]);
        }
    }

    #[test]
    fn test_compute_fold_and_scatter_different_positions() {
        // Same tokens but different positions: [a,b] at [0,1] and [a,b] at [2,3]
        let input_ids = vec![1, 2, 1, 2];
        let position_ids = vec![0, 1, 2, 3];
        let cu_seq_lengths = vec![0, 2, 4];

        let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) =
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths, false);

        // Should NOT deduplicate because positions are different
        assert_eq!(compact_input_ids.len(), 4);
        assert_eq!(compact_position_ids.len(), 4);
        assert_eq!(scatter_indices, vec![0, 1, 2, 3]);
        assert_eq!(fold_gather, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_compute_fold_and_scatter_three_sequences_complex() {
        // Three sequences with various overlaps:
        // Seq1: [a,b,c,d] at [0,1,2,3]
        // Seq2: [a,b,e,f] at [0,1,2,3]
        // Seq3: [a,b,c,g] at [0,1,2,3]
        let input_ids = vec![1, 2, 3, 4, 1, 2, 5, 6, 1, 2, 3, 7];
        let position_ids = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let cu_seq_lengths = vec![0, 4, 8, 12];

        let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) =
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths, false);

        // Should deduplicate:
        // - [a,b] at [0,1] shared by all three
        // - [c] at [2] shared by seq1 and seq3
        assert!(compact_input_ids.len() < 12); // Some deduplication should occur
        assert_eq!(scatter_indices.len(), 12);
        assert_eq!(fold_gather.len(), compact_input_ids.len());

        // Verify reconstruction
        for i in 0..input_ids.len() {
            let compact_idx = scatter_indices[i] as usize;
            assert_eq!(input_ids[i], compact_input_ids[compact_idx]);
            assert_eq!(position_ids[i], compact_position_ids[compact_idx]);
        }
    }

    #[test]
    fn test_compute_fold_and_scatter_edge_case_single_token() {
        // Multiple single-token sequences
        let input_ids = vec![1, 2, 1];
        let position_ids = vec![0, 0, 0];
        let cu_seq_lengths = vec![0, 1, 2, 3];

        let (compact_input_ids, _compact_position_ids, scatter_indices, fold_gather) =
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths, false);

        // Should deduplicate token 1 at position 0
        assert_eq!(compact_input_ids.len(), 2); // [1, 2]
        assert_eq!(scatter_indices, vec![0, 1, 0]); // First and third map to same compact index
        assert_eq!(fold_gather.len(), 2);
    }

    #[test]
    fn test_compute_fold_and_scatter_deterministic_ordering() {
        // Test that the function produces consistent results
        let input_ids = vec![1, 2, 3, 1, 2, 4];
        let position_ids = vec![0, 1, 2, 0, 1, 2];
        let cu_seq_lengths = vec![0, 3, 6];

        let result1 = compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths, false);
        let result2 = compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths, false);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_padding_logic() {
        // Test case 1: Compact size < 1024, needs padding to multiple of 8
        let input_ids_1 = vec![1, 2, 3, 1, 2, 4]; // compact size = 4
        let position_ids_1 = vec![0, 1, 2, 0, 1, 2];
        let cu_seq_lengths_1 = vec![0, 3, 6];
        let (compact_ids_1, _, _, _) =
            compute_fold_and_scatter(&input_ids_1, &position_ids_1, &cu_seq_lengths_1, true);
        assert_eq!(compact_ids_1.len(), 8, "Should pad from 4 to 8");

        // Test case 2: Compact size < 1024, already a multiple of 8
        let input_ids_2 = (0..8).collect::<Vec<u32>>();
        let position_ids_2 = (0..8).collect::<Vec<u32>>();
        let cu_seq_lengths_2 = vec![0, 8];
        let (compact_ids_2, _, _, _) =
            compute_fold_and_scatter(&input_ids_2, &position_ids_2, &cu_seq_lengths_2, true);
        assert_eq!(
            compact_ids_2.len(),
            8,
            "Should not pad when already multiple of 8 and small input?"
        );

        // Test case 3: Compact size > 1024, needs padding to multiple of 64
        let n = 2047;
        let input_ids_3 = (0..n).collect::<Vec<u32>>();
        let position_ids_3 = (0..n).collect::<Vec<u32>>();
        let cu_seq_lengths_3 = vec![0, n];
        let (compact_ids_3, _, _, _) =
            compute_fold_and_scatter(&input_ids_3, &position_ids_3, &cu_seq_lengths_3, true);
        assert_eq!(compact_ids_3.len(), 2048, "Should pad from 2047 to 2048");

        // Test case 4: Compact size > 1024, already a multiple of 64
        let n = 1024;
        let input_ids_4 = (0..n).collect::<Vec<u32>>();
        let position_ids_4 = (0..n).collect::<Vec<u32>>();
        let cu_seq_lengths_4 = vec![0, n];
        let (compact_ids_4, _, _, _) =
            compute_fold_and_scatter(&input_ids_4, &position_ids_4, &cu_seq_lengths_4, true);
        assert_eq!(
            compact_ids_4.len(),
            1024,
            "Should not pad when already multiple of 64"
        );
    }

    #[test]
    fn test_padding_to_multiple_of_8() {
        // Compact size will be 4, padding should bring it to 8.
        let input_ids = vec![1, 2, 3, 1, 2, 4]; // compact: [1,2,3,4]
        let position_ids = vec![0, 1, 2, 0, 1, 2];
        let cu_seq_lengths = vec![0, 3, 6];

        let (compact_input_ids, compact_position_ids, _scatter, fold_gather) =
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths, true);

        assert_eq!(compact_input_ids.len(), 8, "Should be padded to 8");
        assert_eq!(compact_position_ids.len(), 8, "Should be padded to 8");
        assert_eq!(fold_gather.len(), 8, "Should be padded to 8");

        // Check that the first part is correct
        assert_eq!(&compact_input_ids[0..4], &[1, 2, 3, 4]);
        // Check that padding is zeros
        assert_eq!(&compact_input_ids[4..8], &[0, 0, 0, 0]);
        assert_eq!(&compact_position_ids[4..8], &[0, 0, 0, 0]);
        assert_eq!(&fold_gather[4..8], &[0, 0, 0, 0]);

        // Test case where no padding is needed (compact size is already a multiple of 8)
        // Let's create a case that compacts to 8 tokens
        let input_ids_no_pad = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let position_ids_no_pad = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let cu_seq_lengths_no_pad = vec![0, 8];
        let (compact_ids_no_pad, _, _, _) = compute_fold_and_scatter(
            &input_ids_no_pad,
            &position_ids_no_pad,
            &cu_seq_lengths_no_pad,
            true,
        );
        assert_eq!(compact_ids_no_pad.len(), 8, "Should not be padded");
    }

    // also, add some tests that allow you to reconstruct. e.g. do a function where we do the following function.
    // this test is more sophisticated.
    // impagine the baseline is
    // input_ids = [..]
    // position_ids =

    // def positional_embeddings()? e.g. add for each position 0.01 to the input_ids.
    // optional

    // def dummy_mlp(input_tensors: Vec[f32]):
    //    input_ids *= 2 // simulates the mlp part, input ids get embedded.
    //
    /// def dummy_attention(transformed_ids: Vec[f32], cu_seq_lengths):
    ///      let final_values = []
    ///      for start, end in cu_seq_lengths.take_two(): // unsure how
    ///           sequence_only = vector[slice(start, end)]
    ///           // attention part:
    ///           attention = cumsum(sequence_only)
    ///           final_values.push(attention)
    ///      final_values
    ///
    /// now do a range of input_ids, with and without prefix.
    ///
    /// attn_orig = attention(dummy_mlp(input_ids, cu_seq_length)
    ///
    /// for radix mlp approach
    /// fold_ids, fold_positions, scatter = compute_fold_and_scatter()
    /// compact_input_ids = input_ids.index_select(fold_ids)
    /// compact_positions_ids = position_ids.index_select(fold_ids)
    ///
    /// compact_mlp_out = mlp(compact_input_ids) // output and input are len_compact
    /// mlp_unfolded = compact_mlp_out.index_select(compact_mlp_out) // unfolded len is OG length before compact
    /// attention_folded = dummy_attention(mlp_unfolded)
    ///
    /// test with various instances, and always assert that attention_folded and unfolded are always the same.
    /// you could just implement it in plain rust, but also use helpers.
    /// run over a large range of possible samples and interesting range of inputs.
    // Helper functions for simulation
    fn apply_positional_embeddings(input_ids: &[u32], position_ids: &[u32]) -> Vec<f32> {
        input_ids
            .iter()
            .zip(position_ids.iter())
            .map(|(&token, &pos)| {
                let base = token as f32;
                let pos_embed = (pos as f32 * 0.1).sin() * 0.01;
                base + pos_embed
            })
            .collect()
    }

    fn dummy_mlp(input_embeddings: &[f32]) -> Vec<f32> {
        // Simple MLP: multiply by 2 and add small nonlinearity
        input_embeddings
            .iter()
            .map(|&x| x * 2.0 + (x * 0.1).tanh() * 0.1)
            .collect()
    }

    fn dummy_attention(mlp_outputs: &[f32], cu_seq_lengths: &[u32]) -> Vec<f32> {
        let mut final_values = Vec::new();

        for i in 0..cu_seq_lengths.len().saturating_sub(1) {
            let start = cu_seq_lengths[i] as usize;
            let end = cu_seq_lengths[i + 1] as usize;

            if start < end && end <= mlp_outputs.len() {
                let sequence_slice = &mlp_outputs[start..end];

                // Cumulative sum (simplified attention)
                let mut cumsum = 0.0;
                for &value in sequence_slice {
                    cumsum += value;
                    final_values.push(cumsum);
                }
            }
        }

        final_values
    }

    fn index_select_f32(source: &[f32], indices: &[u32]) -> Vec<f32> {
        indices.iter().map(|&idx| source[idx as usize]).collect()
    }

    // Parameterized comparison function
    #[derive(Debug)]
    struct RadixMLPTestResult {
        baseline_output: Vec<f32>,
        radix_output: Vec<f32>,
        compression_ratio: f32,
        original_tokens: usize,
        compact_tokens: usize,
    }

    fn run_radix_mlp_comparison(
        input_ids: &[u32],
        position_ids: &[u32],
        cu_seq_lengths: &[u32],
        pad_multiple_of_8: bool,
    ) -> RadixMLPTestResult {
        // Baseline computation pipeline
        let embeddings = apply_positional_embeddings(input_ids, position_ids);
        let mlp_outputs = dummy_mlp(&embeddings);
        let attention_baseline = dummy_attention(&mlp_outputs, cu_seq_lengths);

        // RadixMLP computation pipeline
        let (compact_input_ids, compact_position_ids, scatter_indices, _fold_gather) =
            compute_fold_and_scatter(input_ids, position_ids, cu_seq_lengths, pad_multiple_of_8);

        let compact_embeddings =
            apply_positional_embeddings(&compact_input_ids, &compact_position_ids);
        let compact_mlp_outputs = dummy_mlp(&compact_embeddings);
        let unfolded_mlp_outputs = index_select_f32(&compact_mlp_outputs, &scatter_indices);
        let attention_radix = dummy_attention(&unfolded_mlp_outputs, cu_seq_lengths);

        // Calculate metrics
        let original_tokens = input_ids.len();
        let compact_tokens = compact_input_ids.len();
        let compression_ratio = if original_tokens > 0 {
            compact_tokens as f32 / original_tokens as f32
        } else {
            1.0
        };

        RadixMLPTestResult {
            baseline_output: attention_baseline,
            radix_output: attention_radix,
            compression_ratio,
            original_tokens,
            compact_tokens,
        }
    }

    fn assert_outputs_equal(result: &RadixMLPTestResult, test_name: &str, tolerance: f32) {
        assert_eq!(
            result.baseline_output.len(),
            result.radix_output.len(),
            "{}: Output length mismatch",
            test_name
        );

        for (i, (baseline, radix)) in result
            .baseline_output
            .iter()
            .zip(result.radix_output.iter())
            .enumerate()
        {
            assert!(
                (baseline - radix).abs() < tolerance,
                "{}: Mismatch at index {}: baseline={}, radix={}, diff={}",
                test_name,
                i,
                baseline,
                radix,
                (baseline - radix).abs()
            );
        }
    }

    fn assert_compression_achieved(
        result: &RadixMLPTestResult,
        test_name: &str,
        expected_compression: bool,
        pad_multiple_of_8: bool,
    ) {
        if expected_compression {
            // When padding is enabled, we might not strictly achieve compression
            // if the overhead of padding > the gain from deduplication.
            // But generally for these tests we construct cases where deduplication is significant.
            // We can relax this check or make it context aware, but for now let's keep it simple.
            // NOTE: logic kept as is, might fail if padding > savings.
            let addition = if pad_multiple_of_8 {
                8 - (result.compact_tokens % 8)
            } else {
                0
            };
            assert!(
                result.compact_tokens < result.original_tokens + addition,
                "{}: Expected compression but got {} -> {} tokens",
                test_name,
                result.original_tokens,
                result.compact_tokens
            );
        } else if pad_multiple_of_8 {
            // With padding, we might not achieve compression if the compact size is already a multiple of 8.
            assert!(
                result.compact_tokens >= result.original_tokens,
                "{}: Expected no compression (>=) but got {} -> {} tokens",
                test_name,
                result.original_tokens,
                result.compact_tokens
            );
        } else {
            // Without padding, we should not have fewer tokens than original.
            assert_eq!(
                result.compact_tokens, result.original_tokens,
                "{}: Expected no compression but got {} -> {} tokens",
                test_name, result.original_tokens, result.compact_tokens
            );
        }
    }

    // Test case structure for parameterized tests
    #[derive(Debug)]
    struct TestCase {
        name: &'static str,
        input_ids: Vec<u32>,
        position_ids: Vec<u32>,
        cu_seq_lengths: Vec<u32>,
        expect_compression: bool,
        expected_compression_ratio: Option<f32>, // None means don't check specific ratio
        pad_multiple_of_8: bool,
    }

    // ...existing basic tests...
    #[test]
    fn test_radix_mlp_reconstruction_parameterized() {
        let test_cases = vec![
            TestCase {
                name: "identical_sequences",
                input_ids: vec![5, 10, 15, 5, 10, 15],
                position_ids: vec![0, 1, 2, 0, 1, 2],
                cu_seq_lengths: vec![0, 3, 6],
                expect_compression: true,
                expected_compression_ratio: Some(0.5), // 6 -> 3 tokens
                pad_multiple_of_8: false,
            },
            TestCase {
                name: "identical_sequences_padded",
                input_ids: vec![5, 10, 15, 5, 10, 15],
                position_ids: vec![0, 1, 2, 0, 1, 2],
                cu_seq_lengths: vec![0, 3, 6],
                expect_compression: false, // 6 -> 3 -> padded to 8. 8 > 6. So strictly no compression in terms of count.
                expected_compression_ratio: None,
                pad_multiple_of_8: true,
            },
            TestCase {
                name: "shared_prefix",
                input_ids: vec![1, 2, 3, 1, 2, 4],
                position_ids: vec![0, 1, 2, 0, 1, 2],
                cu_seq_lengths: vec![0, 3, 6],
                expect_compression: true,
                expected_compression_ratio: Some(4.0 / 6.0), // 6 -> 4 tokens
                pad_multiple_of_8: false,
            },
            TestCase {
                name: "shared_prefix_padded",
                input_ids: vec![1, 2, 3, 1, 2, 4],
                position_ids: vec![0, 1, 2, 0, 1, 2],
                cu_seq_lengths: vec![0, 3, 6],
                expect_compression: false, // 6 -> 4 -> padded to 8.
                expected_compression_ratio: None,
                pad_multiple_of_8: true,
            },
            TestCase {
                name: "no_overlap",
                input_ids: vec![1, 2, 3, 4, 5, 6],
                position_ids: vec![0, 1, 2, 0, 1, 2],
                cu_seq_lengths: vec![0, 3, 6],
                expect_compression: false,
                expected_compression_ratio: Some(1.0),
                pad_multiple_of_8: false,
            },
            TestCase {
                name: "complex_three_sequences",
                input_ids: vec![1, 2, 3, 4, 1, 2, 5, 6, 1, 2, 3, 7],
                position_ids: vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                cu_seq_lengths: vec![0, 4, 8, 12],
                expect_compression: true,
                expected_compression_ratio: None, // Don't check specific ratio
                pad_multiple_of_8: false,
            },
            TestCase {
                name: "complex_three_sequences_padded",
                input_ids: vec![1, 2, 3, 4, 1, 2, 5, 6, 1, 2, 3, 7],
                position_ids: vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                cu_seq_lengths: vec![0, 4, 8, 12],
                expect_compression: true, // 12 -> something < 12. If small enough, even with padding it's < 12.
                // Actual unique: [1,2,3,4,5,6,7] -> 7 unique tokens. Padded to 8. 8 < 12.
                expected_compression_ratio: None,
                pad_multiple_of_8: true,
            },
            TestCase {
                name: "single_tokens",
                input_ids: vec![1, 2, 1],
                position_ids: vec![0, 0, 0],
                cu_seq_lengths: vec![0, 1, 2, 3],
                expect_compression: true,
                expected_compression_ratio: Some(2.0 / 3.0), // 3 -> 2 tokens
                pad_multiple_of_8: false,
            },
            TestCase {
                name: "different_positions",
                input_ids: vec![1, 2, 1, 2],
                position_ids: vec![0, 1, 2, 3],
                cu_seq_lengths: vec![0, 2, 4],
                expect_compression: false,
                expected_compression_ratio: Some(1.0),
                pad_multiple_of_8: false,
            },
        ];

        for test_case in test_cases {
            let result = run_radix_mlp_comparison(
                &test_case.input_ids,
                &test_case.position_ids,
                &test_case.cu_seq_lengths,
                test_case.pad_multiple_of_8,
            );

            // Assert outputs are numerically identical
            assert_outputs_equal(&result, test_case.name, 1e-6);

            // Assert compression expectations
            assert_compression_achieved(
                &result,
                test_case.name,
                test_case.expect_compression,
                test_case.pad_multiple_of_8,
            );

            // Assert specific compression ratio if provided
            if let Some(expected_ratio) = test_case.expected_compression_ratio {
                assert!(
                    (result.compression_ratio - expected_ratio).abs() < 1e-6,
                    "{}: Expected compression ratio {}, got {}",
                    test_case.name,
                    expected_ratio,
                    result.compression_ratio
                );
            }

            println!(
                "{}: {} -> {} tokens (ratio: {:.3})",
                test_case.name,
                result.original_tokens,
                result.compact_tokens,
                result.compression_ratio
            );
        }
    }

    #[test]
    fn test_radix_mlp_edge_cases_parameterized() {
        let edge_cases = vec![
            TestCase {
                name: "empty",
                input_ids: vec![],
                position_ids: vec![],
                cu_seq_lengths: vec![],
                expect_compression: false,
                expected_compression_ratio: None,
                pad_multiple_of_8: false,
            },
            TestCase {
                name: "single_token_single_sequence",
                input_ids: vec![42],
                position_ids: vec![0],
                cu_seq_lengths: vec![0, 1],
                expect_compression: false,
                expected_compression_ratio: Some(1.0),
                pad_multiple_of_8: false,
            },
            TestCase {
                name: "single_token_single_sequence_padded",
                input_ids: vec![42],
                position_ids: vec![0],
                cu_seq_lengths: vec![0, 1],
                expect_compression: false,
                expected_compression_ratio: None,
                pad_multiple_of_8: true,
            },
            TestCase {
                name: "long_identical_sequences",
                input_ids: vec![1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                position_ids: vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                cu_seq_lengths: vec![0, 5, 10, 15],
                expect_compression: true,
                expected_compression_ratio: Some(1.0 / 3.0),
                pad_multiple_of_8: false,
            },
            TestCase {
                name: "long_identical_sequences_with_padding",
                input_ids: vec![1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                position_ids: vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                cu_seq_lengths: vec![0, 5, 10, 15],
                expect_compression: true,
                expected_compression_ratio: Some(8.0 / 15.0), // 15 -> 8 (with padding), ratio = 8/15 ~ 0.5333
                pad_multiple_of_8: true,
            },
        ];

        for test_case in edge_cases {
            if test_case.input_ids.is_empty() {
                let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) =
                    compute_fold_and_scatter(
                        &test_case.input_ids,
                        &test_case.position_ids,
                        &test_case.cu_seq_lengths,
                        test_case.pad_multiple_of_8,
                    );
                assert!(compact_input_ids.is_empty());
                assert!(compact_position_ids.is_empty());
                assert!(scatter_indices.is_empty());
                assert!(fold_gather.is_empty());
                continue;
            }

            let result = run_radix_mlp_comparison(
                &test_case.input_ids,
                &test_case.position_ids,
                &test_case.cu_seq_lengths,
                test_case.pad_multiple_of_8,
            );

            assert_outputs_equal(&result, test_case.name, 1e-6);
            assert_compression_achieved(
                &result,
                test_case.name,
                test_case.expect_compression,
                test_case.pad_multiple_of_8,
            );

            if let Some(expected_ratio) = test_case.expected_compression_ratio {
                assert!(
                    (result.compression_ratio - expected_ratio).abs() < 1e-6,
                    "{}: Expected compression ratio {}, got {}",
                    test_case.name,
                    expected_ratio,
                    result.compression_ratio
                );
            }
        }
    }

    #[test]
    fn fail_and_report_time_large_batch() {
        use std::time::Instant;

        // Relevant-sized problem:
        //  - batch = 32 sequences
        //  - each sequence has a shared prefix of 128 tokens (max dedup)
        //  - plus a unique tail of 200 tokens
        //  -> total ~ 10,496 tokens
        let batch: usize = 32;
        let shared_prefix: usize = 128;
        let tail_len: usize = 200;
        let seq_len: usize = shared_prefix + tail_len;
        let total_tokens: usize = batch * seq_len;

        let mut input_ids: Vec<u32> = Vec::with_capacity(total_tokens);
        let mut position_ids: Vec<u32> = Vec::with_capacity(total_tokens);
        let mut cu_seq_lengths: Vec<u32> = Vec::with_capacity(batch + 1);
        cu_seq_lengths.push(0);

        for seq_idx in 0..batch {
            // Shared prefix across all sequences: same tokens, same positions
            for j in 0..shared_prefix {
                let token = (j as u32 % 1000) + 1;
                input_ids.push(token);
                position_ids.push(j as u32);
            }
            // Unique tail per sequence to keep the problem realistic
            for k in 0..tail_len {
                let token = 1_000_000u32 + (seq_idx as u32) * 10_000 + (k as u32);
                input_ids.push(token);
                position_ids.push((shared_prefix + k) as u32);
            }
            cu_seq_lengths.push(input_ids.len() as u32);
        }

        let t0 = Instant::now();
        let (compact_ids, _compact_pos, _scatter, _fold) =
            super::compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths, false);
        let dt = t0.elapsed();
        let dt_ms = dt.as_secs_f64() * 1000.0;

        let ratio = (compact_ids.len() as f64) / (input_ids.len() as f64);

        // Use println! so you also see this under --nocapture; include details in the panic too.
        println!(
            "compute_fold_and_scatter:\n  batch={}\n  seq_len={}\n  total_tokens={}\n  compact_tokens={}\n  ratio={:.3}\n  elapsed_ms={:.3}",
            batch,
            seq_len,
            input_ids.len(),
            compact_ids.len(),
            ratio,
            dt_ms
        );

        // Intentionally fail so the timing and stats are printed in default test runs.
        // panic!(
        //     "TIMING REPORT (intentional failure to show output): \
        //      batch={}, seq_len={}, total_tokens={}, compact_tokens={}, ratio={:.3}, elapsed_ms={:.3}\n\
        //      scatter_len={}, fold_len={}, compact_pos_len={}",
        //     batch, seq_len, input_ids.len(), compact_ids.len(), ratio, dt_ms,
        //     scatter.len(), fold.len(), compact_pos.len()
        // );
    }
}
