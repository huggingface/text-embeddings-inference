use std::collections::HashMap;

// Transformer inference consists of two phases: \emph{prefill}, which processes all input tokens to initialize attention and MLP states, and \emph{decode}, which generates new tokens autoregressively. Prefill dominates runtime in stateless applications, where caching is either unavailable or reset between requests.

// Systems such as FlashAttention~\citep{dao2022flashattention}, FlashInfer~\citep{zheng2024flashinfer}, and HydraGen~\citep{juravsky2024hydragen} accelerate attention computations using efficient memory layouts. However, the MLP component---typically 40–60\% of inference FLOPs---remains fully recomputed even when many inputs share identical hidden states.

// We adopt the standard \emph{ragged layout} used in PyTorch and TensorRT-LLM:
// \begin{verbatim}
// tokens    = [a,b,c,d,e,f,g, a,b,c, e,f,g,h,i]
// pos       = [0,1,2,3,4,5,6, 0,1,2, 3,4,5,6,7]
// cu_seqlen = [0,7,15]
// \end{verbatim}
// This eliminates padding overhead but not redundant computation across sequences.

// % ---------------------- APPROACH ------------------------
// \section{Approach}
// \subsection{Folded Layout Construction}
// RadixMLP builds a prefix trie across sequences, identifying nodes with identical token and position pairs. Shared nodes are computed once, producing the \emph{folded layout}:
// \begin{verbatim}
// tokens    = [a,b,c, d,e,f,g, e,f,g,h,i]
// pos       = [0,1,2, 3,4,5,6, 3,4,5,6,7]
// cu_seqlen = [0,7,12]
// \end{verbatim}
// This reduces compute from 15 to 12 token evaluations in the example above.

// \subsection{Fold and Scatter Operators}
// Let $R$ denote the ragged layout and $C$ the folded layout.
// \begin{verbatim}
// fold_ids    = [0,1,2,3,4,5,6, 0,1,2,7,8,9,10,11]
// scatter_ids = {0:[0,7], 1:[1,8], 2:[2,9], ...}
// \end{verbatim}
// 
// in paractival matters, we aim to implement both as continous map

pub fn compute_fold_and_scatter(
    input_ids: &[u32],
    position_ids: &[u32],
    cu_seq_lengths: &[u32],
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    if input_ids.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    // Fast path: single sequence -> no cross-sequence dedup possible
    if cu_seq_lengths.len() == 2 {
        let n = input_ids.len() as u32;
        let ids: Vec<u32> = (0..n).collect();
        return (input_ids.to_vec(), position_ids.to_vec(), ids.clone(), ids);
    }

    #[derive(Debug, Clone)]
    struct TrieNode {
        token_id: u32,
        position: u32,
        children: std::collections::HashMap<(u32, u32), usize>,
        compact_index: Option<usize>,
    }

    use std::collections::HashMap;
    let mut trie_nodes: Vec<TrieNode> = Vec::new();
    let mut root_children: HashMap<(u32, u32), usize> = HashMap::new();

    // Build trie per sequence without keeping dangling borrows
    for seq_idx in 0..cu_seq_lengths.len().saturating_sub(1) {
        let start = cu_seq_lengths[seq_idx] as usize;
        let end = cu_seq_lengths[seq_idx + 1] as usize;

        let mut path: Vec<usize> = Vec::new();
        for pos in start..end {
            let token_id = input_ids[pos];
            let position = position_ids[pos];
            let key = (token_id, position);

            let child_idx = if let Some(parent_idx) = path.last().copied() {
                // Try to reuse an existing child under this parent
                if let Some(&idx) = trie_nodes[parent_idx].children.get(&key) {
                    idx
                } else {
                    // Create a new node and link it from the parent
                    let new_idx = trie_nodes.len();
                    trie_nodes.push(TrieNode {
                        token_id,
                        position,
                        children: HashMap::new(),
                        compact_index: None,
                    });
                    trie_nodes[parent_idx].children.insert(key, new_idx);
                    new_idx
                }
            } else {
                // At root
                if let Some(&idx) = root_children.get(&key) {
                    idx
                } else {
                    let new_idx = trie_nodes.len();
                    trie_nodes.push(TrieNode {
                        token_id,
                        position,
                        children: HashMap::new(),
                        compact_index: None,
                    });
                    root_children.insert(key, new_idx);
                    new_idx
                }
            };

            path.push(child_idx);
        }
    }

    // If no reduction, just return identity mappings
    if trie_nodes.len() >= input_ids.len() {
        let n = input_ids.len() as u32;
        let ids: Vec<u32> = (0..n).collect();
        return (input_ids.to_vec(), position_ids.to_vec(), ids.clone(), ids);
    }

    // Assign compact indices in a deterministic DFS: sort keys at each level
    let mut compact_input_ids = Vec::with_capacity(trie_nodes.len());
    let mut compact_position_ids = Vec::with_capacity(trie_nodes.len());
    let mut counter = 0usize;

    fn assign_compact_indices(
        children: &HashMap<(u32, u32), usize>,
        trie_nodes: &mut [TrieNode],
        out_tokens: &mut Vec<u32>,
        out_pos: &mut Vec<u32>,
        counter: &mut usize,
    ) {
        // Sort by (token_id, position) so order is stable across runs
        let mut pairs: Vec<((u32, u32), usize)> =
            children.iter().map(|(k, &v)| (*k, v)).collect();
        pairs.sort_unstable_by_key(|(k, _)| *k);

        for (_, node_idx) in pairs {
            let node = &mut trie_nodes[node_idx];
            node.compact_index = Some(*counter);
            out_tokens.push(node.token_id);
            out_pos.push(node.position);
            *counter += 1;

            // Recurse on a snapshot to avoid borrowing issues
            let child_copy = node.children.clone();
            assign_compact_indices(&child_copy, trie_nodes, out_tokens, out_pos, counter);
        }
    }

    assign_compact_indices(
        &root_children,
        &mut trie_nodes,
        &mut compact_input_ids,
        &mut compact_position_ids,
        &mut counter,
    );

        // Build scatter (for each original token -> compact index) and the first-occurrence gather
    let mut scatter_indices = Vec::with_capacity(input_ids.len());
    // Use the number of compact nodes we actually assigned
    let mut first_occurrence: Vec<Option<u32>> = vec![None; counter];

    for seq_idx in 0..cu_seq_lengths.len().saturating_sub(1) {
        let start = cu_seq_lengths[seq_idx] as usize;
        let end = cu_seq_lengths[seq_idx + 1] as usize;

        // Track where we are in the trie while walking this sequence.
        let mut parent: Option<usize> = None;

        for pos in start..end {
            let token_id = input_ids[pos];
            let position = position_ids[pos];
            let key = (token_id, position);

            // Find the trie node for this (token, position) under the correct parent.
            let node_idx = match parent {
                None => *root_children
                    .get(&key)
                    .expect("Trie must contain root node for first token of sequence"),
                Some(pidx) => *trie_nodes[pidx]
                    .children
                    .get(&key)
                    .expect("Trie must contain child node for subsequent token"),
            };

            let compact_idx = trie_nodes[node_idx]
                .compact_index
                .expect("compact index assigned for every trie node");
            scatter_indices.push(compact_idx as u32);

            if first_occurrence[compact_idx].is_none() {
                first_occurrence[compact_idx] = Some(pos as u32);
            }

            // Advance within the same sequence
            parent = Some(node_idx);
        }
    }

    let fold_gather: Vec<u32> = first_occurrence
        .into_iter()
        .map(|x| x.expect("every compact node appears at least once"))
        .collect();

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
        let input_ids = vec![];
        let position_ids = vec![];
        let cu_seq_lengths = vec![];
        
        let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) = 
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths);
        
        assert_eq!(compact_input_ids, vec![]);
        assert_eq!(compact_position_ids, vec![]);
        assert_eq!(scatter_indices, vec![]);
        assert_eq!(fold_gather, vec![]);
    }

    #[test]
    fn test_compute_fold_and_scatter_single_sequence() {
        // Single sequence: [a, b, c]
        let input_ids = vec![1, 2, 3];
        let position_ids = vec![0, 1, 2];
        let cu_seq_lengths = vec![0, 3];
        
        let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) = 
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths);
        
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
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths);
        
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
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths);
        
        // Should completely deduplicate to single sequence
        assert_eq!(compact_input_ids, vec![1, 2, 3]);
        assert_eq!(compact_position_ids, vec![0, 1, 2]);
        assert_eq!(scatter_indices, vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(fold_gather, vec![0, 1, 2]);
    }

    #[test]
    fn test_compute_fold_and_scatter_no_overlap() {
        // Two sequences with no overlap: [a,b] and [c,d]
        let input_ids = vec![1, 2, 3, 4];
        let position_ids = vec![0, 1, 0, 1];
        let cu_seq_lengths = vec![0, 2, 4];
        
        let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) = 
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths);
        
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
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths);
        
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
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths);
        
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
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths);
        
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
        
        let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) = 
            compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths);
        
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
        
        let result1 = compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths);
        let result2 = compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths);
        
        assert_eq!(result1, result2);
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
        input_ids.iter().zip(position_ids.iter())
            .map(|(&token, &pos)| {
                let base = token as f32;
                let pos_embed = (pos as f32 * 0.1).sin() * 0.01;
                base + pos_embed
            })
            .collect()
    }

    fn dummy_mlp(input_embeddings: &[f32]) -> Vec<f32> {
        // Simple MLP: multiply by 2 and add small nonlinearity
        input_embeddings.iter()
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
        indices.iter()
            .map(|&idx| source[idx as usize])
            .collect()
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
    ) -> RadixMLPTestResult {
        // Baseline computation pipeline
        let embeddings = apply_positional_embeddings(input_ids, position_ids);
        let mlp_outputs = dummy_mlp(&embeddings);
        let attention_baseline = dummy_attention(&mlp_outputs, cu_seq_lengths);

        // RadixMLP computation pipeline
        let (compact_input_ids, compact_position_ids, scatter_indices, _fold_gather) = 
            compute_fold_and_scatter(input_ids, position_ids, cu_seq_lengths);
        
        let compact_embeddings = apply_positional_embeddings(&compact_input_ids, &compact_position_ids);
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
            "{}: Output length mismatch", test_name
        );

        for (i, (baseline, radix)) in result.baseline_output.iter()
            .zip(result.radix_output.iter())
            .enumerate() 
        {
            assert!(
                (baseline - radix).abs() < tolerance,
                "{}: Mismatch at index {}: baseline={}, radix={}, diff={}",
                test_name, i, baseline, radix, (baseline - radix).abs()
            );
        }
    }

    fn assert_compression_achieved(result: &RadixMLPTestResult, test_name: &str, expected_compression: bool) {
        if expected_compression {
            assert!(
                result.compact_tokens < result.original_tokens,
                "{}: Expected compression but got {} -> {} tokens",
                test_name, result.original_tokens, result.compact_tokens
            );
        } else {
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
            },
            TestCase {
                name: "shared_prefix",
                input_ids: vec![1, 2, 3, 1, 2, 4],
                position_ids: vec![0, 1, 2, 0, 1, 2],
                cu_seq_lengths: vec![0, 3, 6],
                expect_compression: true,
                expected_compression_ratio: Some(4.0 / 6.0), // 6 -> 4 tokens
            },
            TestCase {
                name: "no_overlap",
                input_ids: vec![1, 2, 3, 4, 5, 6],
                position_ids: vec![0, 1, 2, 0, 1, 2],
                cu_seq_lengths: vec![0, 3, 6],
                expect_compression: false,
                expected_compression_ratio: Some(1.0),
            },
            TestCase {
                name: "complex_three_sequences",
                input_ids: vec![1, 2, 3, 4, 1, 2, 5, 6, 1, 2, 3, 7],
                position_ids: vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                cu_seq_lengths: vec![0, 4, 8, 12],
                expect_compression: true,
                expected_compression_ratio: None, // Don't check specific ratio
            },
            TestCase {
                name: "single_tokens",
                input_ids: vec![1, 2, 1],
                position_ids: vec![0, 0, 0],
                cu_seq_lengths: vec![0, 1, 2, 3],
                expect_compression: true,
                expected_compression_ratio: Some(2.0 / 3.0), // 3 -> 2 tokens
            },
            TestCase {
                name: "different_positions",
                input_ids: vec![1, 2, 1, 2],
                position_ids: vec![0, 1, 2, 3],
                cu_seq_lengths: vec![0, 2, 4],
                expect_compression: false,
                expected_compression_ratio: Some(1.0),
            },
        ];

        for test_case in test_cases {
            let result = run_radix_mlp_comparison(
                &test_case.input_ids,
                &test_case.position_ids,
                &test_case.cu_seq_lengths,
            );

            // Assert outputs are numerically identical
            assert_outputs_equal(&result, test_case.name, 1e-6);

            // Assert compression expectations
            assert_compression_achieved(&result, test_case.name, test_case.expect_compression);

            // Assert specific compression ratio if provided
            if let Some(expected_ratio) = test_case.expected_compression_ratio {
                assert!(
                    (result.compression_ratio - expected_ratio).abs() < 1e-6,
                    "{}: Expected compression ratio {}, got {}",
                    test_case.name, expected_ratio, result.compression_ratio
                );
            }

            println!(
                "{}: {} -> {} tokens (ratio: {:.3})",
                test_case.name, result.original_tokens, result.compact_tokens, result.compression_ratio
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
            },
            TestCase {
                name: "single_token_single_sequence",
                input_ids: vec![42],
                position_ids: vec![0],
                cu_seq_lengths: vec![0, 1],
                expect_compression: false,
                expected_compression_ratio: Some(1.0),
            },
            TestCase {
                name: "long_identical_sequences",
                input_ids: vec![1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                position_ids: vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                cu_seq_lengths: vec![0, 5, 10, 15],
                expect_compression: true,
                expected_compression_ratio: Some(1.0 / 3.0),
            },
        ];

        for test_case in edge_cases {
            if test_case.input_ids.is_empty() {
                let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) = 
                    compute_fold_and_scatter(&test_case.input_ids, &test_case.position_ids, &test_case.cu_seq_lengths);
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
            );

            assert_outputs_equal(&result, test_case.name, 1e-6);
            assert_compression_achieved(&result, test_case.name, test_case.expect_compression);

            if let Some(expected_ratio) = test_case.expected_compression_ratio {
                assert!(
                    (result.compression_ratio - expected_ratio).abs() < 1e-6,
                    "{}: Expected compression ratio {}, got {}",
                    test_case.name, expected_ratio, result.compression_ratio
                );
            }
        }
    }
}