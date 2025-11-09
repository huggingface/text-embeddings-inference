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

#[derive(Debug, Clone)]
struct TrieNode {
    token_id: u32,
    position: u32,
    children: HashMap<(u32, u32), usize>, // (token_id, position) -> child_index
    compact_index: Option<usize>, // Index in the compacted representation
}

pub fn compute_fold_and_scatter(
    input_ids: &[u32], 
    position_ids: &[u32], 
    cu_seq_lengths: &[u32]
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    if input_ids.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    if cu_seq_lengths.len() == 2 {
        let scatter_indices: Vec<u32> = (0..input_ids.len() as u32).collect();
        let fold_gather: Vec<u32> = (0..input_ids.len() as u32).collect();
        return (input_ids.to_vec(), position_ids.to_vec(), scatter_indices, fold_gather);
    }
    
    let mut trie_nodes: Vec<TrieNode> = Vec::new();
    let mut root_children: HashMap<(u32, u32), usize> = HashMap::new();
    
    // Build trie for each sequence - FIX: Use indices instead of mutable references
    for seq_idx in 0..cu_seq_lengths.len() - 1 {
        let start = cu_seq_lengths[seq_idx] as usize;
        let end = cu_seq_lengths[seq_idx + 1] as usize;
        
        let mut current_path = Vec::new(); // Track path instead of mutable refs
        
        for pos in start..end {
            let token_id = input_ids[pos];
            let position = position_ids[pos];
            let key = (token_id, position);
            
            // Navigate to the right level in the trie
            let mut current_children: &HashMap<(u32, u32), usize> = &root_children;
            for &parent_idx in &current_path {
                current_children = &trie_nodes[parent_idx].children;
            }
            
            if let Some(&existing_idx) = current_children.get(&key) {
                current_path.push(existing_idx);
            } else {
                let new_idx = trie_nodes.len();
                trie_nodes.push(TrieNode {
                    token_id,
                    position,
                    children: HashMap::new(),
                    compact_index: None,
                });
                
                // Insert into the appropriate parent
                if current_path.is_empty() {
                    root_children.insert(key, new_idx);
                } else {
                    let parent_idx = *current_path.last().unwrap();
                    trie_nodes[parent_idx].children.insert(key, new_idx);
                }
                
                current_path.push(new_idx);
            }
        }
    }

    // Early exit if no deduplication achieved
    if trie_nodes.len() >= input_ids.len() {
        let scatter_indices: Vec<u32> = (0..input_ids.len() as u32).collect();
        let fold_gather: Vec<u32> = (0..input_ids.len() as u32).collect();
        return (input_ids.to_vec(), position_ids.to_vec(), scatter_indices, fold_gather);
    }
    
    // Assign compact indices in DFS order
    let mut compact_input_ids = Vec::with_capacity(trie_nodes.len());
    let mut compact_position_ids = Vec::with_capacity(trie_nodes.len());
    let mut compact_counter = 0;
    
    fn assign_compact_indices(
        children: &HashMap<(u32, u32), usize>,
        trie_nodes: &mut [TrieNode],
        compact_input_ids: &mut Vec<u32>,
        compact_position_ids: &mut Vec<u32>,
        compact_counter: &mut usize,
    ) {
        for &node_idx in children.values() {
            let node = &mut trie_nodes[node_idx];
            node.compact_index = Some(*compact_counter);
            compact_input_ids.push(node.token_id);
            compact_position_ids.push(node.position);
            *compact_counter += 1;
            
            let children_copy = node.children.clone();
            assign_compact_indices(&children_copy, trie_nodes, compact_input_ids, compact_position_ids, compact_counter);
        }
    }
    
    assign_compact_indices(&root_children, &mut trie_nodes, &mut compact_input_ids, &mut compact_position_ids, &mut compact_counter);
    
    // Build BOTH mappings in a single pass
    let mut scatter_indices = Vec::with_capacity(input_ids.len());
    let mut first_occurrence = vec![None; trie_nodes.len()];
    
    for seq_idx in 0..cu_seq_lengths.len() - 1 {
        let start = cu_seq_lengths[seq_idx] as usize;
        let end = cu_seq_lengths[seq_idx + 1] as usize;
        
        let mut current_children = &root_children;
        
        for pos in start..end {
            let token_id = input_ids[pos];
            let position = position_ids[pos];
            let key = (token_id, position);
            
            if let Some(&node_idx) = current_children.get(&key) {
                let compact_idx = trie_nodes[node_idx].compact_index.unwrap();
                scatter_indices.push(compact_idx as u32);
                
                if first_occurrence[compact_idx].is_none() {
                    first_occurrence[compact_idx] = Some(pos as u32);
                }
                
                current_children = &trie_nodes[node_idx].children;
            }
        }
    }
    
    let fold_gather: Vec<u32> = first_occurrence.into_iter()
        .map(|opt| opt.unwrap())
        .collect();
    
    (compact_input_ids, compact_position_ids, scatter_indices, fold_gather)
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
    fn test_radix_mlp_stress_test_parameterized() {
        // Generator for test cases
        fn generate_test_case(seed: u64, pattern: &str) -> TestCase {
            let mut rng_state = seed;
            let mut simple_rng = || {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                (rng_state / 65536) % 32768
            };

            let (input_ids, position_ids, cu_seq_lengths) = if *pattern == "random_overlap" { // FIX: Add *
                let num_sequences = 2 + (simple_rng() % 4) as usize;
                let base_tokens = vec![1, 2, 3];
                let mut input_ids = Vec::new();
                let mut position_ids = Vec::new();
                let mut cu_seq_lengths = vec![0];

                for _ in 0..num_sequences {
                    input_ids.extend(&base_tokens);
                    position_ids.extend(0..base_tokens.len() as u32);
                    
                    let suffix_len = 1 + (simple_rng() % 3) as usize;
                    for pos in base_tokens.len()..base_tokens.len() + suffix_len {
                        input_ids.push(10 + (simple_rng() % 5) as u32);
                        position_ids.push(pos as u32);
                    }
                    cu_seq_lengths.push(input_ids.len() as u32);
                }
                (input_ids, position_ids, cu_seq_lengths)
            } else {
                let num_sequences = 2 + (simple_rng() % 3) as usize;
                let mut input_ids = Vec::new();
                let mut position_ids = Vec::new();
                let mut cu_seq_lengths = vec![0];

                for seq_idx in 0..num_sequences {
                    let seq_len = 2 + (simple_rng() % 4) as usize;
                    let base_token = 1 + seq_idx as u32 * 10;
                    
                    for pos in 0..seq_len {
                        input_ids.push(base_token + pos as u32);
                        position_ids.push(pos as u32);
                    }
                    cu_seq_lengths.push(input_ids.len() as u32);
                }
                (input_ids, position_ids, cu_seq_lengths)
            };

            TestCase {
                name: if *pattern == "random_overlap" { "random_overlap" } else { "no_overlap" }, // FIX: Use static strings
                input_ids,
                position_ids,
                cu_seq_lengths,
                expect_compression: *pattern == "random_overlap", // FIX: Add *
                expected_compression_ratio: None,
            }
        }

        let patterns = vec!["random_overlap", "no_overlap"];
        
        for seed in 0..20 {
            for pattern in &patterns {
                let test_case = generate_test_case(seed, pattern);
                
                let result = run_radix_mlp_comparison(
                    &test_case.input_ids,
                    &test_case.position_ids,
                    &test_case.cu_seq_lengths,
                );

                // Assert outputs are numerically identical
                assert_outputs_equal(&result, &format!("{}_seed_{}", pattern, seed), 1e-6);

                // Assert compression expectations for overlap patterns
                if pattern == "random_overlap" {
                    assert!(
                        result.compression_ratio <= 1.0,
                        "Seed {}, Pattern {}: Compression ratio should be <= 1.0, got {}",
                        seed, pattern, result.compression_ratio
                    );
                }
            }
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