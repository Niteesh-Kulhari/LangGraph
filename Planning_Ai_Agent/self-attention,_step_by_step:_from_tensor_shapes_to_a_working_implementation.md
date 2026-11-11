# Self-Attention, Step by Step: From Tensor Shapes to a Working Implementation 

 ## Frame the problem: why let tokens attend to one another?

Consider: “Maya put the cup beside the laptop because it was unstable.” To interpret *it*, a model must connect that token to a relevant earlier token—here, likely *cup*—rather than rely only on nearby words. Self-attention gives each position a way to score and combine information from other positions in the same sequence.

A fixed-size context window represents only the tokens it includes. If the needed reference falls outside that window, the model cannot use it directly. A recurrent network instead passes a hidden state from token to token; that state summarizes the prefix, which can make distant information harder to preserve, and processing is sequential.

For a batch of token embeddings, write the input as `X ∈ ℝ^(B × L × D)`, where `B` is batch size, `L` is sequence length, and `D` is embedding dimension. Self-attention produces an output at each sequence position, using information from positions in that sequence. In the usual implementation, the output has the same shape as `X`.

We’ll first derive single-head scaled dot-product self-attention, then add masking, build an implementation, and discuss trade-offs. The key idea is to compute token-to-token relevance scores and use them to form weighted combinations of token representations.

## Build the mechanism from queries, keys, and values

Let the input be \(X \in \mathbb{R}^{B \times T \times d_{\text{model}}}\), where \(B\) is batch size and \(T\) is sequence length. A self-attention head applies three learned linear projections:

\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
\]

With row-vector embeddings, \(W_Q,W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}\) and \(W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}\). Thus, \(Q,K\) have shape \([B,T,d_k]\), and \(V\) has shape \([B,T,d_v]\). Each position gets a query, key, and value.

Compute pairwise query–key scores, then scale them:

\[
S = \frac{QK^\top}{\sqrt{d_k}}
\]

Transpose the last two dimensions of \(K\), so \(S\) has shape \([B,T,T]\): each query position scores every key position. Dividing by \(\sqrt{d_k}\) keeps dot-product magnitudes from growing too much as \(d_k\) increases. Without scaling, large logits can make softmax overly peaked, weakening gradients.

Apply softmax across the key-position axis—the last dimension—to get attention weights \(A\). Then combine the values:

\[
A = \operatorname{softmax}(S,\text{dim}=-1),\qquad O = AV
\]

Here \(A\) is \([B,T,T]\) and \(O\) is \([B,T,d_v]\). Each output is a weighted sum of value vectors.

For one query, take \(d_k=2\), \(q=[1,0]\), keys \(k_1=[1,0]\), \(k_2=[0,1]\), and values \(v_1=[2,0]\), \(v_2=[0,4]\). The scaled scores are \([1/\sqrt{2},0]\), approximately \([0.707,0]\). Softmax gives weights \([0.670,0.330]\), which sum to 1. The output is \(0.670v_1+0.330v_2=[1.34,1.32]\) (rounded). This verifies that the query’s output is a convex weighted combination of the value vectors.

## Implement a minimal working example

The implementation below uses one attention head. Inputs and outputs use `(batch, sequence, embedding)` layout. Each projection preserves the embedding dimension, so the result has the same shape as the input.

```python
import math
import torch
from torch import nn

def attend(q, k, v):
    # q, k, v: (B, T, D)
    scores = q @ k.transpose(-2, -1)  # (B, T, D) @ (B, D, T) -> (B, T, T)
    scores = scores / math.sqrt(q.size(-1))  # (B, T, T)
    weights = scores.softmax(dim=-1)  # (B, T, T); rows sum to 1
    output = weights @ v  # (B, T, T) @ (B, T, D) -> (B, T, D)
    return output, weights

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, T, D); each projection returns (B, T, D)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return attend(q, k, v)
```

The transpose is essential: it changes `k` from `(B, T, D)` to `(B, D, T)`, making the inner dimensions match for `QKᵀ`. The resulting `(B, T, T)` scores compare every query position with every key position. Softmax over the final axis makes each query’s weights sum to one.

Run the module on a tiny batch. A fixed seed makes initialization and input values reproducible:

```python
torch.manual_seed(7)
model = SelfAttention(dim=4)
x = torch.randn(2, 3, 4)  # (B=2, T=3, D=4)
y, weights = model(x)

assert y.shape == x.shape  # (2, 3, 4)
assert weights.shape == (2, 3, 3)
```

Test two basic behaviors directly. Zero queries and keys produce identical scores, so each row of attention weights must be uniform. Changing the values must change the weighted output:

```python
q = torch.zeros(1, 3, 4)  # (B, T, D)
k = torch.zeros(1, 3, 4)
v1 = torch.randn(1, 3, 4)
v2 = v1 + 1

out1, w = attend(q, k, v1)
out2, _ = attend(q, k, v2)

torch.testing.assert_close(w, torch.full((1, 3, 3), 1 / 3))
assert not torch.allclose(out1, out2)
```

If a matrix-multiplication error reports incompatible dimensions, check the batch/sequence/embedding order and the key transpose first. This example assumes one head and equal query, key, and value dimensions; multi-head attention splits `D` into heads and adds reshape operations.

## Extend the implementation: masks, heads, and efficiency

A padding mask prevents real tokens from attending to padding tokens. Apply it to the score matrix **before** softmax; zeroing probabilities afterward does not produce the same normalized distribution.

```python
# scores: [B, H, T, T]; valid: [B, T], True for real tokens
scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
probs = scores.softmax(dim=-1)

# Every padded key should receive zero probability.
assert torch.all(probs.masked_select(~valid[:, None, None, :]) == 0)
```

This masks key positions; padded query outputs should also be ignored or explicitly zeroed. Ensure each example has at least one valid key: softmax over an all-masked row yields NaNs.

For autoregressive decoding, add a causal mask so position *i* can attend only to positions *j ≤ i*. Combine it with the padding mask when batches contain padding. Validate the causal constraint by changing a future token and checking that earlier outputs stay unchanged. Run the test in evaluation mode, with dropout disabled; bidirectional operations elsewhere in the model would invalidate this guarantee.

Multi-head attention divides the embedding dimension `d_model` into `h` heads, each of width `d_head = d_model / h`. Project inputs to queries, keys, and values, reshape to `[B, h, T, d_head]`, and compute scaled attention independently per head. Concatenate the head outputs back to `[B, T, d_model]`, then apply an output projection. Require `d_model` to be divisible by `h`; more heads provide separate attention subspaces but do not increase the total per-token width.

The score matrix has shape `[B, h, n, n]`, so its memory grows as O(hn²) (or O(n²) per head), and score computation scales roughly as O(n²d_model). Long sequences can make this the bottleneck. Benchmark fused or memory-efficient kernels, such as PyTorch scaled dot-product attention, when profiling shows attention dominates: they can reduce intermediate-memory traffic, though speedups depend on hardware, dtype, and mask support.

## Common mistakes and how to catch them

Assume attention logits have shape `[batch, heads, query_positions, key_positions]`. Softmax must normalize over keys: each query distributes its weight across the available key positions. Using `dim=-2` instead normalizes across queries, so the weights no longer describe which keys each query attends to.

```python
weights = logits.softmax(dim=-1)
assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
```

A finite mask value can also leak probability. For example, `-100` is not guaranteed to be small relative to every possible logit or numeric format. Mask with negative infinity, then test forbidden entries directly:

```python
masked_logits = logits.masked_fill(forbidden, float("-inf"))
weights = masked_logits.softmax(dim=-1)
assert torch.all(weights.masked_select(forbidden) == 0)
```

Avoid fully masked rows: softmax over all `-inf` values produces NaNs.

Without scaling, dot products grow with key dimension. If query and key components have unit variance, their dot-product standard deviation grows roughly as `sqrt(d_k)`. Large logits can saturate softmax and weaken gradients. Compare both the score and gradient scales:

```python
raw = q @ k.transpose(-2, -1)
scaled = raw / (k.size(-1) ** 0.5)
print(raw.std(), scaled.std())
```

In a training test, compare `q.grad.norm()` with and without scaling; scaling usually keeps gradients healthier for large `d_k`.

Finally, verify causal-mask orientation. Rows are queries; columns are keys. For tokens `0, 1, 2`, allowed positions should be:

```text
       key 0  1  2
query 0      ✓  ✗  ✗
      1      ✓  ✓  ✗
      2      ✓  ✓  ✓
```

An upper-triangular allowed mask exposes future tokens. Test the mask itself with this tiny case before running a full model.

## Production checklist and next steps

Before integrating self-attention, verify the component with a focused checklist:

- [ ] **Shapes:** For input `[B, T, D]`, confirm Q, K, and V have compatible dimensions; attention scores should be `[B, H, T, T]`, and the output should return to `[B, T, D]`.
- [ ] **Masks:** Check that padded key positions cannot receive attention and that causal masks block future positions. Confirm the mask’s Boolean convention or additive values match the API.
- [ ] **Finite outputs:** Assert that outputs and attention scores contain no NaNs or infinities, including for fully masked rows. Handle those rows explicitly to avoid softmax over all negative infinity.
- [ ] **Gradient flow:** Run a loss backward pass and check that expected parameters receive finite, nonzero gradients.
- [ ] **Tests:** Test padded batches with different valid lengths and causal inputs where changing future tokens must not change earlier outputs.

For debugging instability, log sequence length, attention-score summaries (for example, min, max, and standard deviation), and activation and gradient norms. Avoid logging token IDs or content; aggregate statistics are usually enough to spot scaling or masking problems without exposing sensitive data.

Benchmark candidate attention kernels using representative batch sizes, sequence lengths, head counts, and precision settings. Measure both latency and peak memory: a kernel that is faster for short sequences may use too much memory or perform worse at production lengths.

Next, integrate positional information, compare single-head and multi-head behavior, and inspect a small attention map. Treat the map as a diagnostic for patterns or masking errors—not a complete explanation of what the model learned.
