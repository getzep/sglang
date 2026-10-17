import unittest

import torch

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

CHUNK_SIZE = 64


def _make_varlen_inputs(seed, lens, num_heads=4, head_dim=128):
    """Packed varlen GDN inputs: [1, sum(lens), H, D] plus a zero fp32 state pool."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(lens)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=generator, device="cuda", dtype=dtype)

    q = randn(1, total, num_heads, head_dim)
    k = randn(1, total, num_heads, head_dim)
    v = (0.1 * randn(1, total, num_heads, head_dim, dtype=torch.float32)).to(
        torch.bfloat16
    )
    # Log-space forget gate in (-1, 0) and a beta in (0, 1), as the model emits.
    g = -torch.rand(1, total, num_heads, generator=generator, device="cuda") * 0.5
    beta = torch.sigmoid(randn(1, total, num_heads, dtype=torch.float32)).to(
        torch.bfloat16
    )
    state = torch.zeros(
        len(lens), num_heads, head_dim, head_dim, device="cuda", dtype=torch.float32
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    return q, k, v, g, beta, state, cu_seqlens


def _run(q, k, v, g, beta, state, cu_seqlens, **kwargs):
    return chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=state,
        initial_state_indices=torch.arange(
            state.shape[0], device="cuda", dtype=torch.int32
        ),
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        **kwargs,
    )


class TestGdnTrackState(CustomTestCase):
    @torch.inference_mode()
    def test_track_state_snapshots_fp32_accumulator(self):
        """The mamba radix track path snapshots the SSM state at a chunk boundary
        inside an extend into the fp32 state pool. Without `track_state` the GDN
        backend reads the per-chunk states `h` (activation dtype, bf16), so a
        prefix-cache hit restores a bf16-rounded state while a cache miss keeps
        fp32. `track_state` must carry the in-kernel fp32 accumulator: identical
        to the fp32 final state of a run truncated at the boundary, and strictly
        more precise than the bf16 `h` row for the same boundary.
        """
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        # seq0: 300 tokens, snapshot at the start of chunk 2 (128 tokens).
        # seq1: 64 tokens, not tracked. seq2: 200 tokens, snapshot at chunk 3.
        lens = [300, 64, 200]
        q, k, v, g, beta, state, cu_seqlens = _make_varlen_inputs(0, lens)
        num_heads, head_dim = q.shape[2], q.shape[3]

        track_state = torch.full(
            (len(lens), num_heads, head_dim, head_dim),
            float("nan"),
            device="cuda",
            dtype=torch.float32,
        )
        track_chunk_idx = torch.tensor([2, -1, 3], dtype=torch.int32, device="cuda")
        o_tracked, _, h = _run(
            q,
            k,
            v,
            g,
            beta,
            state.clone(),
            cu_seqlens,
            track_state=track_state,
            track_chunk_idx=track_chunk_idx,
        )
        o_plain, _, _ = _run(q, k, v, g, beta, state.clone(), cu_seqlens)
        # The snapshot must not change the attention output.
        torch.testing.assert_close(o_tracked, o_plain, rtol=0, atol=0)

        self.assertTrue(torch.all(torch.isnan(track_state[1])))
        self.assertFalse(torch.any(torch.isnan(track_state[0])))
        self.assertFalse(torch.any(torch.isnan(track_state[2])))

        starts = [0, *torch.tensor(lens).cumsum(0).tolist()]
        h_rows = [0, *((torch.tensor(lens) - 1) // CHUNK_SIZE + 1).cumsum(0).tolist()]
        for seq, chunk in ((0, 2), (2, 3)):
            with self.subTest(seq=seq, chunk=chunk):
                lo = starts[seq]
                hi = lo + chunk * CHUNK_SIZE
                ref_state = torch.zeros(
                    1, num_heads, head_dim, head_dim, device="cuda", dtype=torch.float32
                )
                ref_cu_seqlens = torch.tensor(
                    [0, hi - lo], dtype=torch.int32, device="cuda"
                )
                _run(
                    q[:, lo:hi],
                    k[:, lo:hi],
                    v[:, lo:hi],
                    g[:, lo:hi],
                    beta[:, lo:hi],
                    ref_state,
                    ref_cu_seqlens,
                )
                torch.testing.assert_close(
                    track_state[seq], ref_state[0], rtol=1e-5, atol=1e-5
                )
                # `h` holds the same state rounded to bf16 (within one bf16
                # ulp, because a different autotune configuration can change
                # the accumulation order), and the snapshot is closer to the
                # fp32 reference than `h`.
                h_row = h.squeeze(0)[h_rows[seq] + chunk].to(torch.float32)
                torch.testing.assert_close(
                    h_row,
                    ref_state[0].to(torch.bfloat16).to(torch.float32),
                    rtol=2**-7,
                    atol=1e-6,
                )
                snapshot_err = (track_state[seq] - ref_state[0]).abs().max()
                h_err = (h_row - ref_state[0]).abs().max()
                self.assertLess(snapshot_err.item(), h_err.item())


if __name__ == "__main__":
    unittest.main()
