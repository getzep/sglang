"""Unit tests for the joint schema head of Clef checkpoints: decision layouts, and
the batched forward against a per-request loop over the stock torch modules."""

import math
import unittest
from types import SimpleNamespace
from typing import List, Optional, Tuple
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.joint_schema_head import (
    JointSchemaHead,
    JointSchemaPooler,
    LayoutQuestion,
    compute_clef_cache_boundary,
    pack_decision_layout,
    parse_decision_layout,
    validate_clef_lm_head,
)
from sglang.srt.layers.quantization.unquant import (
    UnquantizedEmbeddingMethod,
    UnquantizedLinearMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _questions(offset=0):
    return [
        LayoutQuestion(
            1,
            (10 + offset, 14 + offset),
            (
                (16 + offset, 17 + offset),
                (17 + offset, 19 + offset),
                (19 + offset, 21 + offset),
            ),
        ),
        LayoutQuestion(
            0,
            (22 + offset, 25 + offset),
            ((26 + offset, 28 + offset), (29 + offset, 31 + offset)),
        ),
    ]


class TestDecisionLayout(CustomTestCase):
    def test_rejects_layouts_that_do_not_fit_the_prompt(self):
        layout = pack_decision_layout(40, _questions())
        invalid = {
            "shorter prompt": (layout, 39),
            "not ints": ([float(value) for value in layout], 40),
            "truncated": (layout[:-1], 40),
            "trailing values": (layout + [0], 40),
            "no questions": ([40, 0], 40),
            "empty span": (
                pack_decision_layout(40, [LayoutQuestion(0, (5, 5), ((6, 7),))]),
                40,
            ),
            "past the prompt": (
                pack_decision_layout(40, [LayoutQuestion(0, (5, 6), ((6, 41),))]),
                40,
            ),
            "unknown type": (
                pack_decision_layout(40, [LayoutQuestion(3, (5, 6), ((6, 7),))]),
                40,
            ),
        }
        for name, (values, num_tokens) in invalid.items():
            with self.subTest(name), self.assertRaises(ValueError):
                parse_decision_layout(values, num_tokens)


class _RecordingHead:
    """Stands in for the head, scoring each option with its request's length."""

    def __init__(self):
        self.calls = []
        self.prefix_memory = []

    def project_memory(self, hidden):
        return hidden[:, :2]

    def forward_batch(
        self,
        hidden,
        input_ids,
        lengths,
        layouts,
        embedding_weight,
        prefix_memory=None,
        extend_memory=None,
    ):
        self.calls.append(lengths)
        self.prefix_memory.append(prefix_memory)
        return [
            torch.full((sum(len(q.option_spans) for q in layout),), float(length))
            for length, layout in zip(lengths, layouts)
        ]


class TestJointSchemaPooler(CustomTestCase):
    def test_a_layout_cut_from_its_prompt_is_skipped_and_neighbors_are_scored(self):
        """A prompt shorter than its layout, as after a truncation, must not stop the
        scheduler or the other requests of its batch."""
        head = _RecordingHead()
        pooler = JointSchemaPooler(head, lambda: torch.zeros(1))
        layout = pack_decision_layout(40, _questions())
        lengths = [40, 35, 40]
        forward_batch = SimpleNamespace(
            decision_layouts=[layout, layout, layout],
            extend_seq_lens_cpu=lengths,
            extend_prefix_lens_cpu=[0, 0, 0],
            batch_size=3,
            input_ids=torch.zeros(sum(lengths), dtype=torch.long),
        )
        output = pooler(torch.zeros(sum(lengths), 8), forward_batch)
        self.assertEqual([len(scores) for scores in output.embeddings], [5, 0, 5])
        self.assertEqual(head.calls, [[40], [40]])

    def test_a_span_before_one_prefix_skips_only_that_request(self):
        class FakePool:
            def __init__(self):
                self.rows = torch.zeros(512, 2)

            def store_clef_memory(self, locations, rows):
                self.rows[locations] = rows

            def gather_clef_memory(self, locations):
                return self.rows[locations]

        torch.manual_seed(23)
        head = _RecordingHead()
        pool = FakePool()
        pooler = JointSchemaPooler(head, lambda: torch.zeros(64, 2))
        full_hidden = [torch.randn(40, 4) for _ in range(3)]
        full_ids = [torch.zeros(40, dtype=torch.long) for _ in range(3)]
        prefixes = [0, 12, 4]
        lengths = [40 - prefix for prefix in prefixes]
        layout = pack_decision_layout(40, _questions())

        prefix_locations = torch.arange(120, 124)
        pool.store_clef_memory(
            prefix_locations, head.project_memory(full_hidden[2][:4])
        )
        req_to_token = torch.zeros((3, 40), dtype=torch.long)
        req_to_token[2, :4] = prefix_locations
        extend_hidden = torch.cat(
            [hidden[prefix:] for hidden, prefix in zip(full_hidden, prefixes)]
        )
        extend_ids = torch.cat(
            [ids[prefix:] for ids, prefix in zip(full_ids, prefixes)]
        )
        forward_batch = SimpleNamespace(
            clef_prefix_cache=True,
            token_to_kv_pool=pool,
            req_to_token=req_to_token,
            req_pool_indices=torch.arange(3),
            out_cache_loc=torch.arange(sum(lengths)),
            decision_layouts=[layout, layout, layout],
            extend_seq_lens_cpu=lengths,
            extend_prefix_lens_cpu=prefixes,
            clef_cache_boundaries=[16, 16, 16],
            batch_size=3,
            input_ids=extend_ids,
            rids=["no-prefix", "bad-span", "valid-prefix"],
        )

        with self.assertLogs(
            "sglang.srt.layers.joint_schema_head", level="WARNING"
        ) as warnings:
            mixed = pooler(extend_hidden, forward_batch)
        self.assertEqual(len(warnings.output), 1)
        self.assertIn("bad-span", warnings.output[0])
        self.assertIn("prefix length 12", warnings.output[0])
        self.assertIn("decision span start 10", warnings.output[0])
        self.assertEqual(mixed.embeddings[1].numel(), 0)
        self.assertEqual(head.calls[:2], [[40], [36]])

        def run_alone(index):
            prefix = prefixes[index]
            length = lengths[index]
            start = 300 + index * 40
            single_batch = SimpleNamespace(
                clef_prefix_cache=True,
                token_to_kv_pool=pool,
                req_to_token=req_to_token,
                req_pool_indices=torch.tensor([index]),
                out_cache_loc=torch.arange(start, start + length),
                decision_layouts=[layout],
                extend_seq_lens_cpu=[length],
                extend_prefix_lens_cpu=[prefix],
                clef_cache_boundaries=[16],
                batch_size=1,
                input_ids=full_ids[index][prefix:],
                rids=[forward_batch.rids[index]],
            )
            return pooler(full_hidden[index][prefix:], single_batch).embeddings[0]

        self.assertTrue(torch.equal(mixed.embeddings[0], run_alone(0)))
        self.assertTrue(torch.equal(mixed.embeddings[2], run_alone(2)))


def _reference_logits(head, hidden, input_ids, questions, lm_head):
    """One request through the modules' own forward methods, as the checkpoint's loop does."""
    normalized = head.hidden_norm(hidden)
    memory = head.memory_projection(normalized).unsqueeze(0)
    global_vector = normalized[-1]

    def mean(values, span):
        return values[span[0] : span[1]].mean(dim=0)

    question_vectors = torch.stack(
        [mean(normalized, q.question_span) for q in questions]
    )
    queries, lexicals = [], []
    for index, question in enumerate(questions):
        context = torch.stack(
            [mean(normalized, span) for span in question.option_spans]
        )
        lexical = torch.stack(
            [
                lm_head[input_ids[start:end]].mean(dim=0)
                for start, end in question.option_spans
            ]
        )
        queries.append(
            head.option_context_projection(context)
            + head.option_lexical_projection(lexical)
            + head.option_question_projection(question_vectors[index]).unsqueeze(0)
        )
        lexicals.append(lexical)
    routed = torch.cat(queries).unsqueeze(0)
    for layer in head.evidence_layers:
        attended, _ = layer.attention(
            layer.query_norm(routed),
            layer.memory_norm(memory),
            layer.memory_norm(memory),
            need_weights=False,
        )
        routed = routed + attended
        routed = routed + layer.feedforward(layer.feedforward_norm(routed))
    options = list(torch.split(routed[0], [len(q.option_spans) for q in questions]))
    base_fields = head.question_projection(question_vectors)
    summaries = []
    for field, routed_options in zip(base_fields, options):
        weights = torch.softmax(
            routed_options @ field / math.sqrt(field.shape[-1]), dim=0
        )
        summaries.append((weights.unsqueeze(-1) * routed_options).sum(dim=0))
    fields = (
        base_fields
        + head.option_summary_norm(torch.stack(summaries))
        + head.global_projection(global_vector).unsqueeze(0)
        + head.type_embedding(
            torch.tensor([q.question_type for q in questions], device=hidden.device)
        )
    ).unsqueeze(0)
    for layer in head.layers:
        fields = layer(fields, memory)
    fields = head.field_norm(fields[0])
    prior_scale = head.prior_logit_scale.clamp(max=math.log(100.0)).exp()
    joint_scale = head.joint_logit_scale.clamp(max=math.log(100.0)).exp()
    logits = []
    for index, (field, lexical, routed_options) in enumerate(
        zip(fields, lexicals, options)
    ):
        anchor = F.normalize(question_vectors[index] + global_vector, dim=-1)
        prior = prior_scale * (F.normalize(lexical, dim=-1) @ anchor)
        normed = head.option_norm(routed_options)
        repeated = field.unsqueeze(0).expand_as(normed)
        features = torch.cat(
            [repeated, normed, repeated * normed, torch.abs(repeated - normed)], dim=-1
        )
        residual = head.residual_scorer(features).squeeze(-1)
        cosine = F.cosine_similarity(repeated, normed, dim=-1)
        logits.append(
            prior
            + torch.sigmoid(head.residual_gate) * (joint_scale * cosine + residual)
        )
    return torch.cat(logits).float()


def _shift_layout(questions, prefix):
    return [
        LayoutQuestion(
            question.question_type,
            (
                question.question_span[0] - prefix,
                question.question_span[1] - prefix,
            ),
            tuple(
                (start - prefix, end - prefix) for start, end in question.option_spans
            ),
        )
        for question in questions
    ]


_MAX_GROUP_TOKENS = 65536
_MAX_GROUP_SLOTS = 1 << 18


def _attention(
    module: nn.MultiheadAttention,
    queries: torch.Tensor,
    keys: Optional[torch.Tensor],
    query_bounds: Tuple[torch.Tensor, int],
    key_bounds: Tuple[torch.Tensor, int],
) -> torch.Tensor:
    """nn.MultiheadAttention over packed sequences, keys and values from ``keys``
    (or from ``queries`` when None), attending within each request."""
    width = queries.shape[-1]
    heads = module.num_heads
    if keys is None:
        query, key, value = F.linear(
            queries, module.in_proj_weight, module.in_proj_bias
        ).chunk(3, dim=-1)
    else:
        query_weight, key_value_weight = module.in_proj_weight.split([width, 2 * width])
        query_bias, key_value_bias = module.in_proj_bias.split([width, 2 * width])
        query = F.linear(queries, query_weight, query_bias)
        key, value = F.linear(keys, key_value_weight, key_value_bias).chunk(2, dim=-1)
    head_dim = width // heads
    output = torch.ops.aten._flash_attention_forward(
        query.reshape(-1, heads, head_dim),
        key.reshape(-1, heads, head_dim),
        value.reshape(-1, heads, head_dim),
        query_bounds[0],
        key_bounds[0],
        query_bounds[1],
        key_bounds[1],
        0.0,
        False,
        False,
    )[0]
    return module.out_proj(output.reshape(-1, width))


def _cpu_attention(
    module: nn.MultiheadAttention,
    queries: torch.Tensor,
    keys: Optional[torch.Tensor],
    query_bounds: Tuple[torch.Tensor, int],
    key_bounds: Tuple[torch.Tensor, int],
) -> torch.Tensor:
    query_cu = query_bounds[0].tolist()
    key_cu = key_bounds[0].tolist()
    output = []
    for index in range(len(query_cu) - 1):
        query = queries[query_cu[index] : query_cu[index + 1]].unsqueeze(0)
        if keys is None:
            key = query
        else:
            key = keys[key_cu[index] : key_cu[index + 1]].unsqueeze(0)
        attended, _ = module(query, key, key, need_weights=False)
        output.append(attended.squeeze(0))
    return torch.cat(output, dim=0)


def _groups(lengths: List[int], layouts: List[List[LayoutQuestion]]) -> List[range]:
    """Consecutive requests per head pass, bounded in tokens and padded option slots."""
    groups, start, tokens, questions, widest = [], 0, 0, 0, 0
    for index, (length, layout) in enumerate(zip(lengths, layouts)):
        layout_widest = max(len(question.option_spans) for question in layout)
        grown = (questions + len(layout)) * max(widest, layout_widest)
        if index > start and (
            tokens + length > _MAX_GROUP_TOKENS or grown > _MAX_GROUP_SLOTS
        ):
            groups.append(range(start, index))
            start, tokens, questions, widest = index, 0, 0, 0
        tokens += length
        questions += len(layout)
        widest = max(widest, layout_widest)
    groups.append(range(start, len(lengths)))
    return groups


class _OriginMainJointSchemaHead:
    def _forward_group(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        lengths: List[int],
        layouts: List[List[LayoutQuestion]],
        output_embedding_weight: torch.Tensor,
    ) -> List[torch.Tensor]:
        device, dtype = hidden.device, hidden.dtype
        batch = len(layouts)
        question_bounds, option_bounds = [], []
        type_ids, owner, slot, question_request, last_tokens = [], [], [], [], []
        regions, region_offsets = [], [0]
        token_cu, question_cu, option_cu, option_counts = [0], [0], [0], []
        for request, (length, questions) in enumerate(zip(lengths, layouts)):
            offset = token_cu[-1]
            low = min(min(q.question_span[0], q.option_spans[0][0]) for q in questions)
            high = max(
                max(q.question_span[1], *(e for _, e in q.option_spans))
                for q in questions
            )
            # Prefix row of span position p in this request's schema region.
            base = region_offsets[-1] + request - low
            regions.append((offset + low, offset + high))
            request_options = 0
            for question in questions:
                count = len(question.option_spans)
                question_bounds += (
                    base + question.question_span[0],
                    base + question.question_span[1],
                )
                for option_slot, (option_start, option_end) in enumerate(
                    question.option_spans
                ):
                    option_bounds += (base + option_start, base + option_end)
                    owner.append(len(type_ids))
                    slot.append(option_slot)
                type_ids.append(question.question_type)
                question_request.append(request)
                request_options += count
            last_tokens.append(offset + length - 1)
            token_cu.append(offset + length)
            region_offsets.append(region_offsets[-1] + high - low)
            question_cu.append(len(type_ids))
            option_cu.append(option_cu[-1] + request_options)
            option_counts.append(request_options)

        num_questions = len(type_ids)
        widest = max(len(q.option_spans) for questions in layouts for q in questions)
        sizes = [
            2 * num_questions,
            len(option_bounds),
            num_questions,
            len(owner),
            len(slot),
            num_questions,
            batch,
            3 * (batch + 1),
        ]
        packed = torch.tensor(
            question_bounds
            + option_bounds
            + type_ids
            + owner
            + slot
            + question_request
            + last_tokens
            + token_cu
            + question_cu
            + option_cu,
            dtype=torch.long,
        ).to(device, non_blocking=True)
        (
            question_bounds_t,
            option_bounds_t,
            type_ids_t,
            owner_t,
            slot_t,
            question_request_t,
            last_tokens_t,
            cu_t,
        ) = torch.split(packed, sizes)
        question_bounds_t = question_bounds_t.view(-1, 2)
        option_bounds_t = option_bounds_t.view(-1, 2)
        token_cu_t, question_cu_t, option_cu_t = (
            cu_t.to(torch.int32).view(3, batch + 1).unbind(0)
        )
        token_bounds = (token_cu_t, max(lengths))
        question_bounds_cu = (question_cu_t, max(len(q) for q in layouts))
        option_bounds_cu = (option_cu_t, max(option_counts))

        normalized = self.hidden_norm(hidden)
        # Running sums over each schema region, after one zero row per request.
        region_total = region_offsets[-1] + batch
        hidden_prefix = torch.zeros(
            region_total, normalized.shape[-1], dtype=torch.float32, device=device
        )
        lexical_prefix = torch.zeros(
            region_total,
            output_embedding_weight.shape[-1],
            dtype=torch.float32,
            device=device,
        )
        for request, (low, high) in enumerate(regions):
            row = region_offsets[request] + request + 1
            torch.cumsum(
                normalized[low:high].float(),
                dim=0,
                out=hidden_prefix[row : row + high - low],
            )
            torch.cumsum(
                output_embedding_weight[input_ids[low:high]].float(),
                dim=0,
                out=lexical_prefix[row : row + high - low],
            )

        def span_means(prefix: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
            sums = prefix[bounds[:, 1]] - prefix[bounds[:, 0]]
            return (sums / (bounds[:, 1] - bounds[:, 0]).unsqueeze(-1)).to(dtype)

        memory = self.memory_projection(normalized)
        global_vectors = normalized[last_tokens_t]
        question_vectors = span_means(hidden_prefix, question_bounds_t)
        option_contexts = span_means(hidden_prefix, option_bounds_t)
        lexical = span_means(lexical_prefix, option_bounds_t)

        routed = (
            self.option_context_projection(option_contexts)
            + self.option_lexical_projection(lexical)
            + self.option_question_projection(question_vectors)[owner_t]
        )
        for layer in self.evidence_layers:
            routed = routed + _attention(
                layer.attention,
                layer.query_norm(routed),
                layer.memory_norm(memory),
                option_bounds_cu,
                token_bounds,
            )
            routed = routed + layer.feedforward(layer.feedforward_norm(routed))

        base_fields = self.question_projection(question_vectors)
        scores = torch.matmul(
            routed.unsqueeze(1), base_fields[owner_t].unsqueeze(2)
        ).view(-1) / math.sqrt(routed.shape[-1])
        padded_scores = torch.full(
            (num_questions, widest), float("-inf"), dtype=torch.float32, device=device
        )
        padded_scores[owner_t, slot_t] = scores.float()
        weights = torch.softmax(padded_scores, dim=1).to(dtype)
        padded_options = routed.new_zeros(num_questions, widest, routed.shape[-1])
        padded_options[owner_t, slot_t] = routed
        summaries = torch.sum(weights.unsqueeze(-1) * padded_options, dim=1)
        fields = (
            base_fields
            + self.option_summary_norm(summaries)
            + self.global_projection(global_vectors)[question_request_t]
            + self.type_embedding(type_ids_t)
        )
        for layer in self.layers:
            fields = fields + _attention(
                layer.self_attn,
                layer.norm1(fields),
                None,
                question_bounds_cu,
                question_bounds_cu,
            )
            fields = fields + _attention(
                layer.multihead_attn,
                layer.norm2(fields),
                memory,
                question_bounds_cu,
                token_bounds,
            )
            fields = fields + layer.linear2(
                layer.activation(layer.linear1(layer.norm3(fields)))
            )
        fields = self.field_norm(fields)

        prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
        anchor = F.normalize(
            question_vectors + global_vectors[question_request_t], dim=-1
        )
        prior = prior_scale * torch.matmul(
            F.normalize(lexical, dim=-1).unsqueeze(1), anchor[owner_t].unsqueeze(2)
        ).view(-1)
        options = self.option_norm(routed)
        repeated = fields[owner_t]
        cosine = F.cosine_similarity(repeated, options, dim=-1)
        features = torch.cat(
            [repeated, options, repeated * options, torch.abs(repeated - options)],
            dim=-1,
        )
        residual = self.residual_scorer(features).squeeze(-1)
        logits = prior + torch.sigmoid(self.residual_gate) * (
            joint_scale * cosine + residual
        )
        return list(torch.split(logits.float(), option_counts))


def _origin_main_forward_batch(head, hidden, input_ids, lengths, layouts, lm_head):
    outputs = []
    start = 0
    for group in _groups(lengths, layouts):
        group_lengths = [lengths[i] for i in group]
        end = start + sum(group_lengths)
        outputs += _OriginMainJointSchemaHead._forward_group(
            head,
            hidden[start:end],
            input_ids[start:end],
            group_lengths,
            [layouts[i] for i in group],
            lm_head,
        )
        start = end
    return outputs


class TestJointSchemaHeadPrefixMemory(CustomTestCase):
    def _run_t1(self, device):
        torch.manual_seed(13)
        hidden_size, vocab, length, boundary = 16, 64, 40, 8
        input_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
        questions = _questions()
        layout = pack_decision_layout(length, questions)
        hidden = torch.randn(length, hidden_size, device=device, dtype=input_dtype)
        input_ids = torch.randint(vocab, (length,), device=device)
        lm_head = torch.randn(vocab, hidden_size, device=device, dtype=input_dtype)
        head = (
            JointSchemaHead(
                hidden_size,
                width=8,
                routing_layers=1,
                layers=1,
                heads=2,
                feedforward=16,
            )
            .to(device=device, dtype=input_dtype)
            .eval()
        )

        with torch.no_grad():
            full = head.forward_batch(
                hidden, input_ids, [length], [questions], lm_head
            )[0]
            prefix_memory = head.project_memory(hidden[:boundary])
            split = head.forward_batch(
                hidden[boundary:],
                input_ids[boundary:],
                [length - boundary],
                [_shift_layout(questions, boundary)],
                lm_head,
                prefix_memory=[prefix_memory],
            )[0]
        tolerance = 1e-5 if input_dtype == torch.float32 else 1e-2
        torch.testing.assert_close(full, split, atol=tolerance, rtol=0)

        bf16_head = head.to(torch.bfloat16)
        bf16_hidden = hidden.to(torch.bfloat16)
        bf16_lm_head = lm_head.to(torch.bfloat16)
        with torch.no_grad():
            bf16_full = bf16_head.forward_batch(
                bf16_hidden,
                input_ids,
                [length],
                [parse_decision_layout(layout, length)],
                bf16_lm_head,
            )[0]
            bf16_prefix = bf16_head.project_memory(bf16_hidden[:boundary])
            bf16_split = bf16_head.forward_batch(
                bf16_hidden[boundary:],
                input_ids[boundary:],
                [length - boundary],
                [_shift_layout(questions, boundary)],
                bf16_lm_head,
                prefix_memory=[bf16_prefix],
            )[0]
        max_abs_diff = (bf16_full.float() - bf16_split.float()).abs().max().item()
        print(f"T1 {device} BF16 maximum absolute difference: {max_abs_diff:.8g}")
        self.assertLess(max_abs_diff, 1e-2)

    def test_t1_split_prefix_matches_full_path(self):
        self._run_t1(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_t1_split_prefix_matches_full_path_cuda(self):
        self._run_t1(torch.device("cuda"))

    def _run_t2(self, device):
        torch.manual_seed(17)
        hidden_size, vocab, length = 16, 64, 40
        input_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
        head = (
            JointSchemaHead(
                hidden_size,
                width=8,
                routing_layers=1,
                layers=1,
                heads=2,
                feedforward=16,
            )
            .to(device=device, dtype=input_dtype)
            .eval()
        )
        lm_head = torch.randn(vocab, hidden_size, device=device, dtype=input_dtype)
        full_hidden = [
            torch.randn(length, hidden_size, device=device, dtype=input_dtype)
            for _ in range(3)
        ]
        full_ids = [torch.randint(vocab, (length,), device=device) for _ in range(3)]
        questions = [_questions() for _ in range(3)]
        boundaries = [0, 5, 8]

        with torch.no_grad():
            expected = [
                head.forward_batch(h, ids, [length], [layout], lm_head)[0]
                for h, ids, layout in zip(full_hidden, full_ids, questions)
            ]
            extended_hidden = torch.cat(
                [full_hidden[i][boundaries[i] :] for i in range(3)]
            )
            extended_ids = torch.cat([full_ids[i][boundaries[i] :] for i in range(3)])
            prefix_memory = [
                None if boundary == 0 else head.project_memory(hidden[:boundary])
                for hidden, boundary in zip(full_hidden, boundaries)
            ]
            actual = head.forward_batch(
                extended_hidden,
                extended_ids,
                [length - boundary for boundary in boundaries],
                [
                    _shift_layout(layout, boundary)
                    for layout, boundary in zip(questions, boundaries)
                ],
                lm_head,
                prefix_memory=prefix_memory,
            )
        for expected_logits, actual_logits in zip(expected, actual):
            torch.testing.assert_close(
                actual_logits,
                expected_logits,
                atol=1e-5 if input_dtype == torch.float32 else 1e-2,
                rtol=0,
            )

    def test_t2_mixed_prefix_batch_matches_full_requests(self):
        self._run_t2(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_t2_mixed_prefix_batch_matches_full_requests_cuda(self):
        self._run_t2(torch.device("cuda"))

    def test_t3_no_prefix_is_equal_to_origin_main_forward_batch(self):
        torch.manual_seed(19)
        hidden_size, vocab = 16, 64
        head = JointSchemaHead(
            hidden_size, width=8, routing_layers=1, layers=1, heads=2, feedforward=16
        ).eval()
        lengths = [40, 32]
        layouts = [_questions(), _questions()]
        hidden = torch.randn(sum(lengths), hidden_size)
        input_ids = torch.randint(vocab, (sum(lengths),))
        lm_head = torch.randn(vocab, hidden_size)
        with torch.no_grad():
            actual = head.forward_batch(hidden, input_ids, lengths, layouts, lm_head)
            if hidden.device.type == "cpu":
                with patch(__name__ + "._attention", _cpu_attention):
                    expected = _origin_main_forward_batch(
                        head, hidden, input_ids, lengths, layouts, lm_head
                    )
            else:
                expected = _origin_main_forward_batch(
                    head, hidden, input_ids, lengths, layouts, lm_head
                )
        for actual_logits, expected_logits in zip(actual, expected):
            self.assertTrue(torch.equal(actual_logits, expected_logits))

    def test_t4_stores_rows_and_reuses_slots_without_stale_memory(self):
        class FakePool:
            def __init__(self):
                self.rows = torch.zeros(32, 2)

            def store_clef_memory(self, locations, rows):
                self.rows[locations] = rows

            def gather_clef_memory(self, locations):
                return self.rows[locations]

        pool = FakePool()
        head = _RecordingHead()
        pooler = JointSchemaPooler(head, lambda: torch.zeros(64, 2))
        req_to_token = torch.zeros((2, 8), dtype=torch.long)
        req_to_token[0, :3] = torch.tensor([8, 9, 10])
        req_to_token[1, :3] = torch.tensor([8, 9, 10])

        def batch(*, slot, hidden, locations, layout, prefix, boundary):
            return SimpleNamespace(
                clef_prefix_cache=True,
                token_to_kv_pool=pool,
                req_to_token=req_to_token,
                req_pool_indices=torch.tensor([slot]),
                out_cache_loc=torch.tensor(locations),
                decision_layouts=[layout],
                extend_seq_lens_cpu=[hidden.shape[0]],
                extend_prefix_lens_cpu=[prefix],
                clef_cache_boundaries=[boundary],
                batch_size=1,
                input_ids=torch.zeros(hidden.shape[0], dtype=torch.long),
                rids=[f"request-{slot}"],
            )

        first_rows = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pooler(
            first_rows,
            batch(
                slot=0,
                hidden=first_rows,
                locations=[8, 9, 10],
                layout=None,
                prefix=0,
                boundary=None,
            ),
        )
        self.assertTrue(torch.equal(pool.rows[8:11], first_rows))

        second_rows = torch.tensor([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]])
        pooler(
            second_rows,
            batch(
                slot=1,
                hidden=second_rows,
                locations=[8, 9, 10],
                layout=None,
                prefix=0,
                boundary=None,
            ),
        )
        self.assertTrue(torch.equal(pool.rows[8:11], second_rows))

        layout = pack_decision_layout(5, [LayoutQuestion(0, (2, 3), ((3, 4),))])
        extension = torch.tensor([[20.0, 21.0], [22.0, 23.0], [24.0, 25.0]])
        pooler(
            extension,
            batch(
                slot=1,
                hidden=extension,
                locations=[12, 13, 14],
                layout=layout,
                prefix=2,
                boundary=2,
            ),
        )
        self.assertEqual(len(head.prefix_memory), 1)
        self.assertTrue(torch.equal(head.prefix_memory[0][0], second_rows[:2]))

    def test_t5_boundary_alignment_and_partial_span_skip(self):
        layout = pack_decision_layout(100, [LayoutQuestion(0, (20, 22), ((25, 27),))])
        self.assertEqual(compute_clef_cache_boundary(layout, 100, 4, 8), 16)
        self.assertEqual(compute_clef_cache_boundary(layout, 110, 4, 8), 24)

        pool = SimpleNamespace(
            store_clef_memory=lambda locations, rows: None,
            gather_clef_memory=lambda locations: torch.zeros(len(locations), 2),
        )
        pooler = JointSchemaPooler(_RecordingHead(), lambda: torch.zeros(64, 2))
        partial_layout = pack_decision_layout(5, [LayoutQuestion(0, (2, 3), ((3, 4),))])
        forward_batch = SimpleNamespace(
            clef_prefix_cache=True,
            token_to_kv_pool=pool,
            out_cache_loc=torch.tensor([4, 5]),
            decision_layouts=[partial_layout],
            extend_seq_lens_cpu=[2],
            extend_prefix_lens_cpu=[3],
            clef_cache_boundaries=[3],
            batch_size=1,
            input_ids=torch.zeros(2, dtype=torch.long),
            req_pool_indices=torch.tensor([0]),
            req_to_token=torch.zeros((1, 5), dtype=torch.long),
            rids=["partial"],
        )
        output = pooler(torch.randn(2, 2), forward_batch)
        self.assertEqual(output.embeddings[0].numel(), 0)
        forward_batch.decision_layouts = [
            pack_decision_layout(5, [LayoutQuestion(0, (3, 4), ((4, 5),))])
        ]
        forward_batch.clef_cache_boundaries = [2]
        output = pooler(torch.randn(2, 2), forward_batch)
        self.assertEqual(output.embeddings[0].numel(), 0)

    def test_t6_unquantized_lm_head_methods_pass(self):
        validate_clef_lm_head(
            SimpleNamespace(
                quant_method=UnquantizedEmbeddingMethod(),
                weight=torch.empty(1, dtype=torch.bfloat16),
            )
        )
        validate_clef_lm_head(
            SimpleNamespace(
                quant_method=UnquantizedLinearMethod(),
                weight=torch.empty(1, dtype=torch.bfloat16),
            )
        )
        with self.assertRaisesRegex(ValueError, "unquantized"):
            validate_clef_lm_head(
                SimpleNamespace(quant_method=object(), weight=torch.empty(1))
            )
        with self.assertRaisesRegex(ValueError, "floating-point"):
            validate_clef_lm_head(
                SimpleNamespace(
                    quant_method=UnquantizedEmbeddingMethod(),
                    weight=torch.empty(1, dtype=torch.int32),
                )
            )


@unittest.skipUnless(torch.cuda.is_available(), "the batched head uses FlashAttention")
class TestJointSchemaHeadBatch(CustomTestCase):
    def test_batched_forward_matches_each_request_alone(self):
        torch.manual_seed(0)
        hidden_size, vocab = 256, 512
        head = JointSchemaHead(
            hidden_size, width=128, routing_layers=2, layers=2, heads=4, feedforward=256
        )
        for parameter in (
            head.prior_logit_scale,
            head.joint_logit_scale,
            head.residual_gate,
        ):
            parameter.data.fill_(0.5)
        head = head.to("cuda", torch.bfloat16).eval()
        lm_head = torch.randn(vocab, hidden_size, device="cuda", dtype=torch.bfloat16)
        lengths = [40, 140, 64]
        layouts = [
            _questions(),
            _questions(offset=100),
            [
                LayoutQuestion(
                    2, (3, 9), tuple((10 + 2 * i, 12 + 2 * i) for i in range(20))
                )
            ],
        ]
        hidden = torch.randn(
            sum(lengths), hidden_size, device="cuda", dtype=torch.bfloat16
        )
        input_ids = torch.randint(0, vocab, (sum(lengths),), device="cuda")
        with torch.no_grad():
            batched = head.forward_batch(hidden, input_ids, lengths, layouts, lm_head)
            start = 0
            for length, questions, logits in zip(lengths, layouts, batched):
                end = start + length
                expected = _reference_logits(
                    head, hidden[start:end], input_ids[start:end], questions, lm_head
                )
                self.assertEqual(logits.shape, expected.shape)
                torch.testing.assert_close(logits, expected, atol=5e-2, rtol=5e-2)
                torch.testing.assert_close(
                    logits.softmax(-1), expected.softmax(-1), atol=1e-2, rtol=0
                )
                start = end


if __name__ == "__main__":
    unittest.main()
