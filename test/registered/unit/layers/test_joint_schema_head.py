"""Unit tests for the joint schema head of Clef checkpoints: decision layouts, and
the batched forward against a per-request loop over the stock torch modules."""

import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.layers.joint_schema_head import (
    JointSchemaHead,
    JointSchemaPooler,
    LayoutQuestion,
    _groups,
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


def _origin_main_forward_batch(head, hidden, input_ids, lengths, layouts, lm_head):
    outputs = []
    start = 0
    for group in _groups(lengths, layouts):
        group_lengths = [lengths[i] for i in group]
        end = start + sum(group_lengths)
        outputs += head._forward_group(
            hidden[start:end],
            input_ids[start:end],
            group_lengths,
            [layouts[i] for i in group],
            lm_head,
        )
        start = end
    return outputs


class TestJointSchemaHeadPrefixMemory(CustomTestCase):
    def test_t1_split_prefix_matches_full_path(self):
        torch.manual_seed(13)
        hidden_size, vocab, length, boundary = 16, 64, 40, 8
        questions = _questions()
        layout = pack_decision_layout(length, questions)
        hidden = torch.randn(length, hidden_size)
        input_ids = torch.randint(vocab, (length,))
        lm_head = torch.randn(vocab, hidden_size)
        head = JointSchemaHead(
            hidden_size, width=8, routing_layers=1, layers=1, heads=2, feedforward=16
        ).eval()

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
        torch.testing.assert_close(full, split, atol=1e-5, rtol=0)

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
        print(f"T1 BF16 maximum absolute difference: {max_abs_diff:.8g}")
        self.assertLess(max_abs_diff, 1e-2)

    def test_t2_mixed_prefix_batch_matches_full_requests(self):
        torch.manual_seed(17)
        hidden_size, vocab, length = 16, 64, 40
        head = JointSchemaHead(
            hidden_size, width=8, routing_layers=1, layers=1, heads=2, feedforward=16
        ).eval()
        lm_head = torch.randn(vocab, hidden_size)
        full_hidden = [torch.randn(length, hidden_size) for _ in range(3)]
        full_ids = [torch.randint(vocab, (length,)) for _ in range(3)]
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
                actual_logits, expected_logits, atol=1e-5, rtol=0
            )

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
