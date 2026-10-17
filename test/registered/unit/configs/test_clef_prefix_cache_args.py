"""Compatibility checks for Clef prefix-memory caching."""

from types import SimpleNamespace

from sglang.srt.arg_groups.model_hook import (
    _validate_clef_prefix_cache_compatibility,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestClefPrefixCacheCompatibility(CustomTestCase):
    def _args(self, **overrides):
        values = {
            "clef_prefix_cache": True,
            "enable_hierarchical_cache": False,
            "hicache_storage_backend": None,
            "hicache_storage_backend_extra_config": None,
            "enable_lmcache": False,
            "lmcache_config_file": None,
            "enable_unified_memory": False,
            "dp_size": 1,
            "attn_dp_size": 1,
            "enable_dp_attention": False,
            "speculative_algorithm": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_rejects_external_prefix_loaders(self):
        cases = (
            ({"enable_hierarchical_cache": True}, "--enable-hierarchical-cache"),
            ({"hicache_storage_backend": "file"}, "--hicache-storage-backend"),
            (
                {"hicache_storage_backend_extra_config": "{}"},
                "--hicache-storage-backend-extra-config",
            ),
            ({"enable_lmcache": True}, "--enable-lmcache"),
            ({"lmcache_config_file": "lmcache.yaml"}, "--lmcache-config-file"),
        )
        for overrides, option in cases:
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, option):
                    _validate_clef_prefix_cache_compatibility(self._args(**overrides))

    def test_rejects_unified_memory(self):
        with self.assertRaisesRegex(ValueError, "--enable-unified-memory"):
            _validate_clef_prefix_cache_compatibility(
                self._args(enable_unified_memory=True)
            )

    def test_rejects_data_parallel_attention(self):
        cases = (
            {"dp_size": 2},
            {"attn_dp_size": 2},
            {"enable_dp_attention": True},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "attention-data-parallel-size"):
                    _validate_clef_prefix_cache_compatibility(self._args(**overrides))

    def test_rejects_speculative_decoding(self):
        with self.assertRaisesRegex(ValueError, "--speculative-algorithm"):
            _validate_clef_prefix_cache_compatibility(
                self._args(speculative_algorithm="EAGLE3")
            )

    def test_flag_off_keeps_other_memory_options_unchanged(self):
        args = self._args(
            clef_prefix_cache=False,
            enable_hierarchical_cache=True,
            enable_unified_memory=True,
            speculative_algorithm="EAGLE3",
        )
        _validate_clef_prefix_cache_compatibility(args)
