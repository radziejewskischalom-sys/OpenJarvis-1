"""Tests for LearningConfig and its integration into JarvisConfig."""

from __future__ import annotations

from pathlib import Path

from openjarvis.core.config import (
    HardwareInfo,
    JarvisConfig,
    LearningConfig,
    generate_default_toml,
    load_config,
)


class TestLearningConfig:
    def test_defaults(self) -> None:
        cfg = LearningConfig()
        assert cfg.default_policy == "heuristic"
        assert cfg.reward_weights == ""

    def test_custom_values(self) -> None:
        cfg = LearningConfig(
            default_policy="grpo",
            reward_weights="latency=0.4,cost=0.3,efficiency=0.3",
        )
        assert cfg.default_policy == "grpo"
        assert cfg.reward_weights == "latency=0.4,cost=0.3,efficiency=0.3"

    def test_jarvis_config_has_learning(self) -> None:
        cfg = JarvisConfig()
        assert hasattr(cfg, "learning")
        assert isinstance(cfg.learning, LearningConfig)
        assert cfg.learning.default_policy == "heuristic"

    def test_toml_loading_with_learning(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            '[learning]\ndefault_policy = "grpo"\n'
            'reward_weights = "latency=0.5"\n'
        )
        cfg = load_config(toml_file)
        assert cfg.learning.default_policy == "grpo"
        assert cfg.learning.reward_weights == "latency=0.5"

    def test_toml_loading_without_learning(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[engine]\n")
        cfg = load_config(toml_file)
        assert cfg.learning.default_policy == "heuristic"

    def test_generate_default_toml_includes_learning(self) -> None:
        hw = HardwareInfo()
        toml_str = generate_default_toml(hw)
        assert "[learning]" in toml_str
        assert 'default_policy = "heuristic"' in toml_str
