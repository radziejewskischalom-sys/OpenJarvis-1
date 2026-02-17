"""Tests for configuration, hardware detection, and engine recommendation."""

from __future__ import annotations

from pathlib import Path

from openjarvis.core.config import (
    EngineConfig,
    GpuInfo,
    HardwareInfo,
    JarvisConfig,
    generate_default_toml,
    load_config,
    recommend_engine,
)


class TestDefaults:
    def test_jarvis_config_defaults(self) -> None:
        cfg = JarvisConfig()
        assert cfg.engine.default == "ollama"
        assert cfg.memory.default_backend == "sqlite"
        assert cfg.telemetry.enabled is True

    def test_engine_config_defaults(self) -> None:
        ec = EngineConfig()
        assert ec.ollama_host == "http://localhost:11434"
        assert ec.vllm_host == "http://localhost:8000"


class TestRecommendEngine:
    def test_no_gpu(self) -> None:
        hw = HardwareInfo(platform="linux")
        assert recommend_engine(hw) == "llamacpp"

    def test_apple_silicon(self) -> None:
        hw = HardwareInfo(
            platform="darwin",
            gpu=GpuInfo(vendor="apple", name="Apple M2 Max"),
        )
        assert recommend_engine(hw) == "ollama"

    def test_nvidia_datacenter(self) -> None:
        hw = HardwareInfo(
            platform="linux",
            gpu=GpuInfo(vendor="nvidia", name="NVIDIA A100-SXM4-80GB", vram_gb=80),
        )
        assert recommend_engine(hw) == "vllm"

    def test_nvidia_consumer(self) -> None:
        hw = HardwareInfo(
            platform="linux",
            gpu=GpuInfo(vendor="nvidia", name="NVIDIA GeForce RTX 4090", vram_gb=24),
        )
        assert recommend_engine(hw) == "ollama"

    def test_amd(self) -> None:
        hw = HardwareInfo(
            platform="linux",
            gpu=GpuInfo(vendor="amd", name="Radeon RX 7900 XTX"),
        )
        assert recommend_engine(hw) == "vllm"


class TestTomlLoading:
    def test_load_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "nonexistent.toml")
        assert isinstance(cfg, JarvisConfig)
        # engine default is derived from detected hardware — just ensure it's a string
        assert isinstance(cfg.engine.default, str)

    def test_load_overrides(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            '[engine]\ndefault = "vllm"\n\n[memory]\ndefault_backend = "faiss"\n'
        )
        cfg = load_config(toml_file)
        assert cfg.engine.default == "vllm"
        assert cfg.memory.default_backend == "faiss"


class TestGenerateToml:
    def test_contains_engine_section(self) -> None:
        hw = HardwareInfo(
            platform="linux",
            cpu_brand="Intel Xeon",
            cpu_count=16,
            ram_gb=64.0,
            gpu=GpuInfo(vendor="nvidia", name="NVIDIA H100", vram_gb=80),
        )
        toml = generate_default_toml(hw)
        assert "[engine]" in toml
        assert 'default = "vllm"' in toml
        assert "H100" in toml
