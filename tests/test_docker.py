"""Tests for Docker and deployment files."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestDockerFiles:
    def test_dockerfile_exists(self):
        assert (ROOT / "Dockerfile").is_file()

    def test_dockerfile_gpu_exists(self):
        assert (ROOT / "Dockerfile.gpu").is_file()

    def test_dockerfile_has_entrypoint(self):
        content = (ROOT / "Dockerfile").read_text()
        assert "ENTRYPOINT" in content
        assert "jarvis" in content

    def test_docker_compose_valid_yaml(self):
        import importlib

        yaml_mod = None
        try:
            yaml_mod = importlib.import_module("yaml")
        except ImportError:
            pass

        compose_path = ROOT / "docker-compose.yml"
        assert compose_path.is_file()
        content = compose_path.read_text()

        # Basic structural checks without requiring PyYAML
        assert "services:" in content
        assert "jarvis:" in content

        if yaml_mod is not None:
            data = yaml_mod.safe_load(content)
            assert "services" in data

    def test_docker_compose_has_services(self):
        content = (ROOT / "docker-compose.yml").read_text()
        assert "jarvis:" in content
        assert "ollama:" in content

    def test_systemd_service_exists(self):
        assert (ROOT / "deploy" / "systemd" / "openjarvis.service").is_file()
