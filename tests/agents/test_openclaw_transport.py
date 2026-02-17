"""Tests for OpenClaw transport implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from openjarvis.agents.openclaw_protocol import MessageType, ProtocolMessage
from openjarvis.agents.openclaw_transport import (
    HttpTransport,
    OpenClawTransport,
    SubprocessTransport,
)


class TestOpenClawTransportABC:
    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            OpenClawTransport()

    def test_concrete_required_methods(self):
        class DummyTransport(OpenClawTransport):
            def send(self, msg):
                return msg

            def health(self):
                return True

            def close(self):
                pass

        t = DummyTransport()
        assert t.health() is True


class TestHttpTransport:
    def test_send(self):
        transport = HttpTransport(host="http://localhost:18789")
        msg = ProtocolMessage(type=MessageType.QUERY, content="Hello")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "type": "response",
            "id": "resp-1",
            "content": "World",
        }
        mock_response.status_code = 200

        with patch("httpx.post", return_value=mock_response) as mock_post:
            result = transport.send(msg)
            assert result.type == MessageType.RESPONSE
            assert result.content == "World"
            mock_post.assert_called_once()

    def test_health_ok(self):
        transport = HttpTransport()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp):
            assert transport.health() is True

    def test_health_fail(self):
        transport = HttpTransport()
        with patch("httpx.get", side_effect=Exception("Connection refused")):
            assert transport.health() is False

    def test_close(self):
        transport = HttpTransport()
        transport.close()  # should not raise


class TestSubprocessTransport:
    def test_send(self):
        transport = SubprocessTransport(node_path="node", script_path="test.js")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.return_value = (
            '{"type": "response", "id": "r1", "content": "reply"}\n'
        )

        transport._process = mock_proc

        msg = ProtocolMessage(type=MessageType.QUERY, content="test")
        result = transport.send(msg)
        assert result.type == MessageType.RESPONSE
        assert result.content == "reply"

    def test_health_fail_no_node(self):
        transport = SubprocessTransport(node_path="/nonexistent/node")
        assert transport.health() is False

    def test_close(self):
        transport = SubprocessTransport()
        mock_proc = MagicMock()
        transport._process = mock_proc
        transport.close()
        mock_proc.terminate.assert_called_once()
        assert transport._process is None

    def test_close_no_process(self):
        transport = SubprocessTransport()
        transport.close()  # should not raise
