"""Transport abstraction for OpenClaw agent communication."""

from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from typing import Optional

from openjarvis.agents.openclaw_protocol import (
    MessageType,
    ProtocolMessage,
    deserialize,
    serialize,
)


class OpenClawTransport(ABC):
    """Base class for OpenClaw transport implementations."""

    @abstractmethod
    def send(self, msg: ProtocolMessage) -> ProtocolMessage:
        """Send a message and return the response."""

    @abstractmethod
    def health(self) -> bool:
        """Return True if the transport endpoint is healthy."""

    @abstractmethod
    def close(self) -> None:
        """Release transport resources."""


class HttpTransport(OpenClawTransport):
    """HTTP-based transport for communicating with an OpenClaw server."""

    def __init__(
        self,
        host: str = "http://localhost:18789",
        timeout: float = 30.0,
    ) -> None:
        self._host = host.rstrip("/")
        self._timeout = timeout

    def send(self, msg: ProtocolMessage) -> ProtocolMessage:
        """POST a message to the OpenClaw HTTP endpoint."""
        import httpx

        payload = json.loads(serialize(msg))
        resp = httpx.post(
            f"{self._host}/api/query",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return deserialize(json.dumps(resp.json()))

    def health(self) -> bool:
        """GET /health and return True if status is 200."""
        import httpx

        try:
            resp = httpx.get(
                f"{self._host}/health",
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """No persistent resources to release."""


class SubprocessTransport(OpenClawTransport):
    """Subprocess-based transport — launches a Node.js process and
    communicates via JSON over stdin/stdout."""

    def __init__(
        self,
        node_path: str = "node",
        script_path: str = "",
    ) -> None:
        self._node_path = node_path
        self._script_path = script_path
        self._process: Optional[subprocess.Popen] = None

    def _start_process(self) -> None:
        """Start the Node.js subprocess."""
        self._process = subprocess.Popen(
            [self._node_path, self._script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _ensure_alive(self) -> None:
        """Start the process if it's not running."""
        if self._process is None or self._process.poll() is not None:
            self._start_process()

    def send(self, msg: ProtocolMessage) -> ProtocolMessage:
        """Send a message via stdin and read the response from stdout."""
        self._ensure_alive()
        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None

        line = serialize(msg) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

        response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from OpenClaw subprocess")
        return deserialize(response_line.strip())

    def health(self) -> bool:
        """Check if the subprocess is alive and responsive."""
        try:
            self._ensure_alive()
            health_msg = ProtocolMessage(type=MessageType.HEALTH)
            resp = self.send(health_msg)
            return resp.type == MessageType.HEALTH_OK
        except Exception:
            return False

    def close(self) -> None:
        """Terminate the subprocess."""
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None


__all__ = ["HttpTransport", "OpenClawTransport", "SubprocessTransport"]
