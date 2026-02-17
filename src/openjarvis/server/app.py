"""FastAPI application factory for the OpenJarvis API server."""

from __future__ import annotations

from fastapi import FastAPI

from openjarvis.server.routes import router


def create_app(
    engine,
    model: str,
    *,
    agent=None,
    bus=None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    engine:
        The inference engine to use for completions.
    model:
        Default model name.
    agent:
        Optional agent instance for agent-mode completions.
    bus:
        Optional event bus for telemetry.
    """
    app = FastAPI(
        title="OpenJarvis API",
        description="OpenAI-compatible API server for OpenJarvis",
        version="1.0.0",
    )

    # Store dependencies in app state
    app.state.engine = engine
    app.state.model = model
    app.state.agent = agent
    app.state.bus = bus

    app.include_router(router)

    return app


__all__ = ["create_app"]
