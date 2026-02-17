"""Route handlers for the OpenAI-compatible API server."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from openjarvis.core.types import Message, Role
from openjarvis.server.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    ModelListResponse,
    ModelObject,
    StreamChoice,
    UsageInfo,
)

router = APIRouter()


def _to_messages(chat_messages) -> list[Message]:
    """Convert Pydantic ChatMessage objects to core Message objects."""
    messages = []
    for m in chat_messages:
        role = Role(m.role) if m.role in {r.value for r in Role} else Role.USER
        messages.append(Message(
            role=role,
            content=m.content or "",
            name=m.name,
            tool_call_id=m.tool_call_id,
        ))
    return messages


@router.post("/v1/chat/completions")
async def chat_completions(request_body: ChatCompletionRequest, request: Request):
    """Handle chat completion requests (streaming and non-streaming)."""
    engine = request.app.state.engine
    agent = getattr(request.app.state, "agent", None)
    model = request_body.model

    if request_body.stream:
        return await _handle_stream(engine, model, request_body)

    # Non-streaming: use agent if available, otherwise direct engine call
    if agent is not None:
        return _handle_agent(agent, model, request_body)

    return _handle_direct(engine, model, request_body)


def _handle_direct(
    engine, model: str, req: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Direct engine call without agent."""
    messages = _to_messages(req.messages)
    kwargs: dict[str, Any] = {}
    if req.tools:
        kwargs["tools"] = req.tools
    result = engine.generate(
        messages,
        model=model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        **kwargs,
    )
    content = result.get("content", "")
    usage = result.get("usage", {})

    choice_msg = ChoiceMessage(role="assistant", content=content)
    # Include tool calls if present
    tool_calls = result.get("tool_calls")
    if tool_calls:
        choice_msg.tool_calls = [
            {
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", "{}"),
                },
            }
            for tc in tool_calls
        ]

    return ChatCompletionResponse(
        model=model,
        choices=[Choice(
            message=choice_msg,
            finish_reason=result.get("finish_reason", "stop"),
        )],
        usage=UsageInfo(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        ),
    )


def _handle_agent(
    agent, model: str, req: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Run through agent."""
    from openjarvis.agents._stubs import AgentContext

    # Build context from prior messages
    ctx = AgentContext()
    if len(req.messages) > 1:
        prior = _to_messages(req.messages[:-1])
        for m in prior:
            ctx.conversation.add(m)

    # Last message is the input
    input_text = req.messages[-1].content if req.messages else ""
    result = agent.run(input_text, context=ctx)

    return ChatCompletionResponse(
        model=model,
        choices=[Choice(
            message=ChoiceMessage(role="assistant", content=result.content),
            finish_reason="stop",
        )],
    )


async def _handle_stream(engine, model: str, req: ChatCompletionRequest):
    """Stream response using SSE format."""
    messages = _to_messages(req.messages)
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async def generate():
        # Send role chunk first
        first_chunk = ChatCompletionChunk(
            id=chunk_id,
            model=model,
            choices=[StreamChoice(
                delta=DeltaMessage(role="assistant"),
            )],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        # Stream content
        async for token in engine.stream(
            messages,
            model=model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        ):
            chunk = ChatCompletionChunk(
                id=chunk_id,
                model=model,
                choices=[StreamChoice(
                    delta=DeltaMessage(content=token),
                )],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Send finish chunk
        finish_chunk = ChatCompletionChunk(
            id=chunk_id,
            model=model,
            choices=[StreamChoice(
                delta=DeltaMessage(),
                finish_reason="stop",
            )],
        )
        yield f"data: {finish_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/v1/models")
async def list_models(request: Request) -> ModelListResponse:
    """List available models from the engine."""
    engine = request.app.state.engine
    model_ids = engine.list_models()
    return ModelListResponse(
        data=[ModelObject(id=mid) for mid in model_ids],
    )


@router.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    engine = request.app.state.engine
    healthy = engine.health()
    if not healthy:
        raise HTTPException(status_code=503, detail="Engine unhealthy")
    return {"status": "ok"}


__all__ = ["router"]
