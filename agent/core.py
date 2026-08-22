"""Claude client + tool-use loop (the "brain").

This is the only module that talks to the Claude Messages API. It must stay
completely decoupled from whatever I/O layer is driving it (text CLI, local
voice, Pipecat, Twilio — see transport/) — see CLAUDE.md rule 5. Phase 0 wires
up the loop with zero tools; later phases register tools via the `tools` /
`tool_executor` constructor args without changing anything in here.

The loop is written by hand (not the SDK's beta tool_runner) so the exact
request/response shape stays visible and controllable — see PROJECT_PLAN.md
Phase 0: "a thin wrapper around the Claude Messages API that runs the
standard tool-use loop".
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("agent.core")

# Override with the ANTHROPIC_MODEL env var — e.g. to claude-sonnet-5 for
# lower-latency production voice traffic in later phases — without touching code.
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
DEFAULT_MAX_TOKENS = 1024  # a support-chat turn rarely needs more; raise if replies get cut off


def configure_logging(level: int = logging.INFO) -> None:
    """basicConfig, guarded so importing this module twice doesn't duplicate handlers."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


# A tool executor takes (tool_name, tool_input) and returns a JSON-serializable
# result, or raises — the loop turns a raised exception into an `is_error`
# tool_result rather than letting it escape (Phase 6 needs this: tool failures
# are something the model reasons about, not a crash).
ToolExecutor = Callable[[str, dict[str, Any]], Any] | Callable[[str, dict[str, Any]], Awaitable[Any]]


@dataclass
class TurnResult:
    """What one full turn (user message -> final assistant text) produced."""

    reply: str
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class Agent:
    """One conversation's worth of state plus the tool-use loop.

    Usage:
        agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=dispatch)
        result = await agent.send("What's the status of order 112-3487561-2938471?")
        print(result.reply)

    Create one `Agent` per session/call — `messages` accumulates the full
    conversation history internally, matching the Messages API's stateless
    design (the whole history is resent every turn).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: ToolExecutor | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: anthropic.AsyncAnthropic | None = None,
        max_tool_iterations: int = 8,
    ) -> None:
        if tools and tool_executor is None:
            raise ValueError("tools were provided but no tool_executor to run them")

        self.model = model
        self.system = system
        self.tools = tools or []
        self.tool_executor = tool_executor
        self.max_tokens = max_tokens
        self.client = client or anthropic.AsyncAnthropic()
        self.max_tool_iterations = max_tool_iterations
        self.messages: list[dict[str, Any]] = []

    async def send(self, user_text: str) -> TurnResult:
        """Send one user turn through the tool-use loop and return the final reply.

        send -> if stop_reason == "tool_use": run every requested tool via
        `tool_executor`, feed all results back in a single user message
        (required for parallel tool calls), call again -> repeat until
        stop_reason != "tool_use" (or the iteration cap trips, as a guard
        against a runaway loop).
        """
        self.messages.append({"role": "user", "content": user_text})
        tool_calls: list[dict[str, Any]] = []

        for _ in range(self.max_tool_iterations):
            response = await self._call_api()
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                reply = self._extract_text(response)
                return TurnResult(reply=reply, messages=self.messages, tool_calls=tool_calls)

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_results = []
            for block in tool_use_blocks:
                logger.info("tool_call name=%s input=%s", block.name, block.input)
                tool_calls.append({"name": block.name, "input": block.input})
                tool_results.append(await self._run_tool(block))
            self.messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(
            f"tool-use loop did not terminate after {self.max_tool_iterations} iterations "
            "(possible runaway tool call)"
        )

    async def _run_tool(self, block: Any) -> dict[str, Any]:
        try:
            output = self.tool_executor(block.name, block.input)
            if hasattr(output, "__await__"):
                output = await output
            return {"type": "tool_result", "tool_use_id": block.id, "content": str(output)}
        except Exception as exc:  # noqa: BLE001 — deliberately broad: never let a tool crash the loop
            logger.exception("tool_call failed name=%s", block.name)
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"Error: {exc}",
                "is_error": True,
            }

    async def _call_api(self):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
        }
        if self.system:
            kwargs["system"] = self.system
        if self.tools:
            kwargs["tools"] = self.tools
        return await self.client.messages.create(**kwargs)

    @staticmethod
    def _extract_text(response) -> str:
        return "".join(b.text for b in response.content if b.type == "text")


if __name__ == "__main__":
    # Phase 0 checkpoint (live half): `python -m agent.core` sends "hello" and
    # prints Claude's reply. Requires a real ANTHROPIC_API_KEY in .env.
    import asyncio

    async def _main() -> None:
        configure_logging()
        agent = Agent()
        result = await agent.send("hello")
        print(result.reply)

    asyncio.run(_main())
