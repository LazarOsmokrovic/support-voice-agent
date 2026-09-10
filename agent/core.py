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


def configure_logging(level: int | None = None) -> None:
    """basicConfig, guarded so importing this module twice doesn't duplicate handlers.

    Defaults to WARNING — a chat REPL shouldn't have the SDK's HTTP request
    logs and our own tool_call logs interleaved with the visible
    conversation. Set the LOG_LEVEL env var (e.g. LOG_LEVEL=INFO or DEBUG)
    to turn that verbosity back on for troubleshooting, or pass a level
    explicitly to override both.
    """
    if level is None:
        level = getattr(logging, os.getenv("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
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
                tool_result, output = await self._run_tool(block)
                tool_calls.append({"name": block.name, "input": block.input, "output": output})
                tool_results.append(tool_result)
            self.messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(
            f"tool-use loop did not terminate after {self.max_tool_iterations} iterations "
            "(possible runaway tool call)"
        )

    async def _run_tool(self, block: Any) -> tuple[dict[str, Any], Any]:
        """Returns (the API-shaped tool_result dict, the tool's raw return
        value — or None if it raised). The raw value is surfaced in
        TurnResult.tool_calls so callers (e.g. Phase 4's escalation tracker)
        can inspect what a tool actually returned, not just that it ran.
        """
        try:
            output = self.tool_executor(block.name, block.input)
            if hasattr(output, "__await__"):
                output = await output
            return {"type": "tool_result", "tool_use_id": block.id, "content": str(output)}, output
        except Exception as exc:
            # Deliberately broad: never let a failing tool crash the loop —
            # the model gets a structured error to react to instead.
            logger.exception("tool_call failed name=%s", block.name)
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"Error: {exc}",
                "is_error": True,
            }, None

    async def _call_api(self):
        """One Messages API call, with the fixed prefix marked cacheable.

        The system prompt and the seven tool schemas come to roughly 3,300
        tokens and are byte-identical on every turn of every call. Without
        cache_control the model reprocesses all of it each time, on top of a
        conversation history that grows with the call — which is why a long
        conversation gets steadily slower to answer, the symptom that
        prompted this.

        Marking the LAST tool caches the whole prefix before it, tools and
        system together, since the cache breakpoint covers everything above
        it. Cache hits are also billed at a fraction of input rate, so this
        cuts cost as well as latency.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
        }
        if self.system:
            kwargs["system"] = self.system
        if self.tools:
            # Copy before mutating: self.tools is shared module state
            # (agent/session.py's TOOLS) and every session would otherwise
            # accumulate cache_control markers on the same dicts.
            tools = [dict(tool) for tool in self.tools]
            tools[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = tools
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
