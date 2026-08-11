"""
llm_client.py

LLM Provider Client Factory.

Responsibility
--------------
Build a chat-completion client for whichever provider config.LLM_PROVIDER
selects, and hand every caller back the SAME shape regardless of which one
that is: `.chat.completions.create(model=..., messages=[...], temperature=,
tools=, tool_choice=)` returning an object with `.choices[0].message.content`
and/or `.choices[0].message.tool_calls[0].function.arguments` (a JSON
string) -- the exact OpenAI-SDK shape every call site in this app
(router.py, verifier.py, and every tools/*.py file) was already written
against, unmodified.

Why a shim instead of touching every call site
-------------------------------------------------
Every one of those ~10 call sites builds its request the same narrow way
(a system message + a user message, an optional single forced tool call or
`tool_choice="auto"`) and parses the response the same narrow way -- but
Anthropic's actual SDK shape is genuinely different at both ends: `system`
is a separate top-level param instead of a role in `messages`, tools are
`{"name","description","input_schema"}` instead of `{"type":"function",
"function":{...}}`, `tool_choice` uses `{"type":"tool","name":...}` instead
of `{"type":"function","function":{"name":...}}`, `max_tokens` is required
(OpenAI's isn't), and a tool call comes back as a `tool_use` content block
whose `.input` is already a dict (OpenAI's `.arguments` is a JSON STRING
every call site already `json.loads()`s). Translating ALL of that here,
once, behind the exact interface every call site already expects, means
switching providers is really just switching config.LLM_PROVIDER -- not a
second, parallel implementation of every tool/prompt in this codebase.

This module DOES NOT:
    ✗ Make any chat completion calls itself
    ✗ Build prompts
    ✗ Decide which provider is active -- see config.LLM_PROVIDER
"""

import json

from .. import config


def get_client():
    """
    Return a client exposing `.chat.completions.create(...)` in the OpenAI
    shape, backed by whichever provider config.LLM_PROVIDER selects. Use
    this for calls whose OUTPUT is what the user actually sees/gets --
    a generated answer, generated code -- where model quality genuinely
    changes what comes out.
    """

    if config.LLM_PROVIDER == "anthropic":
        return _get_anthropic_client()
    return _get_azure_client()


def get_fast_client():
    """
    Same interface as get_client(), but ALWAYS the fast/cheap tier
    (config.FAST_CHAT_MODEL), regardless of config.LLM_PROVIDER. Use this
    for internal routing/classification calls -- "which tool", "is this
    in scope", "SUMMARY or QUESTION mode", "which files need planning" --
    every one a forced single-choice/schema decision, not user-facing
    generated content.

    Why this exists: measured directly, a single repo_summarizer_tool
    query chains 5 sequential LLM calls (scope check -> router ->
    mode-classify -> answer generation -> verifier), and only ONE of
    those five is actual generated content the user reads. Running all
    five through a larger/slower model (e.g. Claude Sonnet, ~3x gpt-4o-
    mini's measured latency per call) multiplies that cost across every
    classification step for zero quality benefit -- these are narrow,
    structured decisions a fast model already handles correctly, so the
    only thing a bigger model adds here is latency. Falls back to
    get_client() if the fast tier's own credentials aren't configured,
    so this is never a HARDER dependency than the main toggle.
    """

    try:
        return _get_azure_client()
    except RuntimeError:
        return get_client()


def _get_azure_client():
    from openai import AzureOpenAI

    if not (config.AZURE_ENDPOINT and config.AZURE_API_KEY):
        raise RuntimeError(
            "Azure OpenAI credentials are missing. Copy .env.example to .env "
            "and fill in your values."
        )
    return AzureOpenAI(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
    )


def _get_anthropic_client():
    from anthropic import Anthropic

    if not (config.ANTHROPIC_API_KEY and config.ANTHROPIC_BASE_URL):
        raise RuntimeError(
            "Anthropic credentials are missing. Set ANTHROPIC_API_KEY and "
            "ANTHROPIC_BASE_URL in .env (LLM_PROVIDER=anthropic is set)."
        )
    real_client = Anthropic(api_key=config.ANTHROPIC_API_KEY, base_url=config.ANTHROPIC_BASE_URL)
    return _AnthropicClientAdapter(real_client)


# ---------------------------------------------------------------------------
# Anthropic -> OpenAI-shape adapter
# ---------------------------------------------------------------------------
# Generous, fixed default -- Anthropic requires max_tokens (OpenAI's calls
# never pass one, relying on the model default) and no call site in this
# app is designed to stream/paginate a longer answer, so one generous
# ceiling for every call is simpler and safer than guessing per call site.
# Measured directly: a SUMMARY-mode repository report from a verbose model
# can run ~3,000 tokens; 4096 covers every case observed with headroom.
_ANTHROPIC_MAX_TOKENS = 4096


def _messages_to_anthropic(messages):
    """OpenAI puts the system prompt IN `messages` as a {"role": "system"}
    entry; Anthropic wants it as its own top-level `system` param and
    `messages` holding only user/assistant turns. Every call site in this
    app sends exactly one system + one user message, but this doesn't
    assume that -- it just pulls out whichever messages are role=="system"
    (concatenating if there happens to be more than one) and passes the
    rest through unchanged."""

    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    chat_messages = [m for m in messages if m.get("role") != "system"]
    system = "\n\n".join(system_parts) if system_parts else None
    return system, chat_messages


def _tools_to_anthropic(tools):
    if not tools:
        return None
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"]["parameters"],
        }
        for t in tools
    ]


def _tool_choice_to_anthropic(tool_choice):
    if not tool_choice or tool_choice == "auto":
        return {"type": "auto"}
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {"type": "tool", "name": tool_choice["function"]["name"]}
    return {"type": "auto"}


class _ToolCallFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments  # JSON string -- every call site does json.loads(...) on this


class _ToolCall:
    def __init__(self, function: _ToolCallFunction):
        self.function = function


class _Message:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message: _Message):
        self.message = message


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.prompt_tokens = input_tokens
        self.completion_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens


class _Response:
    def __init__(self, choices, usage: _Usage):
        self.choices = choices
        self.usage = usage


class _AnthropicCompletions:
    def __init__(self, real_client):
        self._client = real_client

    def create(self, model, messages, temperature=0, tools=None, tool_choice=None, **_ignored):
        """
        `model` is accepted (matching the OpenAI signature every call site
        already uses) but not forwarded as-is -- callers pass
        config.CHAT_MODEL, which already resolves to the right model
        string for whichever provider is active (see config.py), so this
        just uses it directly; there's no separate translation needed here.
        `**_ignored` absorbs any OpenAI-only kwarg a call site might add
        later that Anthropic has no equivalent for, so a new kwarg doesn't
        hard-crash this adapter -- it just gets silently dropped, same as
        it would functionally no-op on many OpenAI-side optional kwargs.
        """

        system, chat_messages = _messages_to_anthropic(messages)

        kwargs = {
            "model": model,
            "max_tokens": _ANTHROPIC_MAX_TOKENS,
            "temperature": temperature,
            "messages": chat_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools)
            kwargs["tool_choice"] = _tool_choice_to_anthropic(tool_choice)

        resp = self._client.messages.create(**kwargs)

        text = "".join(block.text for block in resp.content if block.type == "text")
        tool_calls = [
            _ToolCall(_ToolCallFunction(block.name, json.dumps(block.input)))
            for block in resp.content if block.type == "tool_use"
        ]

        message = _Message(content=text or None, tool_calls=tool_calls or None)
        usage = _Usage(resp.usage.input_tokens, resp.usage.output_tokens)
        return _Response([_Choice(message)], usage)


class _AnthropicChat:
    def __init__(self, real_client):
        self.completions = _AnthropicCompletions(real_client)


class _AnthropicClientAdapter:
    """Exposes `.chat.completions.create(...)` -- the exact surface every
    call site in this app already uses -- backed by the real Anthropic SDK
    client, so no call site needs to know or care which provider is
    actually active."""

    def __init__(self, real_client):
        self.chat = _AnthropicChat(real_client)
