import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import dotenv
from loguru import logger
from openai import OpenAI

from tau2.config import DEFAULT_MAX_RETRIES
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.io_utils import load_file

dotenv.load_dotenv(override=True)


_MODELS_YAML_NAME = "models.yaml"
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _models_yaml_path() -> Path:
    return Path(__file__).resolve().parents[3] / _MODELS_YAML_NAME


@lru_cache(maxsize=1)
def _load_models_yaml() -> dict[str, Any]:
    path = _models_yaml_path()
    data = load_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid models.yaml format at {path}: expected a mapping")
    return data


def _expand_env_vars(value: Any, *, model: str) -> Any:
    if isinstance(value, str):
        def _repl(match: re.Match[str]) -> str:
            var = match.group(1)
            env_val = os.getenv(var)
            if env_val is None:
                raise ValueError(
                    f"Environment variable '{var}' is required for model '{model}' in models.yaml"
                )
            return env_val

        return _ENV_VAR_PATTERN.sub(_repl, value)

    if isinstance(value, dict):
        return {k: _expand_env_vars(v, model=model) for k, v in value.items()}

    if isinstance(value, list):
        return [_expand_env_vars(v, model=model) for v in value]

    return value


@lru_cache(maxsize=None)
def _get_model_config(model: str) -> dict[str, Any]:
    model = (model or "").strip()
    if not model:
        raise ValueError("Model name must be a non-empty string")

    data = _load_models_yaml()

    default_cfg = data.get("default") or {}
    if not isinstance(default_cfg, dict):
        raise ValueError("Invalid models.yaml: 'default' must be a mapping")

    models = data.get("models") or []
    if not isinstance(models, list):
        raise ValueError("Invalid models.yaml: 'models' must be a list")

    model_lc = model.lower()
    matched: Optional[dict[str, Any]] = None
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.lower() == model_lc:
            matched = item
            break

    if matched is None:
        raise ValueError(f"Model '{model}' not found in models.yaml")

    merged = {**default_cfg, **matched}
    merged = _expand_env_vars(merged, model=model)
    return merged


@lru_cache(maxsize=None)
def _build_client(model: str, max_retries: int) -> OpenAI:
    cfg = _get_model_config(model)
    base_url = cfg.get("base_url")
    api_key = cfg.get("api_key")
    if not base_url:
        raise ValueError(f"Missing 'base_url' for model '{model}' in models.yaml")
    if not api_key:
        raise ValueError(f"Missing 'api_key' for model '{model}' in models.yaml")

    kwargs: dict[str, Any] = {
        "max_retries": max_retries,
        "base_url": base_url,
        "api_key": api_key,
    }
    return OpenAI(**kwargs)


def _get_client_for_model(model: str, max_retries: Optional[int]) -> OpenAI:
    retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES
    return _build_client(model.lower(), retries)


def _model_uses_reasoning_details(model: str) -> bool:
    return _get_reasoning_details_provider(model) is not None


def _get_reasoning_details_provider(model: str) -> Optional[str]:
    model_lc = (model or "").lower()
    if "openai" in model_lc:
        return "openai"
    if "anthropic" in model_lc:
        return "anthropic"
    if "google" in model_lc:
        return "google"
    return None


def _accumulate_reasoning_details_items(
    delta_reasoning: list[Any],
    reasoning_acc: dict[tuple[Any, ...], dict[str, Any]],
    reasoning_order: list[tuple[Any, ...]],
    *,
    key_fn: Any,
) -> list[Any]:
    for item in delta_reasoning:
        if not isinstance(item, dict):
            continue

        key = key_fn(item)

        if key not in reasoning_acc:
            reasoning_acc[key] = {}
            reasoning_order.append(key)

        acc_item = reasoning_acc[key]

        for k, v in item.items():
            if k in ("summary", "text", "data"):
                if isinstance(v, str):
                    prev = acc_item.get(k, "") or ""
                    acc_item[k] = prev + v
            elif k == "signature":
                if isinstance(v, str):
                    prev = acc_item.get(k, "") or ""
                    acc_item[k] = prev + v
            elif k not in acc_item:
                acc_item[k] = v

    return [reasoning_acc[k] for k in reasoning_order]


def _accumulate_openai_reasoning_details(
    delta_reasoning: list[Any],
    reasoning_acc: dict[tuple[Any, ...], dict[str, Any]],
    reasoning_order: list[tuple[Any, ...]],
) -> list[Any]:
    def _key(item: dict[str, Any]) -> tuple[Any, ...]:
        item_id = item.get("id")
        if item_id:
            return ("id", item_id)
        return (item.get("index"), item.get("type"), item.get("format"))

    return _accumulate_reasoning_details_items(
        delta_reasoning,
        reasoning_acc,
        reasoning_order,
        key_fn=_key,
    )


def _accumulate_anthropic_reasoning_details(
    delta_reasoning: list[Any],
    reasoning_acc: dict[tuple[Any, ...], dict[str, Any]],
    reasoning_order: list[tuple[Any, ...]],
) -> list[Any]:
    def _key(item: dict[str, Any]) -> tuple[Any, ...]:
        return (item.get("index"), item.get("type"), item.get("format"))

    return _accumulate_reasoning_details_items(
        delta_reasoning,
        reasoning_acc,
        reasoning_order,
        key_fn=_key,
    )


def _accumulate_google_reasoning_details(
    delta_reasoning: list[Any],
    reasoning_acc: dict[tuple[Any, ...], dict[str, Any]],
    reasoning_order: list[tuple[Any, ...]],
) -> list[Any]:
    def _key(item: dict[str, Any]) -> tuple[Any, ...]:
        return (item.get("index"), item.get("type"), item.get("format"))

    return _accumulate_reasoning_details_items(
        delta_reasoning,
        reasoning_acc,
        reasoning_order,
        key_fn=_key,
    )


def get_response_usage(response: Any) -> Optional[dict]:
    usage = getattr(response, "usage", None)
    if usage is None:
        choices = getattr(response, "choices", None)
        if choices:
            first_choice = choices[0]
            usage = getattr(first_choice, "usage", None)
    if usage is None:
        return None

    completion_tokens = getattr(usage, "completion_tokens", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)

    if completion_tokens is None and prompt_tokens is None and isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")

    if completion_tokens is None and prompt_tokens is None:
        return None

    return {
        "completion_tokens": completion_tokens or 0,
        "prompt_tokens": prompt_tokens or 0,
    }


def to_openai_messages(messages: list[Message], model: str = "") -> list[dict]:
    """Convert internal messages to OpenAI-compatible format."""

    uses_reasoning_details = _model_uses_reasoning_details(model)
    
    openai_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            openai_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            # Standard OpenAI/OpenRouter format
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            # Use None instead of empty string for content
            content = message.content if message.content else None
            openai_message = {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
            # Pass reasoning back using the appropriate field name
            if message.reasoning_details is not None:
                if uses_reasoning_details:
                    openai_message["reasoning_details"] = message.reasoning_details
                else:
                    reasoning = message.reasoning_details
                    if isinstance(reasoning, list):
                        reasoning = "".join(
                            item.get("text", "") or item.get("summary", "") or ""
                            for item in reasoning
                            if isinstance(item, dict)
                        )
                    openai_message["reasoning_content"] = reasoning
            openai_messages.append(openai_message)
        elif isinstance(message, ToolMessage):
            openai_messages.append(
                {
                    "role": "tool",
                    "content": message.content if message.content else "(no output)",
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            openai_messages.append({"role": "system", "content": message.content})
    return openai_messages


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: The assistant message.
    """
    num_retries = kwargs.pop("num_retries", None)
    reasoning_details_provider = _get_reasoning_details_provider(model)
    uses_reasoning_details = _model_uses_reasoning_details(model)
    openai_messages = to_openai_messages(messages, model=model)
    openai_tools = [tool.openai_schema for tool in tools] if tools else None
    if openai_tools and tool_choice is None:
        tool_choice = "auto"
    client = _get_client_for_model(model, num_retries)
    content_parts: list[str] = []
    tool_calls_data: dict[int, dict[str, Any]] = {}
    role: Optional[str] = None
    reasoning: Optional[Any] = None
    reasoning_parts: list[str] = []
    reasoning_acc: dict[tuple, dict[str, Any]] = {}
    reasoning_order: list[tuple] = []
    finish_reason = None
    usage: Optional[dict] = None
    raw_data: Optional[dict] = None
    model_id: Optional[str] = None
    last_chunk: Any = None

    extra_kwargs = {}
    if "openai" in model.lower():
        extra_kwargs["reasoning_effort"] = "high"
    else:
        extra_kwargs["extra_body"] = {"reasoning": {"enabled": True}}

    try:
        
        stream = client.chat.completions.create(
            model=model,
            messages=openai_messages,
            tools=openai_tools,
            tool_choice=tool_choice,
            stream=True,
            **kwargs,
            **extra_kwargs,
        )
        for chunk in stream:
            last_chunk = chunk
            if getattr(chunk, "model", None) is not None:
                model_id = chunk.model

            chunk_usage = get_response_usage(chunk)
            if chunk_usage is not None:
                usage = chunk_usage

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            delta_role = getattr(delta, "role", None)
            if delta_role is not None and role is None:
                role = delta_role

            delta_content = getattr(delta, "content", None)
            if delta_content:
                content_parts.append(delta_content)

            delta_tool_calls = getattr(delta, "tool_calls", None)
            if delta_tool_calls:
                for idx, tc in enumerate(delta_tool_calls):
                    tc_index = getattr(tc, "index", None)
                    if tc_index is None:
                        tc_index = idx
                    if tc_index not in tool_calls_data:
                        tool_calls_data[tc_index] = {
                            "id": None,
                            "name": None,
                            "arguments": "",
                        }
                    data = tool_calls_data[tc_index]
                    if getattr(tc, "id", None):
                        data["id"] = tc.id
                    function = getattr(tc, "function", None)
                    if function is not None:
                        if getattr(function, "name", None):
                            data["name"] = function.name
                        if getattr(function, "arguments", None):
                            data["arguments"] += function.arguments or ""

            if uses_reasoning_details:
                delta_reasoning = getattr(delta, "reasoning_details", None)
            else:
                delta_reasoning = getattr(delta, "reasoning_content", None)
            if delta_reasoning:
                # For string-based streaming reasoning, accumulate all pieces
                if isinstance(delta_reasoning, str):
                    reasoning_parts.append(delta_reasoning)
                    reasoning = "".join(reasoning_parts)
                elif isinstance(delta_reasoning, list):
                    if reasoning_details_provider == "openai":
                        reasoning = _accumulate_openai_reasoning_details(
                            delta_reasoning,
                            reasoning_acc,
                            reasoning_order,
                        )
                    elif reasoning_details_provider == "anthropic":
                        reasoning = _accumulate_anthropic_reasoning_details(
                            delta_reasoning,
                            reasoning_acc,
                            reasoning_order,
                        )
                    elif reasoning_details_provider == "google":
                        reasoning = _accumulate_google_reasoning_details(
                            delta_reasoning,
                            reasoning_acc,
                            reasoning_order,
                        )
                    else:
                        reasoning = delta_reasoning
                else:
                    # For non-string payloads, keep the last non-empty value
                    reasoning = delta_reasoning

            if getattr(choice, "finish_reason", None) is not None:
                finish_reason = choice.finish_reason
    except Exception as e:
        logger.error(e)
        raise e

    if finish_reason == "length":
        logger.warning("Output might be incomplete due to token limit!")

    if role is None:
        role = "assistant"
    assert role == "assistant", (
        "The response should be an assistant message"
    )

    content = "".join(content_parts) if content_parts else ""

    tool_calls_list: list[ToolCall] = []
    for index in sorted(tool_calls_data.keys()):
        data = tool_calls_data[index]
        if not data.get("name"):
            continue
        arguments_str = data.get("arguments") or "{}"
        try:
            arguments = json.loads(arguments_str)
        except Exception:
            logger.warning(
                f"Failed to parse tool call arguments as JSON: {arguments_str}"
            )
            arguments = {}
        tool_calls_list.append(
            ToolCall(
                id=data.get("id"),
                name=data["name"],
                arguments=arguments,
            )
        )
    tool_calls = tool_calls_list or None

    if last_chunk is not None:
        try:
            raw_data = last_chunk.model_dump()
        except Exception:
            raw_data = None

    if isinstance(raw_data, dict):
        choices = raw_data.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            aggregated_delta: dict[str, Any] = {}

            if role is not None:
                aggregated_delta["role"] = role
            if content:
                aggregated_delta["content"] = content
            if reasoning is not None:
                if uses_reasoning_details:
                    aggregated_delta["reasoning_details"] = reasoning
                else:
                    aggregated_delta["reasoning_content"] = reasoning
            if tool_calls:
                aggregated_delta["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]

            first_choice["delta"] = aggregated_delta
            if finish_reason is not None:
                first_choice["finish_reason"] = finish_reason

        raw_data["request_model"] = model
        if model_id is not None and "model" not in raw_data:
            raw_data["model"] = model_id

    # Build assistant message with usage/raw_data attached
    message = AssistantMessage(
        role="assistant",
        content=content,
        reasoning_details=reasoning,
        tool_calls=tool_calls,
        cost=None,
        usage=usage,
        raw_data=raw_data,
    )

    if not (message.has_text_content() or message.is_tool_call()):
        if isinstance(raw_data, dict):
            try:
                raw_keys = list(raw_data.keys())
            except Exception:
                raw_keys = []
        else:
            raw_keys = []
        error_msg = (
            "LLM returned an empty assistant message (no content or tool calls). "
            f"model={model}, raw_data keys={raw_keys}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    return message


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
