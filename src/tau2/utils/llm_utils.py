import json
import os
import re
from functools import lru_cache
from typing import Any, Optional

import dotenv
from loguru import logger
from openai import OpenAI

from tau2.config import DEFAULT_LLM_AGENT, DEFAULT_LLM_USER, DEFAULT_MAX_RETRIES
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.token_cost import TOKEN_COST_PER_MILLION

dotenv.load_dotenv(override=True)


def _get_env_var(*names: str) -> Optional[str]:
    for name in names:
        candidates = {name, name.upper(), name.lower()}
        for candidate in candidates:
            value = os.getenv(candidate)
            if value:
                return value.strip('"')
    return None


def _resolve_model_override(model: str) -> tuple[Optional[str], Optional[str]]:
    model_lc = model.lower()
    if model_lc.startswith("deepseek"):
        return (
            _get_env_var("deepseek_base_url"),
            _get_env_var("deepseek_api_key"),
        )
    if model_lc.startswith("kimi"):
        return (
            _get_env_var("kimi_base_url"),
            _get_env_var("KIMI_API_KEY", "kimi_api_key"),
        )

    if model_lc.startswith(("openai", "google", "anthropic")):
        return (
            _get_env_var("openrouter_base_url"),
            _get_env_var("openrouter_api_key"),
        )
    if model_lc.startswith("glm"):
        return (
            _get_env_var("glm_base_url"),
            _get_env_var("glm_api_key"),
        )
    if model_lc.startswith("minimax"):
        return (
            _get_env_var("minimax_base_url"),
            _get_env_var("minimax_api_key"),
        )
    return (None, None)


@lru_cache(maxsize=None)
def _build_client(model: str, max_retries: int) -> OpenAI:
    base_url, api_key = _resolve_model_override(model)
    kwargs: dict[str, Any] = {"max_retries": max_retries}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAI(**kwargs)


def _get_client_for_model(model: str, max_retries: Optional[int]) -> OpenAI:
    retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES
    return _build_client(model.lower(), retries)

def get_response_cost(response: Any) -> float:
    return 0.0


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
    """Convert internal messages to OpenAI-compatible format.
    
    Different models use different field names for reasoning:
    - DeepSeek Reasoner: reasoning_content (required for all assistant messages)
    - OpenRouter/Anthropic: reasoning_details
    """
    model_lc = model.lower()
    is_deepseek_reasoner = "deepseek-reasoner" in model_lc or "deepseek-r1" in model_lc
    
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
                if is_deepseek_reasoner:
                    # DeepSeek uses reasoning_content field
                    # Convert list format to string if needed
                    reasoning = message.reasoning_details
                    if isinstance(reasoning, list):
                        # Extract text from structured reasoning
                        reasoning = "".join(
                            item.get("text", "") or item.get("summary", "") or ""
                            for item in reasoning if isinstance(item, dict)
                        )
                    openai_message["reasoning_content"] = reasoning if reasoning else ""
                else:
                    openai_message["reasoning_details"] = message.reasoning_details
            elif is_deepseek_reasoner:
                # DeepSeek requires reasoning_content even if empty
                openai_message["reasoning_content"] = ""
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

    Returns: A tuple containing the message and the cost.
    """
    num_retries = kwargs.pop("num_retries", None)
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
    reasoning_list: list[Any] = []
    reasoning_acc: dict[tuple, dict[str, Any]] = {}
    reasoning_order: list[tuple] = []
    finish_reason = None
    usage: Optional[dict] = None
    raw_data: Optional[dict] = None
    model_id: Optional[str] = None
    last_chunk: Any = None

    try:
        
        stream = client.chat.completions.create(
            model=model,
            messages=openai_messages,
            tools=openai_tools,
            tool_choice=tool_choice,
            stream=True,
            **kwargs,
            extra_body={"reasoning": {"enabled": True}},

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

            delta_reasoning = getattr(delta, "reasoning_details", None)
            if delta_reasoning is None:
                delta_reasoning = getattr(delta, "reasoning_content", None)
            if delta_reasoning:
                # For string-based streaming reasoning, accumulate all pieces
                if isinstance(delta_reasoning, str):
                    reasoning_parts.append(delta_reasoning)
                    reasoning = "".join(reasoning_parts)
                elif isinstance(delta_reasoning, list):
                    # For list-based structured reasoning (e.g., OpenAI/Gemini),
                    # aggregate entries by (index, type, format) and concatenate
                    # their string fields (summary/text/data) across chunks.
                    for item in delta_reasoning:
                        if not isinstance(item, dict):
                            reasoning_list.append(item)
                            continue

                        key = (
                            item.get("index"),
                            item.get("type"),
                            item.get("format"),
                        )

                        if key not in reasoning_acc:
                            reasoning_acc[key] = {}
                            reasoning_order.append(key)

                        acc_item = reasoning_acc[key]

                        # Copy all fields, concatenating string content fields
                        for k, v in item.items():
                            if k in ("summary", "text", "data"):
                                # Concatenate content string fields across chunks
                                if isinstance(v, str):
                                    prev = acc_item.get(k, "") or ""
                                    acc_item[k] = prev + v
                            elif k == "signature":
                                # Concatenate signature across chunks
                                if isinstance(v, str):
                                    prev = acc_item.get(k, "") or ""
                                    acc_item[k] = prev + v
                            elif k not in acc_item:
                                # Copy other fields once
                                acc_item[k] = v

                    if reasoning_order:
                        reasoning_list = [reasoning_acc[k] for k in reasoning_order]
                        reasoning = reasoning_list
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
                aggregated_delta["reasoning_details"] = reasoning
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

    # Compute per-message cost using the same logic as aggregate get_cost
    try:
        res = get_cost([message])
    except Exception as e:
        logger.warning(f"Failed to compute per-message cost: {e}")
    else:
        if res is not None:
            agent_cost, _ = res
            message.cost = agent_cost

    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    agent_cost = 0.0
    user_cost = 0.0
    has_cost = False

    for message in messages:
        # Only assistant and user messages incur LLM cost
        if not isinstance(message, (AssistantMessage, UserMessage)):
            continue

        usage = message.usage
        if not usage:
            continue

        # Try to get the actual model id from the raw response first
        model_name = None
        if isinstance(message.raw_data, dict):
            # Prefer the original requested model name if present, fall back to provider model id
            model_name = message.raw_data.get("request_model") or message.raw_data.get(
                "model"
            )

        # Fallback to defaults if model is missing
        if not model_name:
            if isinstance(message, AssistantMessage):
                model_name = DEFAULT_LLM_AGENT
            elif isinstance(message, UserMessage):
                model_name = DEFAULT_LLM_USER

        if not model_name:
            continue

        key = str(model_name).lower()
        # Handle simple fine-tuned naming like ft:base:provider::id
        if key.startswith("ft:"):
            parts = key.split(":", 2)
            if len(parts) >= 2 and parts[1]:
                key = parts[1]

        price = TOKEN_COST_PER_MILLION.get(key)
        if not price:
            continue

        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0

        input_cost = (prompt_tokens / 1_000_000.0) * float(price["input"])
        output_cost = (completion_tokens / 1_000_000.0) * float(price["output"])
        total_cost = input_cost + output_cost

        if isinstance(message, AssistantMessage):
            agent_cost += total_cost
            has_cost = True
        elif isinstance(message, UserMessage):
            user_cost += total_cost
            has_cost = True

    if not has_cost:
        return None

    return round(agent_cost, 6), round(user_cost, 6)


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
