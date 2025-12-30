import json
import re
from typing import Any, Optional
import ast
from typing import Any, Dict, List

import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# litellm._turn_on_debug()

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


ALLOW_SONNET_THINKING = False

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")


#------------------LFMs Function Calling Parser--------------------
import ast
import json
import os
import re
import time
from typing import Any, Dict, List
import random


def extract_think_block(text: str) -> str | None:
    """
    Returns the content inside <think>...</think>, or None if not present.
    """
    if not text:
        return ""
    m = re.search(r"<think>([\s\S]*?)</think>", text)
    return m.group(1).strip() if m else ""

def parse_liquid_response(response: str | None) -> str:
    """
    Parse the response from LiquidAI and return the function call content.
    Extracts content from <|tool_call_start|>...<|tool_call_end|> tags if present.
    """
    if response is None:
        return "No Response"
    if not isinstance(response, str):
        try:
            response = str(response)
        except Exception:
            return ""
    
    if "<|tool_call_start|>" in response and "<|tool_call_end|>" in response:
        match = re.search(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", response, re.DOTALL)
        if match:
            answer = match.group(1)
        else:
            answer = response
        answer = re.sub(r"<\|tool_call_start\|>|<\|tool_call_end\|>", "", answer).strip()
        return answer
    else:
        return response

def generate_id():
    return f"tool_call_{random.randint(1, 100_000)}"

def _eval_node(node: ast.AST):
    """Convert a limited subset of AST nodes into Python objects."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.List):
        return [_eval_node(elt) for elt in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt) for elt in node.elts)
    elif isinstance(node, ast.Dict):
        return {
            _eval_node(k): _eval_node(v)
            for k, v in zip(node.keys, node.values)
        }
    elif isinstance(node, ast.Name):
        # For things like order=ascending / descending
        return node.id
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        # Handle negative numbers
        return -_eval_node(node.operand)
    else:
        raise ValueError(f"Unsupported AST node: {ast.dump(node)}")


def _get_function_name(node: ast.expr) -> str:
    """Extract function name from AST node, handling dotted names."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return _get_function_name(node.value) + "." + node.attr
    else:
        raise ValueError(f"Invalid function name node: {node}")


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Extract all tool calls from Liquid format response.
    
    Format: [func_name(arg1="val", arg2=123), func2(...)]
    
    Returns a list of {"name": str, "arguments": dict}.
    """
    if not text:
        return []

    calls: List[Dict[str, Any]] = []

    # Find ALL tool call blocks
    try:
        # Parse the function calls
        parsed = ast.parse(text).body[0].value.elts
        for call in parsed:
            try:
                function_name = _get_function_name(call.func)
                args = {kw.arg: _eval_node(kw.value) for kw in call.keywords}
                calls.append({'name': function_name, 'arguments': args})
            except Exception as e:
                # Log but continue processing other calls
                print(f"Warning: Failed to parse individual call: {e}")
                continue
    except Exception as e:
        # Log but continue processing other matches
        print(f"Warning: Failed to parse tool call block: {e}")
        return []
    
    return calls


def _is_tool_call_response_format(items: list) -> bool:
    """Check if the response is in the expected tool call format."""
    if not isinstance(items, list) or not items:
        return False
    for it in items:
        if not isinstance(it, dict):
            return False
        if set(it.keys()) != {"name", "arguments"}:
            return False
    return True


def parse_assistant_message(text: str) -> str:
    if not text:
        return ""
    think_block = extract_think_block(text)
    if "<think>" in text and "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.lstrip('\n')
    parsed_text = parse_liquid_response(text)
    if "<|tool_call_start|>" in text or "<|tool_call_end|>" in text:
        text = text.replace(parsed_text, "")
        text = text.replace("<|tool_call_start|>", "")
        text = text.replace("<|tool_call_end|>", "")
        text = text.strip()
    return text

def parse_tool_calls(text: str) -> List[ToolCall]:
    if not text:
        return []
    think_block = extract_think_block(text)
    if "<think>" in text and "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.lstrip('\n')
    parsed_text = parse_liquid_response(text)
    extracted_tool_calls = []
    if "<|tool_call_start|>" in text and "<|tool_call_end|>" in text:
        try:
            extracted_tool_calls = extract_tool_calls(parsed_text)
        except Exception as e:
            print(f"Warning: Failed to parse tool call block: {e}")
            return []
    output_calls = []
    for call in extracted_tool_calls:
        output_calls.append(ToolCall(
            id=generate_id(),
            name=call['name'],
            arguments=call['arguments'],
        ))
    return output_calls


def liquid_api_handler(content: str):
    tool_calls = parse_tool_calls(content) or None
    content = parse_assistant_message(content)
    return tool_calls, content
#-------------------------------------------------------------------


#------------------Granite Function Calling Parser--------------------
def granite_api_handler(content: str):
    tool_calls = []
    content = content
    if "<tool_call>" in content:
        tool_calls = [
            match.strip()
            for match in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
        ]
        for tool_call in tool_calls:
            try:
                if isinstance(tool_call, str):
                    call_dict = json.loads(tool_call)
                    tool_calls.append(ToolCall(
                        id=generate_id(),
                        name=call_dict['name'],
                        arguments=call_dict['arguments'],
                    ))
                elif isinstance(tool_call, dict):
                    tool_calls.append(ToolCall(
                        id=generate_id(),
                        name=tool_call['name'],
                        arguments=tool_call['arguments'],
                    ))
                elif isinstance(tool_call, ToolCall):
                    tool_calls.append(tool_call)
            except Exception as e:
                print(f"Warning: Failed to parse tool call: {e}")
                continue
    if not tool_calls:
        tool_calls = None
    return tool_calls, content
#-------------------------------------------------------------------

def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        #logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
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
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


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
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    litellm_messages = to_litellm_messages(messages)
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"
    try:
        response = completion(
            model=model,
            messages=litellm_messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
    except Exception as e:
        logger.error(e)
        message = AssistantMessage(
        role="assistant",
        content="Please use ###STOP### to exit the current conversation and try again later.",
        tool_calls=None,
        cost=0.0,
        usage={},
        raw_data={},
    )
        return message
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]
    try:
        finish_reason = response.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        message = AssistantMessage(
        role="assistant",
        content="Please use ###STOP### to exit the current conversation and try again later.",
        tool_calls=None,
        cost=0.0,
        usage={},
        raw_data={},
    )
        return message
    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response.message.content
    tool_calls = response.message.tool_calls or []
    #if model != "openrouter/qwen/qwen3-235b-a22b-2507":
        #print("tool_calls: ", tool_calls)
        #print("model: ", model)
        #print("content: ", content)
    if "liquid-api-Prompt" in model:
        tool_calls, content = liquid_api_handler(content)
    else:
        temp_tool_calls = []
        for tool_call in tool_calls:
            try:
                arg_item = {}
                if tool_call.function.arguments != "":
                    arg_item = json.loads(tool_call.function.arguments)
                temp_tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=arg_item,
                    )
                )
            except Exception as e:
                break
        tool_calls = temp_tool_calls or None
    if content and "<think>" in content and "</think>" in content:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = content.lstrip('\n')
    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
    )
    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


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
