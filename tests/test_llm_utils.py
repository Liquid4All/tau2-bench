import pytest

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import generate, to_litellm_messages


@pytest.fixture
def model() -> str:
    return "gpt-4o-mini"


@pytest.fixture
def messages() -> list[Message]:
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(role="user", content="What is the capital of the moon?"),
    ]
    return messages


@pytest.fixture
def tool() -> Tool:
    def calculate_square(x: int) -> int:
        """Calculate the square of a number.
            Args:
            x (int): The number to calculate the square of.
        Returns:
            int: The square of the number.
        """
        return x * x

    return as_tool(calculate_square)


@pytest.fixture
def tool_call_messages() -> list[Message]:
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(
            role="user",
            content="What is the square of 5? Just give me the number, no explanation.",
        ),
    ]
    return messages


def test_generate_no_tool_call(model: str, messages: list[Message]):
    response = generate(model, messages)
    assert isinstance(response, AssistantMessage)
    assert response.content is not None


def test_generate_tool_call(model: str, tool_call_messages: list[Message], tool: Tool):
    response = generate(model, tool_call_messages, tools=[tool])
    assert isinstance(response, AssistantMessage)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "calculate_square"
    assert response.tool_calls[0].arguments == {"x": 5}
    follow_up_messages = [
        response,
        ToolMessage(role="tool", id=response.tool_calls[0].id, content="25"),
    ]
    response = generate(
        model,
        tool_call_messages + follow_up_messages,
        tools=[tool],
    )
    assert isinstance(response, AssistantMessage)
    assert response.tool_calls is None
    assert response.content == "25"


def test_to_litellm_messages_liquid_prompt_tool_history():
    messages = [
        SystemMessage(role="system", content="Follow the policy."),
        UserMessage(role="user", content="Look up my account."),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="get_account",
                    arguments={"email": "test@example.com", "include_orders": True},
                )
            ],
        ),
        ToolMessage(role="tool", id="call_1", content='{"status": "active"}'),
    ]

    rendered = to_litellm_messages(messages, model="openai/liquid-api-Prompt")

    assert rendered[2] == {
        "role": "assistant",
        "content": (
            "<|tool_call_start|>"
            "[get_account(email='test@example.com', include_orders=True)]"
            "<|tool_call_end|>"
        ),
    }
    assert "tool_calls" not in rendered[2]
    assert rendered[3] == {"role": "tool", "content": '{"status": "active"}'}


def test_to_litellm_messages_liquid_prompt_same_for_hf_and_gguf_alias():
    messages = [
        SystemMessage(role="system", content="Follow the policy."),
        UserMessage(role="user", content="Look up my account."),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="get_account",
                    arguments={"email": "test@example.com"},
                )
            ],
        ),
    ]

    hf_messages = to_litellm_messages(messages, model="liquid-api-Prompt")
    gguf_messages = to_litellm_messages(messages, model="openai/liquid-api-Prompt")

    assert hf_messages == gguf_messages
