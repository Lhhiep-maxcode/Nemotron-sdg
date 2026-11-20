"""Preserve model-returned reasoning across response formats and assistant turns."""

from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion
from retrieval_sdg.conversation.engine import _assistant_msg
from retrieval_sdg.core.llm import (
    _apply_majority_vote,
    _choice_to_dict,
    _message_to_dict,
    message_text_and_reasoning,
)


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize("tool_call", [False, True])
def test_sdk_reasoning_survives_response_parsing_and_assistant_messages(field, tool_call):
    message = {"role": "assistant", "content": None if tool_call else "108.", field: "Model-provided reasoning."}
    if tool_call:
        message["tool_calls"] = [{"id": "call_1", "type": "function", "function": {
            "name": "calculator", "arguments": '{"expression":"54*2"}'}}]
    completion = ChatCompletion.model_validate({"id": "test", "object": "chat.completion", "created": 0,
                                                "model": "test", "choices": [{"index": 0, "message": message,
                                                    "finish_reason": "tool_calls" if tool_call else "stop"}]})
    normalized = _choice_to_dict(completion.choices[0])
    saved = _assistant_msg(normalized, normalized.get("tool_calls"))
    assert saved["reasoning_content"] == "Model-provided reasoning."
    assert saved["content"] == ("" if tool_call else "108.")
    assert "reasoning" not in saved
    if tool_call:
        assert saved["tool_calls"] == message["tool_calls"]


def test_facade_reasoning_alias_is_normalized():
    result = _message_to_dict(SimpleNamespace(content="Answer", reasoning="Model rationale", tool_calls=[]))
    assert result == {"role": "assistant", "content": "Answer", "reasoning_content": "Model rationale"}


def test_complete_leading_think_block_is_split_from_answer():
    content = "<think>\nModel-provided reasoning.\n</think>\n\nThe answer is 108."
    assert message_text_and_reasoning({"content": content}) == ("The answer is 108.", "Model-provided reasoning.")
    saved = _assistant_msg({"content": content})
    assert saved["content"] == "The answer is 108." and saved["reasoning_content"] == "Model-provided reasoning."


@pytest.mark.parametrize("content", ["The literal tag is <think>example</think>.", "<think>incomplete reasoning"])
def test_quoted_or_incomplete_think_blocks_are_not_misclassified(content):
    assert message_text_and_reasoning({"content": content}) == (content, "")


def test_explicit_reasoning_wins_and_whitespace_field_falls_back_to_alias():
    content = "<think>A literal leading example</think> is part of this answer."
    assert message_text_and_reasoning({"content": content, "reasoning_content": "Native trace"}) \
        == (content, "Native trace")
    assert message_text_and_reasoning({"content": "Answer", "reasoning_content": "  ", "reasoning": "Native trace"}) \
        == ("Answer", "Native trace")


def test_majority_vote_retains_selected_model_reasoning():
    response = {"content": "", "reasoning_content": "The source values support this calculation.", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {
            "name": "calculator", "arguments": '{"expression":"54*2"}'}}]}
    selected = _apply_majority_vote([response, response, response])
    saved = _assistant_msg(selected, selected["tool_calls"])
    assert saved["reasoning_content"] == response["reasoning_content"]
    assert saved["tool_calls"][0]["function"]["name"] == "calculator"
