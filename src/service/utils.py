from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)

from schema import ChatMessage
import uuid
import logging

logger = logging.getLogger(__name__)

def convert_message_content_to_string(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    text: list[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item["type"] == "text":
            text.append(content_item["text"])
    return "".join(text)


def remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content and format them consistently."""
    if isinstance(content, str):
        return content
    # Handle both Anthropic's tool_use and OpenAI's function_call formats
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) 
        or (content_item.get("type") not in ["tool_use", "function_call"])
    ]


def langchain_to_chat_message(message: BaseMessage) -> ChatMessage:
    """Create a ChatMessage from a LangChain message."""
    try:
        match message:
            case HumanMessage():
                human_message = ChatMessage(
                    type="human",
                    content=convert_message_content_to_string(message.content),
                )
                return human_message
            case AIMessage():
                ai_message = ChatMessage(
                    type="ai",
                    content=convert_message_content_to_string(message.content),
                )
                if message.tool_calls:
                    ai_message.tool_calls = message.tool_calls
                if message.additional_kwargs:
                    ai_message.response_metadata = message.additional_kwargs
                return ai_message
            case ToolMessage():
                tool_message = ChatMessage(
                    type="tool",
                    content=convert_message_content_to_string(message.content),
                    tool_call_id=message.tool_call_id,
                )
                return tool_message
            case LangchainChatMessage():
                if message.role == "custom":
                    custom_message = ChatMessage(
                        type="custom",
                        content="",
                        custom_data=message.content[0] if isinstance(message.content, list) else message.content,
                    )
                    return custom_message
                elif message.role == "assistant":
                    return ChatMessage(type="ai", content=convert_message_content_to_string(message.content))
                elif message.role == "user":
                    return ChatMessage(type="human", content=convert_message_content_to_string(message.content))
                else:
                    raise ValueError(f"Unsupported chat message role: {message.role}")
            case _:
                raise ValueError(f"Unsupported message type: {message.__class__.__name__}")
    except Exception as e:
        logger.error(f"Error in langchain_to_chat_message: {e}")
        return ChatMessage(type="error", content=str(e))
