from datetime import datetime
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools.rag_tool import rag_tool
from core import get_model, settings


AgentState = MessagesState | dict[Literal["safety", "is_last_step"], LlamaGuardOutput | IsLastStep]


web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [web_search]

# Add RAG tool
tools.append(rag_tool)

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
if settings.OPENWEATHERMAP_API_KEY:
    tools.append(OpenWeatherMapQueryRun(name="Weather"))

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    Você é um assistente para Agentes de Mudança e Designer Organizacionais. Você tem acesso a uma base de dados sobre o modelo de contratação chamado qt.k da empresa Corsan. 
    A data de hoje é {current_date}.

    NOTA: O conteúdo da resposta da Tool não é mostrado ao usuário. 

    A few things to remember:
    - Gere respostas com markdown. Cite narrativas específicas. 
    - Use the 'rag' tool to query the knowledge base for relevant information. The RAG tool returns information
      from our narratives and themes collected in interviews with people from the organization.
      When using the RAG tool:
      - For simple factual questions, use k=1-2 results
      - For complex questions requiring multiple perspectives, use k=3-5
      - For comprehensive questions, use k=5-10
    - If you need to search the web, use the 'WebSearch' tool
    - Always summarize and cite the relevant information from tool responses in your own words
    - Do not include raw tool responses in your messages
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    # Ensure 'messages' and 'is_last_step' keys exist in state
    state.setdefault("messages", [])
    state.setdefault("is_last_step", False)
    
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    
    # Convert messages to proper format for OpenAI
    formatted_messages = []
    last_assistant_message = None
    
    for msg in state["messages"]:
        if hasattr(msg, "role"):
            if msg.role == "tool":
                # Skip tool messages - we'll handle them with the preceding assistant message
                continue
            elif msg.role == "assistant" and hasattr(msg, "tool_calls") and msg.tool_calls:
                # Store assistant message with tool calls to pair with subsequent tool messages
                last_assistant_message = msg
                formatted_messages.append(msg)
            elif msg.role == "function":
                # If we have a preceding assistant message with tool calls, add this as the function response
                if last_assistant_message and hasattr(last_assistant_message, "tool_calls"):
                    formatted_messages.append(msg)
                last_assistant_message = None
            else:
                formatted_messages.append(msg)
        else:
            formatted_messages.append(msg)
    
    # Update state with properly formatted messages
    state["messages"] = formatted_messages
    
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

    if state["is_last_step"] and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

research_assistant = agent.compile(checkpointer=MemorySaver())
