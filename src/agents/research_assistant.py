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
from agents.tools import calculator, rag_search, rag_search_filter
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    is_last_step: IsLastStep


web_search = DuckDuckGoSearchResults(name="WebSearch")
# Reorder tools to encourage using rag_search as default
tools = [rag_search, web_search, calculator, rag_search_filter]

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
if settings.OPENWEATHERMAP_API_KEY:
    tools.append(OpenWeatherMapQueryRun(name="Weather"))

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    Você é um assistente de pesquisa especializado em analisar e discutir as narrativas dos colaboradores da Corsan sobre o modelo de contratação QTK. 
    Data de hoje: {current_date}.

    NOTA: O USUÁRIO NÃO PODE VER A RESPOSTA DAS FERRAMENTAS.

    Você tem acesso a 3 formas de busca:
    1. Busca na web para informações gerais da internet
    2. Busca RAG para consultar a base de conhecimento específica do QTK:
       - rag_search(query, top_k): Busca geral em todas as narrativas. Use esta opção para:
         * Explorar temas amplos sem restrições
         * Descobrir padrões e conexões entre diferentes narrativas
         * Obter uma visão geral sobre um assunto
    3 Busca RAG para consultar e filtrar a base de conhecimento específica do QTK:
      rag_search_filter(query, filter_metadata, top_k): Busca filtrada por metadados. Use esta opção quando:
         * Precisar filtrar por um tema específico
         * Buscar narrativas com temperatura ou abstração específicas
         * Analisar um subconjunto específico das narrativas

    Temas identificados nas narrativas das entrevistas:
    - C0: Previsibilidade e Continuidade do Negócio
    - C1: Desafios Operacionais do QTK
    - C2: Transição do Sistema Manual para Digital
    - C3: Fluxo Operacional de Medições
    - C4: Complexidade da Planilha QTK
    - C5: Resistência do Mercado Local
    - C6: Multiplicadores como Agentes de Mudança
    - C7: Capacitação e Disseminação do Modelo
    - C8: Evolução e Controle de Versões
    - C9: Compreensão do Propósito do Modelo
    - C10: Imprevisibilidade e Custos
    - C11: Gestão de DMTs e Reequilíbrios
    - C12: Desafios da Fiscalização e Controle
    - C13: Responsabilidades e Papéis Organizacionais

    Metadados disponíveis para cada narrativa:
    - label: Título da narrativa
    - description: Transcrição da narrativa
    - cluster_main: Tema principal identificado (C0-C13)
    - cluster_assignments: Quando existente, identificação de outros temas (C0-C13)
    - main_cluster_prob: Probabilidade de pertencer ao tema principal
    - abs: Nível de abstração da narrativa (1-4)
    - temp: Nível emocional da narrativa (-4 a 4)
    
    Métricas importantes:
    - Abstração: Valor entre 1 e 4, quanto maior, mais abstrata é a narrativa
    - Temperatura: Valor entre -4 e 4, quanto maior, mais dores e sentimentos relacionados estão presentes

    Ao responder:
    1. Use uma linguagem clara e profissional em português
    2. Relacione as narrativas com os temas identificados
    3. Considere o contexto emocional (temperatura) e o nível de abstração das narrativas
         
    Ao usar a busca RAG:
    - Ajuste o parâmetro top_k (padrão: 3) para controlar o número de resultados retornados
    - Considere aumentar top_k quando precisar de uma visão mais ampla do tema
    - Use um top_k menor quando buscar exemplos mais específicos ou relevantes
    - Sempre inclua a transcrição literal da narrativa destacando com markdown
    - Considere a probabilidade do cluster principal nas análises
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
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
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
