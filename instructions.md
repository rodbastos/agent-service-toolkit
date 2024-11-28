LangGraph
LangGraph is LangChain's graph-based agent framework and one of the most popular frameworks for building graph-based agents. It focuses on providing more "fine-grained" control over an agent's flow and state.

State
At the core of a graph in LangGraph is the agent state. The state is a mutable object where we track the current state of the agent execution as we pass through the graph. We can include different parameters within the state, in our research agent we will use a minimal example containing three parameters:

input: This is the user's most recent query. Usually, this is a question that we want to answer with our research agent.
chat_history: We are building a conversational agent that can support multiple interactions. To allow previous interactions to provide additional context throughout our agent logic, we include the chat history in the agent state.
intermediate_steps provides a record of all steps the research agent will take between the user asking a question via input and the agent providing a final answer. These can include "search arxiv", "perform general purpose web search," etc. These intermediate steps are crucial to allowing the agent to follow a path of coherent actions and ultimately producing an informed final answer.
Follow Along with Code!
The remainder of this article will be focused on the implementation of a LangGraph research agent using Python. You can find a copy of the code here.

If following along, make sure to install prerequisite packages first:


    langchain-pinecone==0.1.1 \
    langchain-openai==0.1.9 \
    langchain==0.2.5 \
    langchain-core==0.2.9 \
    langgraph==0.1.1 \
    semantic-router==0.0.48 \


We define our minimal agent state object like so:

from typing import TypedDict, Annotated
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
   input: str
   chat_history: list[BaseMessage]
   intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

Creating our Tools
The agent graph consists of several nodes, five of which are custom tools we need to create. Those tools are:


RAG search: We have a knowledge base on Pinecone vector store. This tool provides our agent with access to this knowledge.
RAG search with filter: Sometimes, our agent may need more information from a specific paper. This tool allows our agent to do just that.

We'll setup each of these tools, which our LLM and graph-logic will later be able to execute.


RAG Tools
We provide two RAG-focused tools for our agent. The rag_search allows the agent to perform a simple RAG search for some information across all indexed documents. The rag_search_filter also searches, but within a specific paper filtered for via some metadata.

The metadada that we have in our Pinecone is:

metadata = {
            'label': str(row['Label']),
            'description': str(row['Description']),
            'cluster_main': str(row['Cluster_main']),
            'num_clusters': int(row['Num_Clusters']),
            'cluster_assignments': str(row['Cluster_Assignments']),
            'cluster_color': int(row['Cluster_Color']),
            'main_cluster_prob': float(row['Main_Cluster_Prob']),
            'size_prob': float(row['Size_Prob']),
            'user_id': str(row['UserID']),
            'abs': int(row['Abs']),
            'temp': int(row['Temp'])
        }

We also define the format_rag_contexts function to handle the transformation of our Pinecone results from a JSON object to a readble plaintext format. 


Code exemple, that is going to be adapted to our neeeds:

def format_rag_contexts(matches: list):
   contexts = []
   for x in matches:
       text = (
           f"Title: {x['metadata']['title']}\n"
           f"Content: {x['metadata']['content']}\n"
           f"ArXiv ID: {x['metadata']['arxiv_id']}\n"
           f"Related Papers: {x['metadata']['references']}\n"
       )
       contexts.append(text)
   context_str = "\n---\n".join(contexts)
   return context_str


More code to be adapted to our needs:

@tool("rag_search_filter")
def rag_search_filter(query: str, arxiv_id: str):
   """Finds information from our vector database using a natural language query
   and a specific metadata filter. Allows us to learn more details about a specific narrative."""
   xq = encoder([query])
   xc = index.query(vector=xq, top_k=6, include_metadata=True, filter={"arxiv_id": arxiv_id})
   context_str = format_rag_contexts(xc["matches"])
   return context_str


@tool("rag_search")
def rag_search(query: str):
   """Finds narratives colllected in interviews using a natural language query."""
   xq = encoder([query])
   xc = index.query(vector=xq, top_k=5, include_metadata=True)
   context_str = format_rag_contexts(xc["matches"])
   return context_str