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


# Pinecone - Understanding metadata
You can attach metadata key-value pairs to vectors in an index. When you query the index, you can then filter by metadata to ensure only relevant records are scanned.

Searches with metadata filters retrieve exactly the number of nearest-neighbor results that match the filters. For most cases, the search latency will be even lower than unfiltered searches.

Searches without metadata filters do not consider metadata. To combine keywords with semantic search, see sparse-dense embeddings.

​
Supported metadata types
Metadata payloads must be key-value pairs in a JSON object. Keys must be strings, and values can be one of the following data types:

String
Number (integer or floating point, gets converted to a 64 bit floating point)
Booleans (true, false)
List of strings
Null metadata values are not supported. Instead of setting a key to hold a
null value, we recommend you remove that key from the metadata payload.

For example, the following would be valid metadata payloads:

JSON

{
    "genre": "action",
    "year": 2020,
    "length_hrs": 1.5
}

{
    "color": "blue",
    "fit": "straight",
    "price": 29.99,
    "is_jeans": true
}
​
Supported metadata size
Pinecone supports 40KB of metadata per vector.

​
Metadata query language
Pinecone’s filtering query language is based on MongoDB’s query and projection operators. Pinecone currently supports a subset of those selectors:

Filter	Description	Supported types
$eq	Matches vectors with metadata values that are equal to a specified value.	Number, string, boolean
$ne	Matches vectors with metadata values that are not equal to a specified value.	Number, string, boolean
$gt	Matches vectors with metadata values that are greater than a specified value.	Number
$gte	Matches vectors with metadata values that are greater than or equal to a specified value.	Number
$lt	Matches vectors with metadata values that are less than a specified value.	Number
$lte	Matches vectors with metadata values that are less than or equal to a specified value.	Number
$in	Matches vectors with metadata values that are in a specified array.	String, number
$nin	Matches vectors with metadata values that are not in a specified array.	String, number
$exists	Matches vectors with the specified metadata field.	Boolean
For example, the following has a "genre" metadata field with a list of strings:

JSON

{ "genre": ["comedy", "documentary"] }
This means "genre" takes on both values, and requests with the following filters will match:

JSON

{"genre":"comedy"}

{"genre": {"$in":["documentary","action"]}}

{"$and": [{"genre": "comedy"}, {"genre":"documentary"}]}
However, requests with the following filter will not match:

JSON

{ "$and": [{ "genre": "comedy" }, { "genre": "drama" }] }
Additionally, requests with the following filters will not match because they are invalid. They will result in a compilation error:


# INVALID QUERY:
{"genre": ["comedy", "documentary"]}

# INVALID QUERY:
{"genre": {"$eq": ["comedy", "documentary"]}}
​
Manage high-cardinality in pod-based indexes
For pod-based indexes, Pinecone indexes all metadata by default. When metadata contains many unique values, pod-based indexes will consume significantly more memory, which can lead to performance issues, pod fullness, and a reduction in the number of possible vectors that fit per pod.

To avoid indexing high-cardinality metadata that is not needed for filtering, use selective metadata indexing, which lets you specify which fields need to be indexed and which do not, helping to reduce the overall cardinality of the metadata index while still ensuring that the necessary fields are able to be filtered.

Since high-cardinality metadata does not cause high memory utilization in serverless indexes, selective metadata indexing is not supported.

​
Considerations for serverless indexes
For each serverless index, Pinecone clusters records that are likely to be queried together. When you query a serverless index with a metadata filter, Pinecone first uses internal metadata statistics to exclude clusters that do not have records matching the filter and then chooses the most relevant remaining clusters.

Note the following considerations:

When filtering by numeric metadata that cannot be ordered in a meaningful way (e.g., IDs as opposed to dates or prices), the chosen clusters may not be accurate. This is because the metadata statistics for each cluster reflect the min and max metadata values in the cluster, and min and max are not helpful when there is no meaningful order.

In such cases, it is best to store the metadata as strings instead of numbers. When filtering by string metadata, the chosen clusters will be more accurate, with a low false-positive rate, because the string metadata statistics for each cluster reflect the actual string values, compressed for space-efficiency.

When you use a highly selective metadata filter (i.e., a filter that rejects the vast majority of records in the index), the chosen clusters may not contain enough matching records to satisfy the designated top_k.

For more details about query execution, see Serverless architecture.

​
Use metadata
The following operations support metadata:

Query an index with metadata filters
Insert metadata into an index
Delete vectors by metadata filter
Pinecone Assistant also supports metadata filters. For more information, see Understanding Pinecone Assistant - Filter with metadata.

Was this page helpful?


Yes

No