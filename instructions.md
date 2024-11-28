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



# Troubleshooting Your Deploy in Render
Sometimes, an app that runs fine locally might fail to deploy to Render at first. When this happens, it’s almost always because of differences between your local development environment and the environment that Render uses to build and run your code.

These environmental differences might include:

The version of your programming language
The values of important environment variables
The availability of particular tools and utilities
The versions of your project’s dependencies
For recommended steps to resolve issues with your app’s Render deploy, see below.

1. Check the logs
Whenever your app misbehaves in any way, always check the logs first. Logs are available in the Render Dashboard:

If a particular deploy fails, view its logs by clicking the word Deploy in your app’s Events feed:

Selecting a deploy to view logs

If your running app encounters an error, open its Logs page to search and filter runtime logs with the explorer:

Log explorer in the Render Dashboard

In either case, searching the log explorer for the word error can often direct you to a relevant log line. If the meaning of the full error message is unclear, try searching the web (or an individual site like Stack Overflow or the Render Community) to help identify a root cause.

Learn more about logging.

2. Ensure matching versions and configuration
Render’s runtime environment might use a different version of your app’s programming language, or of an installed dependency. The values of certain environment variables might also differ from those on your local machine.

Check your app’s configuration for the following:

Runtime mismatches

While creating a service, you select the runtime that corresponds to your language (Node, Python, and so on). There’s also a Docker runtime for projects that build from a Dockerfile or pull a prebuilt image.
If you’ve selected an incorrect runtime for your app, the fastest fix is usually to create a new service with the correct runtime.
You can also change an existing service’s runtime via Render Blueprints or the API. See details.
Version mismatches

Each programming language has a default version on Render, which you can override to match your local machine’s version. See details.
Perform a fresh install of your project on your local machine to confirm that you’re using exactly the dependency versions specified in your repository (such as in your yarn.lock file).
Configuration mismatches

Your local machine might set environment variables as part of your app’s start script, or via a file like .env. Make sure you’re setting necessary environment variables on Render as well.
When applicable, confirm that you’ve set necessary configuration to run your app in “production mode” (e.g., by setting NODE_ENV to production).
To use any tools or utilities besides those included by default in Render’s native runtimes, make that sure you install them as part of your app’s build command.
Confirm that all of your app’s dependencies are compatible with a Linux runtime and file system.
Check your logs to confirm the following:
Your app’s dependencies are all installed as expected.
Your service’s start command runs and completes successfully.
Common errors
Build & deploy errors
Many first-time build and deploy errors are caused by one of the following issues:

Missing or incorrectly referenced resources
Module Not Found / ModuleNotFoundError: Usually indicates one of the following:
A referenced dependency was not found (e.g., in your package.json or requirements.txt file).
A referenced file (such as app.js or app.py) was not found at a specified location.
If you’re developing on Windows or another platform with a case-insensitive filesystem, make sure that all file paths, names, and extensions are cased correctly. You might need to check the contents of your Git repo directly.
Language / dependency version conflicts
SyntaxError: Unexpected token '??=': The app’s Node.js version doesn’t support the indicated operator or method.
The engine "node" is incompatible with this module. Expected version…: The app’s Node.js version doesn’t work with the specified module.
requires Python >= 3.8: A dependency is not compatible with the app’s Python version.
Invalid configuration
Invalid build command: The command that Render runs to install your project’s dependencies and/or perform a build is missing or invalid.
This usually should match the command you run to build your app locally.
Common build commands include npm install (Node.js) and pip install -r requirements.txt (Python).
Invalid start command: The command that Render runs to start your app is missing or invalid.
This usually should match the command you run to start your app locally.
Common start command formats include npm start (Node.js) and gunicorn myapp:app (Python).
Missing environment variables: Some apps require certain environment variables to be set for them to build and start successfully.
Add environment variables to your app in the Render Dashboard, or via a render.yaml blueprint file.
Missing Dockerfile CMD or ENTRYPOINT: If you build and run your app from a Dockerfile, that file must include a CMD or ENTRYPOINT directive.
Render uses one of these directives to run your app after the build completes.
If you omit both of these directives, your deploy might appear to hang indefinitely in the Render Dashboard.
Misconfigured health checks: If you’ve added a health check endpoint to your app, Render uses it to verify that your app is responsive before marking it as live.
If the health check endpoint responds with an unexpected value (or doesn’t respond at all), Render cancels your deploy.
Runtime errors
Many common runtime errors surface as HTTP error codes returned to your browser or other client. For errors returned to your browser, the Network panel of your browser’s developer tools helps provide more details about the error.

Listed below are the most common error codes and some of their most common causes:

400 Bad Request
A Django app doesn’t include its associated custom domain in its ALLOWED_HOSTS setting.
404 Not Found
A static site has misconfigured redirects and/or rewrites.
A web service or static site has misconfigured its routing.
A service is attempting to access a nonexistent file on disk. This might be because:
The file is no longer available because the service doesn’t have a persistent disk.
The service has provided the wrong path (such as by misspelling or incorrectly capitalizing a path component).
A Django app is not correctly serving its static files.
500 Internal Server Error
A service has thrown an uncaught exception while responding to a request, possibly causing the service to crash or restart.
A service is experiencing database connection issues, such as SSL connection has been closed unexpectedly.
In this case, setting sslmode=require and/or a setting up a connection pool can help.
A service or database is overwhelmed, often by too many concurrent connections or constrained resources (such as CPU or RAM).
In this case, warnings about resource constraints usually appear in the service’s logs and on the service’s Events page in the Render Dashboard.
To resolve, consider scaling your service to help alleviate load.
502 Bad Gateway
A web service has misconfigured its host and port.
Bind your host to 0.0.0.0 and optionally set the PORT environment variable to use a custom port (the default port is 10000).
A newly added custom domain is not yet redirecting to its web service.
In most cases this resolves within a few minutes, but it might take up to an hour.
A Node.js web service is experiencing intermittent timeouts or Connection reset by peer errors. Try increasing the values for server.keepAliveTimeout and server.headersTimeout (such as to 120000 for 120 seconds).
A service is experiencing WORKER, SIGKILL, or SIGTERM warnings (e.g., [CRITICAL] WORKER TIMEOUT).
Consider increasing your timeout values and worker limits (e.g., via the gunicorn timeout parameter).
When to contact support
Render’s support team is available and happy to assist with issues that are specific to the capabilities, conventions, and underlying infrastructure of our platform.

Our support team cannot assist with more general development issues like the following:

Debugging of application code
Software design and architecture
Performance optimization
Programming nuances specific to a particular library or framework
For help with issues like these, please consult sites and services that specialize in these forms of assistance.