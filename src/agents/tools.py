import math
import re
import logging
from typing import List, Dict, Any

import numexpr
import pinecone
import cohere
from langchain_core.tools import BaseTool, tool
from core import settings


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


def format_rag_contexts(matches: List[Dict[str, Any]]) -> str:
    """Format the RAG search results into a readable string.
    
    Args:
        matches: List of matches from Pinecone search
        
    Returns:
        Formatted string containing the search results
    """
    contexts = []
    for match in matches:
        metadata = match.metadata
        context = f"""
**{metadata.get('label', 'N/A')}**

{metadata.get('description', 'N/A')}

Tema Principal: {metadata.get('cluster_main', 'N/A')}
Outros Temas: {metadata.get('cluster_assignments', 'N/A')}
Probabilidade do Tema Principal: {metadata.get('main_cluster_prob', 'N/A')}
Nível de Abstração: {metadata.get('abs', 'N/A')}
Temperatura: {metadata.get('temp', 'N/A')}
        """
        contexts.append(context)
    return "\n\n".join(contexts)


def get_embeddings(text: str) -> List[float]:
    """Get embeddings using Cohere's API v2.
    
    Args:
        text: Text to get embeddings for
        
    Returns:
        List of embeddings
    """
    co = cohere.ClientV2(settings.COHERE_API_KEY.get_secret_value())
    response = co.embed(
        texts=[text],
        model=settings.EMBEDDINGS_MODEL,
        input_type=settings.EMBEDDINGS_INPUT_TYPE,
        embedding_types=["float"]  # Default float embeddings for v3 model
    )
    return response.embeddings.float[0]  # Access float embeddings from the response


def init_pinecone() -> pinecone.Index:
    """Initialize Pinecone client and return index.
    
    Returns:
        Pinecone index instance
    """
    pc = pinecone.Pinecone(
        api_key=settings.PINECONE_API_KEY.get_secret_value(),
        environment=settings.PINECONE_ENVIRONMENT
    )
    return pc.Index(settings.PINECONE_INDEX)


def format_pinecone_filter(filter_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Format metadata filter for Pinecone query.
    
    Args:
        filter_metadata: Raw filter dictionary with metadata fields
            Supports:
            - String values (e.g. cluster_main: "C4") -> {"$in": [value]}
            - Numeric comparisons for temp/abs (e.g. ">2") -> {"$gt": 2.0}
            - Direct values: numbers passed as is
            
    Returns:
        Formatted filter dictionary compatible with Pinecone
    """
    formatted_filter = {}
    
    # Fields that should be treated as numeric with comparisons
    numeric_fields = {"temp", "abs"}
    
    for key, value in filter_metadata.items():
        if key in numeric_fields and isinstance(value, str):
            # Handle numeric comparisons (>2, <1, etc)
            try:
                if value.startswith(">"):
                    if value.startswith(">="):
                        formatted_filter[key] = {"$gte": float(value[2:])}
                    else:
                        formatted_filter[key] = {"$gt": float(value[1:])}
                elif value.startswith("<"):
                    if value.startswith("<="):
                        formatted_filter[key] = {"$lte": float(value[2:])}
                    else:
                        formatted_filter[key] = {"$lt": float(value[1:])}
                elif value.startswith("="):
                    formatted_filter[key] = float(value[1:])
                else:
                    # Try to convert direct number
                    formatted_filter[key] = float(value)
            except ValueError:
                # Skip invalid numeric values
                continue
        elif isinstance(value, str):
            # For strings (like cluster_main), use $in operator
            formatted_filter[key] = {"$in": [value]}
        else:
            # For other types (direct numbers, etc)
            formatted_filter[key] = value
            
    return formatted_filter


@tool("rag_search")
def rag_search(query: str, top_k: int = 3) -> str:
    """Search across ALL documents in the knowledge base without any filters. Use this as the default search option.
    This tool is best for:
    - Exploring broad themes and patterns
    - Getting a general overview of a topic
    - Finding connections between different narrativas
    - When you don't need to filter by specific metadata
    
    Args:
        query: Natural language query to search for
        top_k: Number of top results to return (default: 3)
        
    Returns:
        Relevant information from the knowledge base
    """
    index = init_pinecone()
    query_embedding = get_embeddings(query)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    if not results.matches:
        return "Nenhuma narrativa encontrada."
    
    # Debug: Print first result metadata
    if results.matches:
        print("DEBUG - First result metadata:", results.matches[0].metadata)
        
    return format_rag_contexts(results.matches)


@tool("rag_search_filter")
def rag_search_filter(query: str, filter_metadata: Dict[str, Any], top_k: int = 3) -> str:
    """Search with specific metadata filters. Only use this tool when you need to filter results by specific criteria.
    This tool should be used ONLY when:
    - You need to find narratives from a specific cluster/theme
    - You want to filter by temperature or abstraction level
    - You need to analyze a specific subset of narratives
    - The user explicitly asks for filtered results
    
    Args:
        query: Natural language query to search for
        filter_metadata: Dictionary of metadata fields to filter on
            Can include:
            - cluster_main: cluster code (e.g. "C4")
            - temp: temperature with comparison (e.g. ">2", "<1", "=0")
            - abs: abstraction with comparison (e.g. ">2", "<1", "=0")
        top_k: Number of top results to return (default: 3)
            
    Returns:
        Relevant information from the filtered documents
    """
    index = init_pinecone()
    query_embedding = get_embeddings(query)
    
    # Format the filter using helper function
    formatted_filter = format_pinecone_filter(filter_metadata)
    
    # Debug: print the formatted filter
    print("DEBUG - Formatted filter:", formatted_filter)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=formatted_filter
    )
    
    if not results.matches:
        return "Nenhuma narrativa encontrada com os filtros especificados."
    
    # Debug: Print first result metadata
    if results.matches:
        print("DEBUG - First result metadata:", results.matches[0].metadata)
    
    return format_rag_contexts(results.matches)


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"
