from pathlib import Path
from typing import List, Dict, Any
from .rag_tool import RAGTool
import logging

logger = logging.getLogger(__name__)

def search_knowledge_base(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search the ChromaDB knowledge base using a query and return the most relevant results.
    
    Args:
        query (str): The search query
        n_results (int): Number of results to return (default: 3)
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing the search results with documents and metadata
    """
    try:
        # Get the absolute path to the data directory
        current_dir = Path(__file__).resolve().parent
        chroma_dir = current_dir.parents[2] / "data" / "chromadb"
        
        # Initialize RAG tool with the correct collection
        rag_tool = RAGTool(
            collection_name="qtk_narratives_and_themes",
            persist_dir=str(chroma_dir)
        )
        
        # Query using the RAG tool
        response = rag_tool._run(query=query, k=n_results)
        
        # Check for errors
        if "error" in response:
            logger.error(f"RAG search error: {response['error']}")
            return []
            
        # Get results
        results = response.get("result", "")
        if not results or "No relevant information found" in results:
            logger.info("No relevant information found in RAG search")
            return []
            
        # Parse the results string into individual results
        formatted_results = []
        result_blocks = results.split("\n\n")
        
        for block in result_blocks:
            if block.strip():
                # Extract source and content from the block
                lines = block.split("\n")
                if len(lines) >= 2:
                    try:
                        # Extract source from the first line
                        first_line = lines[0]
                        if "from" in first_line:
                            source = first_line.split("from")[1].strip()
                        else:
                            source = "Unknown"
                            
                        # Join the remaining lines as content
                        content = "\n".join(lines[1:]).strip()
                        
                        if content:  # Only add if we have content
                            formatted_results.append({
                                'document': content,
                                'metadata': {'source': source}
                            })
                    except Exception as e:
                        logger.error(f"Error parsing result block: {str(e)}")
                        continue
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error in search_knowledge_base: {str(e)}")
        return []

if __name__ == "__main__":
    # Test the search function
    query = "What are the main themes related to customer satisfaction?"
    results = search_knowledge_base(query)
    
    # Print results
    if results:
        print("\nSearch Results:")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Document: {result['document']}")
            print(f"Source: {result['metadata']['source']}")
    else:
        print("\nNo results found.")
