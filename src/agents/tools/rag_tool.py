from typing import Optional, Any, Dict
from langchain_core.tools import BaseTool
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from core import settings
import logging
import os
from pydantic import Field, ConfigDict
from pathlib import Path

logger = logging.getLogger(__name__)

def get_embeddings() -> Embeddings:
    """Get embeddings model."""
    if settings.OPENAI_API_KEY:
        return OpenAIEmbeddings()
    else:
        raise ValueError("No OpenAI API key available for embeddings. Set OPENAI_API_KEY in your environment.")

class RAGTool(BaseTool):
    """Tool for querying the RAG database."""
    name: str = "rag"
    description: str = """Query the vector store for relevant information.
    Use this tool to search for information about specific topics or questions.
    The 'k' parameter controls how many results to return:
    - 1-2 for simple factual questions
    - 3-5 for questions requiring multiple perspectives
    - 5-10 for comprehensive research questions"""
    
    embeddings: Embeddings = Field(default_factory=get_embeddings)
    vectorstore: Optional[Chroma] = Field(default=None)
    collection_name: str = Field(default="qtk_narratives_and_themes")
    persist_dir: str = Field(default="./data/chromadb")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **kwargs):
        """Initialize the RAG tool."""
        super().__init__(**kwargs)  # Pass kwargs to parent
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Ensure persist_dir is an absolute path
        if not os.path.isabs(self.persist_dir):
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.persist_dir = os.path.join(str(base_dir), self.persist_dir)
        
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize the vector store."""
        try:
            # Import here to avoid circular imports
            import chromadb
            from chromadb.config import Settings

            # Initialize ChromaDB with SQLite
            client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(
                    is_persistent=True,
                    persist_directory=self.persist_dir,
                    anonymized_telemetry=False
                )
            )

            logger.info(f"Attempting to connect to collection '{self.collection_name}'")
            
            # Get the collection
            try:
                collection = client.get_collection(name=self.collection_name)
                logger.info(f"Successfully connected to collection '{self.collection_name}'")
            except ValueError as e:
                logger.error(f"Collection '{self.collection_name}' not found: {str(e)}")
                return

            # Initialize Langchain's Chroma wrapper
            self.vectorstore = Chroma(
                client=client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
            
            # Verify collection has documents
            count = collection.count()
            logger.info(f"Collection size: {count} documents")
            
            if count == 0:
                logger.warning(f"Collection '{self.collection_name}' exists but is empty")
                
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            self.vectorstore = None  # Ensure it's None on failure
            raise  # Re-raise to see the full error in logs
    
    def _run(self, query: str, k: int = 3) -> str:
        """Run the tool to query the vector store."""
        if not self.vectorstore:
            logger.warning("Vector store not initialized, attempting to initialize...")
            self._initialize_vectorstore()
            
        if not self.vectorstore:
            return "Error: Vector store could not be initialized. Please check the logs for details."
        
        try:
            logger.info(f"Querying RAG with: {query}")
            docs = self.vectorstore.similarity_search(query, k=k)
            
            if not docs:
                return "No relevant information found."
            
            formatted_results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                formatted_results.append(f"Result {i} from {source}:\n{doc.page_content}")
                
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            error_msg = f"Error querying the vector store: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def _arun(self, query: str, k: int = 3) -> str:
        """Run the tool asynchronously."""
        return self._run(query, k)

# Initialize the RAG tool with the qtk collection
rag_tool = RAGTool()
