"""Load and process QTK narratives and themes for the RAG system."""
import os
import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import time
import re
import traceback

def clean_text(text):
    """Clean and normalize text data."""
    if pd.isna(text):
        return ""
    
    # Convert non-string values to strings
    if not isinstance(text, str):
        text = str(text).strip()
    
    # Add space after labels
    text = re.sub(r'Label:(\S)', r'Label: \1', text)
    text = re.sub(r'Description:(\S)', r'Description: \1', text)
    text = re.sub(r'Resumo:(\S)', r'Resumo: \1', text)
    
    # Remove any non-breaking spaces and normalize whitespace
    text = text.replace('\xa0', ' ')
    text = ' '.join(text.split())
    return text

def chunk_data(narratives_df, themes_df, batch_size=5):
    """Chunk the data into batches of 5 documents each."""
    all_texts = []
    all_metadatas = []
    current_batch_texts = []
    current_batch_metadatas = []
    
    def add_to_batch(text, metadata):
        nonlocal current_batch_texts, current_batch_metadatas, all_texts, all_metadatas
        
        # Clean and validate text
        text = clean_text(text)
        if not text:
            return
            
        # Clean metadata values and ensure they are strings
        cleaned_metadata = {}
        for key, value in metadata.items():
            if pd.isna(value):
                cleaned_metadata[key] = ""
            else:
                cleaned_metadata[key] = clean_text(str(value))
        
        # Validate metadata
        required_keys = ["source", "topic", "type", "label"]
        for key in required_keys:
            if key not in cleaned_metadata:
                print(f"Warning: Missing required metadata key '{key}'. Setting to empty string.")
                cleaned_metadata[key] = ""
        
        # Debug output
        print("\nProcessing document:")
        print(f"Text: {text[:100]}...")
        print("Metadata:", cleaned_metadata)
        
        # If current batch is full (5 documents), save it and start new batch
        if len(current_batch_texts) >= batch_size:
            all_texts.append(current_batch_texts)
            all_metadatas.append(current_batch_metadatas)
            current_batch_texts = []
            current_batch_metadatas = []
        
        current_batch_texts.append(text)
        current_batch_metadatas.append(cleaned_metadata)
    
    # Process all narratives
    print("Processing narratives...")
    for idx, row in narratives_df.iterrows():
        label = str(row['Label']).strip() if pd.notna(row['Label']) else ""
        desc = str(row['Description']).strip() if pd.notna(row['Description']) else ""
        if not label and not desc:
            continue
        
        # Create the document text
        text = ""
        if label:
            text += f"Label: {label}\n"
        if desc:
            text += f"Description: {desc}"
        text = text.strip()
        
        # Create metadata ensuring all values are strings
        metadata = {
            "source": "narrativas.txt",
            "topic": str(row['Cluster_main']).strip() if pd.notna(row['Cluster_main']) else "",
            "type": "narrative",
            "label": label
        }
        
        add_to_batch(text, metadata)
    
    # Process all themes
    print("Processing themes...")
    for idx, row in themes_df.iterrows():
        if pd.notna(row['Resumo']):
            label = str(row['Label']).strip() if pd.notna(row['Label']) else ""
            resumo = str(row['Resumo']).strip()
            if not resumo:
                continue
            
            # Create the document text
            text = ""
            if label:
                text += f"Label: {label}\n"
            text += f"Resumo: {resumo}"
            text = text.strip()
            
            # Create metadata ensuring all values are strings
            metadata = {
                "source": "temas.txt",
                "topic": str(row['Cluster']).strip() if pd.notna(row['Cluster']) else "",
                "type": "theme",
                "label": label
            }
            
            add_to_batch(text, metadata)
    
    # Add any remaining texts in the current batch
    if current_batch_texts:
        all_texts.append(current_batch_texts)
        all_metadatas.append(current_batch_metadatas)
    
    return all_texts, all_metadatas

class DebugOpenAIEmbeddingFunction(embedding_functions.OpenAIEmbeddingFunction):
    """Wrapper around OpenAI embedding function to log API calls."""
    def __call__(self, texts):
        try:
            print("\n=== OpenAI API Call ===")
            print(f"Sending {len(texts)} texts to OpenAI API\n")
            
            for i, text in enumerate(texts, 1):
                print(f"Text {i}:")
                print(f"Length: {len(text)} characters")
                print("Content:\n---")
                print(text)
                print("---\n")
            
            print("Calling OpenAI API...")
            embeddings = super().__call__(texts)
            print("[OK] OpenAI API call successful!")
            
            print("\n=== API Response ===")
            print(f"Number of embeddings received: {len(embeddings)}\n")
            
            for i, embedding in enumerate(embeddings, 1):
                print(f"Embedding {i}:")
                print(f"Dimension: {len(embedding)}")
                print(f"First 5 values: {embedding[:5]}\n")
            
            return embeddings
            
        except Exception as e:
            print(f"[ERROR] Error in OpenAI embedding function: {str(e)}")
            print(f"Full traceback:")
            print(traceback.format_exc())
            raise

    def __getstate__(self):
        """Make the embedding function picklable."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Restore the embedding function from pickle."""
        self.__dict__.update(state)

def load_qtk_data():
    """Load QTK narratives and themes data into the vector store."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get the absolute path to the data directory
        current_dir = Path(__file__).resolve().parent
        data_dir = current_dir.parents[2] / "data" / "chroma"
        chroma_dir = current_dir.parents[2] / "data" / "chromadb"
        
        # Clean up existing ChromaDB directory
        if chroma_dir.exists():
            print("Removing existing ChromaDB directory...")
            try:
                import shutil
                shutil.rmtree(chroma_dir)
                print("[OK] Successfully cleaned ChromaDB directory")
            except Exception as e:
                print(f"[ERROR] Error cleaning ChromaDB directory: {str(e)}")
                raise
        
        # Create fresh ChromaDB directory
        try:
            chroma_dir.mkdir(parents=True, exist_ok=True)
            print("[OK] Created fresh ChromaDB directory")
        except Exception as e:
            print(f"[ERROR] Error creating ChromaDB directory: {str(e)}")
            raise
        
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable")
        
        # Initialize OpenAI embeddings with debug wrapper
        print("\nInitializing OpenAI embeddings...")
        try:
            openai_ef = DebugOpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small"
            )
            print("[OK] OpenAI embedding function initialized")
        except Exception as e:
            print(f"[ERROR] Error initializing OpenAI embeddings: {str(e)}")
            raise
        
        # Initialize ChromaDB client
        print("\nConnecting to ChromaDB...")
        try:
            print(f"ChromaDB directory: {chroma_dir}")
            client = chromadb.PersistentClient(path=str(chroma_dir))
            print("[OK] Connected to ChromaDB")
            print(f"ChromaDB version: {chromadb.__version__}")
        except Exception as e:
            print(f"[ERROR] Error connecting to ChromaDB: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Create collection with simpler settings
        try:
            print("\nCreating ChromaDB collection...")
            collection = client.create_collection(
                name="qtk_data",
                embedding_function=openai_ef
            )
            print("[OK] Created ChromaDB collection")
            print(f"Collection name: {collection.name}")
        except Exception as e:
            print(f"[ERROR] Error creating ChromaDB collection: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Load and process data
        print("\nLoading data files...")
        try:
            narratives_df = pd.read_csv(data_dir / "qtk-narrativas - dash.csv", encoding='utf-8')
            themes_df = pd.read_csv(data_dir / "qtk-temas.csv", encoding='utf-8')
            print("[OK] Data files loaded successfully")
            print(f"Found {len(narratives_df)} narratives and {len(themes_df)} themes")
        except Exception as e:
            print(f"[ERROR] Error loading data files: {str(e)}")
            raise
        
        print("\nChunking data...")
        try:
            all_texts, all_metadatas = chunk_data(narratives_df, themes_df)
            print(f"[OK] Created {len(all_texts)} batches")
            print(f"First batch preview:")
            print(f"- Number of texts: {len(all_texts[0])}")
            print(f"- Number of metadatas: {len(all_metadatas[0])}")
            print(f"- First text: {all_texts[0][0][:200]}...")
            print(f"- First metadata: {all_metadatas[0][0]}")
        except Exception as e:
            print(f"[ERROR] Error chunking data: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Process batches
        total_docs = 0
        for batch_idx, (texts, metadatas) in enumerate(zip(all_texts, all_metadatas)):
            print(f"\nProcessing batch {batch_idx + 1}/{len(all_texts)}")
            
            try:
                # Generate unique IDs
                ids = [f"doc_{batch_idx}_{i}" for i in range(len(texts))]
                
                # Debug metadata
                print("\nMetadata inspection for this batch:")
                for i, metadata in enumerate(metadatas):
                    print(f"\nDocument {i + 1} metadata:")
                    for key, value in metadata.items():
                        print(f"  {key}: {repr(value)}")  # Using repr to show exact string content
                    if not all(isinstance(v, str) for v in metadata.values()):
                        print("[WARNING] Not all metadata values are strings!")
                        print("Non-string values:", {k: type(v) for k, v in metadata.items() if not isinstance(v, str)})
                
                # Verify ChromaDB state before adding
                try:
                    print("\nVerifying ChromaDB state...")
                    current_count = collection.count()
                    print(f"Current documents in ChromaDB: {current_count}")
                    
                    # Test query to ensure collection is responsive
                    print("Testing collection with a query...")
                    collection.query(
                        query_texts=["test query"],
                        n_results=1
                    )
                    print("[OK] ChromaDB collection is responsive")
                except Exception as e:
                    print(f"[ERROR] ChromaDB verification failed:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    raise
                
                # Add to collection
                print("\nAdding documents to ChromaDB...")
                try:
                    print(f"Adding batch with:")
                    print(f"- {len(texts)} documents")
                    print(f"- {len(metadatas)} metadata entries")
                    print(f"- {len(ids)} IDs")
                    print("\nIDs to be added:", ids)
                    
                    # Try batch add first
                    try:
                        collection.add(
                            documents=texts,
                            metadatas=metadatas,
                            ids=ids
                        )
                        print("[OK] Batch add successful")
                    except Exception as batch_error:
                        print(f"[ERROR] Batch add failed, trying one at a time:")
                        print(f"Batch error: {str(batch_error)}")
                        
                        # Try adding one at a time
                        for i, (text, metadata, doc_id) in enumerate(zip(texts, metadatas, ids)):
                            try:
                                collection.add(
                                    documents=[text],
                                    metadatas=[metadata],
                                    ids=[doc_id]
                                )
                                print(f"[OK] Added document {i+1}/{len(texts)}")
                            except Exception as e:
                                print(f"[ERROR] Failed to add document {i+1}:")
                                print(f"Error: {str(e)}")
                                print(f"Document: {text[:100]}...")
                                print(f"Metadata: {metadata}")
                                raise
                    
                    # Verify the documents were added
                    new_count = collection.count()
                    print(f"Previous count: {current_count}")
                    print(f"New count: {new_count}")
                    print(f"Added {new_count - current_count} documents")
                    
                except Exception as e:
                    print(f"\n[ERROR] ChromaDB add operation failed:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    print("\nFull traceback:")
                    print(traceback.format_exc())
                    raise
                
                total_docs += len(texts)
                print(f"[OK] Added {len(texts)} documents. Total: {total_docs}")
                
                # Verify collection state
                try:
                    print("\nVerifying collection state after add:")
                    get_result = collection.get(
                        ids=ids,
                        include=["metadatas", "documents"]
                    )
                    print(f"[OK] Successfully retrieved {len(get_result['ids'])} documents")
                except Exception as e:
                    print(f"[WARNING] Failed to verify added documents:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                
                # Sleep between batches to respect rate limits
                if batch_idx < len(all_texts) - 1:
                    print("Sleeping for 3 seconds...")
                    time.sleep(3)
                    
            except Exception as e:
                print(f"\n[ERROR] Error processing batch {batch_idx + 1}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("\nBatch that failed:")
                for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                    print(f"\nDocument {i + 1}:")
                    print(f"Text: {text}")
                    print(f"Metadata: {metadata}")
                raise
        
        print(f"\n[OK] Processing complete!")
        print(f"Added {total_docs} documents to the vector store")
        
    except Exception as e:
        print(f"\n[ERROR] Fatal error in load_qtk_data:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        raise

def load_themes_only():
    """Load only the themes data into the vector store."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get the absolute path to the data directory
        current_dir = Path(__file__).resolve().parent
        data_dir = current_dir.parents[2] / "data" / "chroma"
        chroma_dir = current_dir.parents[2] / "data" / "chromadb"
        
        # Clean up existing ChromaDB directory
        if chroma_dir.exists():
            print("Removing existing ChromaDB directory...")
            try:
                import shutil
                shutil.rmtree(chroma_dir)
                print("[OK] Successfully cleaned ChromaDB directory")
            except Exception as e:
                print(f"[ERROR] Error cleaning ChromaDB directory: {str(e)}")
                raise
        
        # Create fresh ChromaDB directory
        try:
            chroma_dir.mkdir(parents=True, exist_ok=True)
            print("[OK] Created fresh ChromaDB directory")
        except Exception as e:
            print(f"[ERROR] Error creating ChromaDB directory: {str(e)}")
            raise
        
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable")
        
        # Initialize OpenAI embeddings
        print("\nInitializing OpenAI embeddings...")
        try:
            openai_ef = DebugOpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small"
            )
            print("[OK] OpenAI embedding function initialized")
        except Exception as e:
            print(f"[ERROR] Error initializing OpenAI embeddings: {str(e)}")
            raise
        
        # Initialize ChromaDB client
        print("\nConnecting to ChromaDB...")
        try:
            print(f"ChromaDB directory: {chroma_dir}")
            client = chromadb.PersistentClient(path=str(chroma_dir))
            print("[OK] Connected to ChromaDB")
        except Exception as e:
            print(f"[ERROR] Error connecting to ChromaDB: {str(e)}")
            raise
        
        # Create collection
        try:
            print("\nCreating ChromaDB collection...")
            collection = client.create_collection(
                name="qtk_narratives_and_themes",
                embedding_function=openai_ef
            )
            print("[OK] Created ChromaDB collection")
        except Exception as e:
            print(f"[ERROR] Error creating ChromaDB collection: {str(e)}")
            raise
        
        # Load themes data
        print("\nLoading themes data...")
        try:
            themes_df = pd.read_csv(data_dir / "qtk-temas.csv", encoding='utf-8')
            print("[OK] Themes data loaded successfully")
            print(f"Found {len(themes_df)} themes")
        except Exception as e:
            print(f"[ERROR] Error loading themes data: {str(e)}")
            raise
        
        # Process themes one at a time
        total_docs = 0
        for idx, row in themes_df.iterrows():
            if pd.notna(row['Resumo']):
                try:
                    # Prepare document
                    label = str(row['Label']).strip() if pd.notna(row['Label']) else ""
                    resumo = str(row['Resumo']).strip()
                    if not resumo:
                        continue
                    
                    # Create text and metadata
                    text = f"Label: {label}\nResumo: {resumo}" if label else resumo
                    metadata = {
                        "source": "temas.txt",
                        "topic": str(row['Cluster']).strip() if pd.notna(row['Cluster']) else "",
                        "type": "theme",
                        "label": label
                    }
                    doc_id = f"theme_{idx}"
                    
                    print(f"\nProcessing theme {idx + 1}:")
                    print(f"Text preview: {text[:100]}...")
                    print(f"Metadata: {metadata}")
                    
                    # Add to collection
                    collection.add(
                        documents=[text],
                        metadatas=[metadata],
                        ids=[doc_id]
                    )
                    
                    total_docs += 1
                    print(f"[OK] Added theme {idx + 1}")
                    
                    # Sleep briefly between documents
                    if idx < len(themes_df) - 1:
                        time.sleep(1)
                        
                except Exception as e:
                    print(f"\n[ERROR] Error processing theme {idx + 1}:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    raise
        
        print(f"\n[OK] Processing complete!")
        print(f"Added {total_docs} themes to the vector store")
        
    except Exception as e:
        print(f"\n[ERROR] Fatal error in load_themes_only:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        raise

def add_narratives_subset(num_rows=85):
    """Add a subset of narratives to the existing collection."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get the absolute path to the data directory
        current_dir = Path(__file__).resolve().parent
        data_dir = current_dir.parents[2] / "data" / "chroma"
        chroma_dir = current_dir.parents[2] / "data" / "chromadb"
        
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable")
        
        # Initialize OpenAI embeddings
        print("\nInitializing OpenAI embeddings...")
        try:
            openai_ef = DebugOpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small"
            )
            print("[OK] OpenAI embedding function initialized")
        except Exception as e:
            print(f"[ERROR] Error initializing OpenAI embeddings: {str(e)}")
            raise
        
        # Initialize ChromaDB client and get collection
        print("\nConnecting to ChromaDB...")
        try:
            print(f"ChromaDB directory: {chroma_dir}")
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collection = client.get_collection(
                name="qtk_narratives_and_themes",
                embedding_function=openai_ef
            )
            print("[OK] Connected to ChromaDB and got collection")
        except Exception as e:
            print(f"[ERROR] Error connecting to ChromaDB: {str(e)}")
            raise
        
        # Load narratives data
        print("\nLoading narratives data...")
        try:
            narratives_df = pd.read_csv(data_dir / "qtk-narrativas - dash.csv", encoding='utf-8')
            # Take only the first num_rows
            narratives_df = narratives_df.head(num_rows)
            print("[OK] Narratives data loaded successfully")
            print(f"Processing first {num_rows} narratives")
        except Exception as e:
            print(f"[ERROR] Error loading narratives data: {str(e)}")
            raise
        
        # Process narratives one at a time
        total_docs = 0
        for idx, row in narratives_df.iterrows():
            try:
                # Prepare document
                label = str(row['Label']).strip() if pd.notna(row['Label']) else ""
                desc = str(row['Description']).strip() if pd.notna(row['Description']) else ""
                if not label and not desc:
                    continue
                
                # Create text and metadata
                text = ""
                if label:
                    text += f"Label: {label}\n"
                if desc:
                    text += f"Description: {desc}"
                text = text.strip()
                
                metadata = {
                    "source": "narrativas.txt",
                    "topic": str(row['Cluster_main']).strip() if pd.notna(row['Cluster_main']) else "",
                    "type": "narrative",
                    "label": label
                }
                
                # Add to ChromaDB
                collection.add(
                    documents=[text],
                    metadatas=[metadata],
                    ids=[f"narrative_{idx}"]
                )
                total_docs += 1
                print(f"[OK] Added narrative {idx}")
                
            except Exception as e:
                print(f"[ERROR] Error processing narrative {idx}: {str(e)}")
                continue
        
        print(f"\n[OK] Processing complete!")
        print(f"Added {total_docs} narratives to the vector store")
        
        # Check contents
        check_chroma_contents()
        
    except Exception as e:
        print(f"\n[ERROR] Fatal error in add_narratives_subset:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        raise

def add_remaining_narratives():
    """Add the remaining narratives (after index 85) to the existing collection."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get the absolute path to the data directory
        current_dir = Path(__file__).resolve().parent
        data_dir = current_dir.parents[2] / "data" / "chroma"
        chroma_dir = current_dir.parents[2] / "data" / "chromadb"
        
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable")
        
        # Initialize OpenAI embeddings
        print("\nInitializing OpenAI embeddings...")
        try:
            openai_ef = DebugOpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small"
            )
            print("[OK] OpenAI embedding function initialized")
        except Exception as e:
            print(f"[ERROR] Error initializing OpenAI embeddings: {str(e)}")
            raise
        
        # Initialize ChromaDB client and get collection
        print("\nConnecting to ChromaDB...")
        try:
            print(f"ChromaDB directory: {chroma_dir}")
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collection = client.get_collection(
                name="qtk_narratives_and_themes",
                embedding_function=openai_ef
            )
            print("[OK] Connected to ChromaDB and got collection")
        except Exception as e:
            print(f"[ERROR] Error connecting to ChromaDB: {str(e)}")
            raise
        
        # Load narratives data
        print("\nLoading narratives data...")
        try:
            narratives_df = pd.read_csv(data_dir / "qtk-narrativas - dash.csv", encoding='utf-8')
            # Skip the first 85 rows that were already processed
            narratives_df = narratives_df.iloc[85:]
            print("[OK] Narratives data loaded successfully")
            print(f"Processing remaining {len(narratives_df)} narratives")
        except Exception as e:
            print(f"[ERROR] Error loading narratives data: {str(e)}")
            raise
        
        # Process narratives one at a time
        total_docs = 0
        for idx, row in narratives_df.iterrows():
            try:
                # Prepare document
                label = str(row['Label']).strip() if pd.notna(row['Label']) else ""
                desc = str(row['Description']).strip() if pd.notna(row['Description']) else ""
                if not label and not desc:
                    continue
                
                # Create text and metadata
                text = ""
                if label:
                    text += f"Label: {label}\n"
                if desc:
                    text += f"Description: {desc}"
                text = text.strip()
                
                metadata = {
                    "source": "narrativas.txt",
                    "topic": str(row['Cluster_main']).strip() if pd.notna(row['Cluster_main']) else "",
                    "type": "narrative",
                    "label": label
                }
                
                # Add to ChromaDB
                collection.add(
                    documents=[text],
                    metadatas=[metadata],
                    ids=[f"narrative_{idx}"]
                )
                total_docs += 1
                print(f"[OK] Added narrative {idx}")
                
            except Exception as e:
                print(f"[ERROR] Error processing narrative {idx}: {str(e)}")
                continue
        
        print(f"\n[OK] Processing complete!")
        print(f"Added {total_docs} remaining narratives to the vector store")
        
        # Check contents
        check_chroma_contents()
        
    except Exception as e:
        print(f"\n[ERROR] Fatal error in add_remaining_narratives:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        raise

def check_chroma_contents():
    """Check the contents of ChromaDB and compare with source files."""
    try:
        # Get paths
        current_dir = Path(__file__).resolve().parent
        data_dir = current_dir.parents[2] / "data" / "chroma"
        chroma_dir = current_dir.parents[2] / "data" / "chromadb"
        
        # Load source files to get expected counts
        print("\nChecking source files...")
        try:
            narratives_df = pd.read_csv(data_dir / "qtk-narrativas - dash.csv", encoding='utf-8')
            themes_df = pd.read_csv(data_dir / "qtk-temas.csv", encoding='utf-8')
            print(f"Source files contain:")
            print(f"- Narratives: {len(narratives_df)} rows")
            print(f"- Themes: {len(themes_df)} rows")
            total_source = len(narratives_df) + len(themes_df)
            print(f"Total: {total_source} documents expected")
        except Exception as e:
            print(f"[ERROR] Error loading source files: {str(e)}")
            raise
        
        # Check ChromaDB
        print("\nChecking ChromaDB...")
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collections = client.list_collections()
            
            if not collections:
                print("No collections found in ChromaDB!")
                return
                
            collection = client.get_collection(name="qtk_narratives_and_themes")
            count = collection.count()
            print(f"\nChromaDB contains:")
            print(f"- Collection: qtk_narratives_and_themes")
            print(f"- Documents: {count}")
            
            # Get some stats about the documents
            results = collection.get(
                limit=count,
                include=['metadatas']
            )
            
            if results and 'metadatas' in results:
                metadatas = results['metadatas']
                types = {}
                sources = {}
                
                for metadata in metadatas:
                    doc_type = metadata.get('type', 'unknown')
                    source = metadata.get('source', 'unknown')
                    types[doc_type] = types.get(doc_type, 0) + 1
                    sources[source] = sources.get(source, 0) + 1
                
                print("\nDocument types:")
                for doc_type, count in types.items():
                    print(f"- {doc_type}: {count}")
                    
                print("\nSources:")
                for source, count in sources.items():
                    print(f"- {source}: {count}")
            
            # Compare with expected
            themes_only = any(collection.name == "qtk_narratives_and_themes" for collection in collections)
            if themes_only:
                expected_themes = len(themes_df)
                actual_themes = types.get('theme', 0)
                actual_narratives = types.get('narrative', 0)
                
                if actual_themes == expected_themes:
                    print(f"\n[OK] Number of themes is correct: {actual_themes}")
                else:
                    print(f"\n[ERROR] Theme count mismatch!")
                    print(f"Expected themes: {expected_themes}")
                    print(f"Found themes: {actual_themes}")
                
                if actual_narratives > 0:
                    print(f"[OK] Added {actual_narratives} narratives")
                
        except Exception as e:
            print(f"[ERROR] Error checking ChromaDB: {str(e)}")
            raise
        
    except Exception as e:
        print(f"\n[ERROR] Fatal error in check_chroma_contents:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        raise

def rename_collection():
    """Rename the collection from 'themes_only' to 'qtk_narratives_and_themes'."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get the absolute path to the data directory
        current_dir = Path(__file__).resolve().parent
        chroma_dir = current_dir.parents[2] / "data" / "chromadb"
        
        # Initialize ChromaDB client
        print("\nConnecting to ChromaDB...")
        try:
            print(f"ChromaDB directory: {chroma_dir}")
            client = chromadb.PersistentClient(path=str(chroma_dir))
            
            # Check if old collection exists
            collections = client.list_collections()
            old_name = "themes_only"
            new_name = "qtk_narratives_and_themes"
            
            if not any(c.name == old_name for c in collections):
                print(f"[WARNING] Old collection '{old_name}' not found!")
                return
                
            if any(c.name == new_name for c in collections):
                print(f"[WARNING] New collection '{new_name}' already exists!")
                return
            
            # Get the old collection
            old_collection = client.get_collection(name=old_name)
            
            # Get all data from old collection
            count = old_collection.count()
            data = old_collection.get(
                include=['documents', 'metadatas', 'embeddings'],
                limit=count
            )
            
            # Create new collection with same embedding function
            new_collection = client.create_collection(
                name=new_name,
                embedding_function=old_collection._embedding_function
            )
            
            # Add all data to new collection
            if data['ids']:
                new_collection.add(
                    ids=data['ids'],
                    documents=data['documents'],
                    metadatas=data['metadatas'],
                    embeddings=data['embeddings']
                )
            
            # Delete old collection
            client.delete_collection(name=old_name)
            
            print(f"[OK] Successfully renamed collection from '{old_name}' to '{new_name}'")
            print(f"Transferred {count} documents")
            
        except Exception as e:
            print(f"[ERROR] Error connecting to ChromaDB: {str(e)}")
            raise
            
    except Exception as e:
        print(f"\n[ERROR] Fatal error in rename_collection:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Add the project root to Python path
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(project_root))
    
    # Rename the collection
    rename_collection()
