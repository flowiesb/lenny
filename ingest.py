#!/usr/bin/env python3
"""
Ingestion script for Lenny's Podcast Transcripts.
Processes all transcript files, chunks them, generates embeddings,
and stores them in ChromaDB for RAG retrieval.
"""

import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Configuration
TRANSCRIPTS_DIR = Path(__file__).parent / "Lenny's Podcast Transcripts Archive"
CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_db"
METADATA_FILE = Path(__file__).parent / "episode_metadata.json"
CHUNK_SIZE = 1000  # tokens (approximately 4 chars per token)
CHUNK_OVERLAP = 200
BATCH_SIZE = 100  # Documents per batch for embedding
NUM_WORKERS = 5   # Parallel workers for embedding generation


def extract_guest_name(filename: str) -> str:
    """Extract guest name from filename (removes .txt extension)."""
    return filename.replace(".txt", "")


def load_episode_metadata() -> dict:
    """Load episode metadata from JSON file."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}


def load_transcripts() -> list[Document]:
    """Load all transcript files and create Document objects with metadata."""
    documents = []
    episode_metadata = load_episode_metadata()
    
    if not TRANSCRIPTS_DIR.exists():
        raise FileNotFoundError(f"Transcripts directory not found: {TRANSCRIPTS_DIR}")
    
    transcript_files = list(TRANSCRIPTS_DIR.glob("*.txt"))
    print(f"Found {len(transcript_files)} transcript files")
    
    if episode_metadata:
        print(f"Loaded episode metadata for {len(episode_metadata)} guests")
    
    for filepath in transcript_files:
        guest_name = extract_guest_name(filepath.name)
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Get episode metadata if available
        guest_metadata = episode_metadata.get(guest_name, {})
        
        doc = Document(
            page_content=content,
            metadata={
                "guest": guest_name,
                "source": filepath.name,
                "filepath": str(filepath),
                "episode_title": guest_metadata.get("episode_title", f"Lenny's Podcast: {guest_name}"),
                "episode_date": guest_metadata.get("date", "Unknown"),
                "guest_description": guest_metadata.get("guest_description", "Product management expert")
            }
        )
        documents.append(doc)
    
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks while preserving metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 4,  # Approximate chars (4 chars per token)
        chunk_overlap=CHUNK_OVERLAP * 4,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunked_docs = []
    
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        
        for i, chunk in enumerate(chunks):
            chunked_doc = Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            chunked_docs.append(chunked_doc)
    
    return chunked_docs


def create_vector_store(documents: list[Document]) -> Chroma:
    """Create ChromaDB vector store with OpenAI embeddings using parallel workers."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set. Please add it to your .env file.")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    # Remove existing database if present
    if CHROMA_PERSIST_DIR.exists():
        import shutil
        print(f"Removing existing database at {CHROMA_PERSIST_DIR}")
        shutil.rmtree(CHROMA_PERSIST_DIR)
    
    print(f"Creating vector store with OpenAI embeddings ({NUM_WORKERS} parallel workers)...")
    print(f"Processing {len(documents)} chunks in batches of {BATCH_SIZE}...")
    
    # Split documents into batches
    batches = [documents[i:i + BATCH_SIZE] for i in range(0, len(documents), BATCH_SIZE)]
    total_batches = len(batches)
    print(f"Total batches: {total_batches}")
    
    # Create initial empty vector store
    vector_store = Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings
    )
    
    # Process batches in parallel
    completed = 0
    
    def process_batch(batch_idx: int, batch: list[Document]) -> int:
        """Process a single batch of documents."""
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        vector_store.add_texts(texts=texts, metadatas=metadatas)
        return len(batch)
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_batch, i, batch): i 
            for i, batch in enumerate(batches)
        }
        
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                count = future.result()
                completed += 1
                print(f"  Batch {completed}/{total_batches} complete ({count} chunks)")
            except Exception as e:
                print(f"  Batch {batch_idx} failed: {e}")
    
    return vector_store


def main():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("Lenny's Podcast Transcripts - Ingestion Pipeline")
    print("=" * 60)
    
    # Step 1: Load transcripts
    print("\n[1/3] Loading transcripts...")
    documents = load_transcripts()
    print(f"Loaded {len(documents)} transcript documents")
    
    # Step 2: Chunk documents
    print("\n[2/3] Chunking documents...")
    chunked_docs = chunk_documents(documents)
    print(f"Created {len(chunked_docs)} chunks")
    
    # Step 3: Create vector store
    print("\n[3/3] Creating vector store and generating embeddings...")
    vector_store = create_vector_store(chunked_docs)
    
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print(f"Vector store saved to: {CHROMA_PERSIST_DIR}")
    print(f"Total chunks indexed: {len(chunked_docs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
