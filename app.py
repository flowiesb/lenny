#!/usr/bin/env python3
"""
Lenny's Podcast Knowledge Base - Streamlit RAG Application

A chat interface for product managers to query insights from
302 Lenny's Podcast transcripts using GPT 5.2.
"""

import os
from pathlib import Path
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables (for local development)
load_dotenv()

# Configuration
CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_db"
TOP_K_RESULTS = 8  # Number of chunks to retrieve

# Get API key from Streamlit secrets (cloud) or environment variable (local)
def get_openai_api_key():
    """Get OpenAI API key from Streamlit secrets or environment variable."""
    try:
        # Try Streamlit secrets first (for cloud deployment)
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, AttributeError):
        # Fall back to environment variable (for local development)
        return os.getenv("OPENAI_API_KEY")

# System prompt for PM-focused responses with expert attribution
SYSTEM_PROMPT = """You are an expert product management advisor with access to insights from Lenny's Podcast, 
featuring interviews with the world's best product leaders, founders, and operators.

Your role is to help product managers by synthesizing wisdom from these expert interviews.

CRITICAL INSTRUCTIONS:
1. **Always attribute insights to the specific expert who said them.** Use their name in bold when citing them.
   Example: "**Marty Cagan** emphasizes that empowered product teams are given problems to solve, not features to build."

2. **Select the most relevant experts** from the provided context for each question. Don't force-fit irrelevant sources.

3. **When multiple experts have perspectives on a topic**, synthesize and contrast their views with proper attribution.
   Example: "While **April Dunford** focuses on positioning as the foundation, **Teresa Torres** adds that continuous discovery ensures you're solving the right problems."

4. **Never make claims without attributing them to a specific guest.** If the context doesn't contain relevant information, say so.

5. **For PRD reviews or strategic questions**, evaluate the user's approach against the experts' frameworks and provide specific, actionable feedback.

6. **Be conversational but substantive.** Product managers need practical, implementable advice.

7. **If asked about a topic not covered in the transcripts**, be honest about the limitation rather than making up information.

CONTEXT FROM LENNY'S PODCAST TRANSCRIPTS:
{context}

Remember: Your value is in connecting users to the specific wisdom of these product experts. Always cite who said what."""


def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None


def run_ingestion():
    """Run the ingestion pipeline to create the vector store."""
    import json
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    
    TRANSCRIPTS_DIR = Path(__file__).parent / "Lenny's Podcast Transcripts Archive"
    METADATA_FILE = Path(__file__).parent / "episode_metadata.json"
    
    # Load episode metadata
    episode_metadata = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            episode_metadata = json.load(f)
    
    # Load transcripts
    transcript_files = list(TRANSCRIPTS_DIR.glob("*.txt"))
    documents = []
    
    for filepath in transcript_files:
        guest_name = filepath.stem
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        guest_meta = episode_metadata.get(guest_name, {})
        doc = Document(
            page_content=content,
            metadata={
                "guest": guest_name,
                "source": filepath.name,
                "episode_title": guest_meta.get("episode_title", f"Lenny's Podcast: {guest_name}"),
                "episode_date": guest_meta.get("date", "Unknown"),
                "guest_description": guest_meta.get("guest_description", "Product management expert")
            }
        )
        documents.append(doc)
    
    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=800,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunked_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunked_docs.append(Document(
                page_content=chunk,
                metadata={**doc.metadata, "chunk_index": i}
            ))
    
    # Create vector store
    api_key = get_openai_api_key()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR)
    )
    
    return vector_store


def load_vector_store():
    """Load the ChromaDB vector store, or create it if it doesn't exist."""
    # Set API key for embeddings
    api_key = get_openai_api_key()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    if not CHROMA_PERSIST_DIR.exists():
        return None
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings
    )
    return vector_store


def retrieve_context(vector_store, query: str) -> tuple[str, list[dict]]:
    """Retrieve relevant chunks from the vector store."""
    results = vector_store.similarity_search_with_score(query, k=TOP_K_RESULTS)
    
    context_parts = []
    sources = []
    seen_guests = set()
    
    for doc, score in results:
        guest = doc.metadata.get("guest", "Unknown")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        
        # Format context with clear attribution
        context_parts.append(f"[From {guest}'s interview]:\n{doc.page_content}")
        
        # Track unique sources with episode metadata
        if guest not in seen_guests:
            sources.append({
                "guest": guest,
                "source": doc.metadata.get("source", ""),
                "episode_title": doc.metadata.get("episode_title", f"Lenny's Podcast: {guest}"),
                "episode_date": doc.metadata.get("episode_date", "Unknown"),
                "guest_description": doc.metadata.get("guest_description", "Product management expert"),
                "relevance_score": round(1 - score, 3)  # Convert distance to similarity
            })
            seen_guests.add(guest)
    
    context = "\n\n---\n\n".join(context_parts)
    return context, sources


def generate_response(query: str, context: str) -> str:
    """Generate a response using GPT 5.2."""
    # Ensure API key is set
    api_key = get_openai_api_key()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    llm = ChatOpenAI(
        model="gpt-5.2",
        temperature=0.7
    )
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=query)
    ]
    
    response = llm.invoke(messages)
    return response.content


def render_sources(sources: list[dict]):
    """Render the sources section with episode metadata."""
    with st.expander("📚 Sources Referenced", expanded=False):
        for source in sources:
            st.markdown("---")
            st.markdown(f"### {source.get('episode_title', source['guest'])}")
            st.markdown(f"**Guest:** {source['guest']}")
            if source.get('episode_date') and source['episode_date'] != "Unknown":
                st.markdown(f"**Date:** {source['episode_date']}")
            st.markdown(f"**About:** {source.get('guest_description', 'Product management expert')}")
            st.caption(f"Relevance: {source['relevance_score']:.0%}")


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Lenny's Podcast Knowledge Base",
        page_icon="🎙️",
        layout="centered"
    )
    
    # Custom styling
    st.markdown("""
        <style>
        .stApp {
            max-width: 900px;
            margin: 0 auto;
        }
        .source-card {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Logo
    logo_path = Path(__file__).parent / "recare_logo_rgb_master.png"
    if logo_path.exists():
        logo = Image.open(logo_path)
        # Resize logo to reasonable size (max width 200px)
        logo.thumbnail((200, 200), Image.Resampling.LANCZOS)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(logo, use_container_width=False)
    
    # Header
    st.title("Lenny's Podcast Knowledge Base")
    st.caption("Ask questions and get insights from 302 expert interviews on product management")
    
    # Initialize session state
    init_session_state()
    
    # Clear button (always visible when there are messages)
    if st.session_state.messages:
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    # Check for API key
    api_key = get_openai_api_key()
    if not api_key:
        st.error("⚠️ Please set your `OPENAI_API_KEY` in Streamlit Cloud secrets or `.env` file for local development.")
        st.stop()
    
    # Load vector store
    if st.session_state.vector_store is None:
        with st.spinner("Loading knowledge base..."):
            st.session_state.vector_store = load_vector_store()
    
    # Auto-generate if not found
    if st.session_state.vector_store is None:
        st.info("🔄 Building knowledge base for the first time. This takes about 5-10 minutes...")
        progress_bar = st.progress(0, text="Indexing 302 podcast transcripts...")
        
        try:
            st.session_state.vector_store = run_ingestion()
            progress_bar.progress(100, text="Done!")
            st.success("✅ Knowledge base ready!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to build knowledge base: {e}")
            st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                render_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about product management..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Consulting the experts..."):
                # Retrieve relevant context
                context, sources = retrieve_context(
                    st.session_state.vector_store, 
                    prompt
                )
                
                # Generate response
                response = generate_response(prompt, context)
                
                st.markdown(response)
                render_sources(sources)
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })


if __name__ == "__main__":
    main()
