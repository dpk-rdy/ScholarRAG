import streamlit as st
import os
import shutil
from scholar_search import search_papers
from pdf_processor import download_pdf, process_pdf
from rag_engine import RAGSystem

# Set page config
st.set_page_config(page_title="Scholar RAG", page_icon="📚", layout="wide")

st.title("📚 Google Scholar RAG Assistant")

# Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    if st.button("Clear Database"):
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            st.success("Database cleared!")
            st.rerun()

# check for API key
if not os.environ.get("OPENAI_API_KEY"):
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
    st.stop()

# Initialize Session State
if "papers" not in st.session_state:
    st.session_state.papers = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem()

# Search Section
st.header("1. Search Papers")
query = st.text_input("Enter research topic:")
limit = st.slider("Number of papers to find", 1, 10, 3)

if st.button("Search"):
    with st.spinner("Searching Google Scholar..."):
        results = search_papers(query, limit=limit)
        st.session_state.papers = results
        if not results:
            st.warning("No papers found.")
        else:
            st.success(f"Found {len(results)} papers.")

# Display Results
if st.session_state.papers:
    st.subheader("Search Results")
    selected_papers = []
    
    for i, paper in enumerate(st.session_state.papers):
        with st.expander(f"{paper['title']} ({paper['year']})"):
            st.write(f"**Authors:** {', '.join(paper['authors'])}")
            st.write(f"**Abstract:** {paper['abstract']}")
            if paper['pdf_url']:
                st.write(f"[PDF Link]({paper['pdf_url']})")
                if st.checkbox(f"Select for Analysis", key=f"select_{i}"):
                    selected_papers.append(paper)
            else:
                st.warning("No direct PDF link available.")

    # Process Selection
    if st.button("2. Analyze Selected Papers"):
        if not selected_papers:
            st.warning("Please select at least one paper with a PDF link.")
        else:
            # Re-initialize RAG system to ensure clean start or append
            # For this simple app, let's just append
            rag_system = st.session_state.rag_system
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_papers = len(selected_papers)
            processed_count = 0
            
            for paper in selected_papers:
                status_text.text(f"Downloading/Processing: {paper['title']}...")
                pdf_path = download_pdf(paper['pdf_url'])
                
                if pdf_path:
                    chunks = process_pdf(pdf_path)
                    rag_system.ingest(chunks)
                    processed_count += 1
                
                progress_bar.progress((processed_count) / total_papers)
            
            status_text.text("Ingestion Complete!")
            st.success(f"Successfully processed {processed_count} papers.")

# Chat Interface
st.header("3. Chat with Papers")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_system.query(prompt)
                message_placeholder.markdown(response)
                full_response = response
            except Exception as e:
                st.error(f"Error: {e}")
                
    st.session_state.messages.append({"role": "assistant", "content": full_response})
