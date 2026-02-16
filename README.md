# Scholar RAG

A Retreival-Augmented Generation (RAG) system that parses research papers from Google Scholar and answers questions based on them.

## Features
- **Search**: Find relevant papers on Google Scholar.
- **Download**: Automatically downloads available PDFs (Open Access).
- **Process**: Extracts text and indexes it into a local vector database.
- **Chat**: Ask questions and get answers based *only* on the paper contents.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenAI API Key**:
   You will need an OpenAI API Key. You can enter it in the application sidebar, or set it as an environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

1. Enter a research topic (e.g., "Transformers in NLP") and click **Search**.
2. Review the found papers. If a PDF link is available, check the "Select for Analysis" box.
3. Click **Analyze Selected Papers**. The system will download and index them.
4. Use the **Chat with Papers** section to ask questions about the research.

## Project Structure
- `app.py`: Main application interface.
- `scholar_search.py`: Google Scholar scraping module.
- `pdf_processor.py`: PDF download and text extraction.
- `rag_engine.py`: RAG logic (Embeddings, Vector Store, Retrieval).
