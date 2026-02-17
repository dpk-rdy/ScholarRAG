# Scholar RAG: Context-Aware Research Assistant for Google Scholar

*How I built a research assistant that connects Google Scholar to GPT-4o.*

---

## The Information Overload Problem
We live in the golden age of AI. Large Language Models (LLMs) can synthesize complex topics, explain difficult concepts, and brainstorm ideas in seconds. However, when it comes to academic research, the workflow often remains surprisingly manual.

You search Google Scholar, download a dozen PDFs, skim through abstracts, and struggle to keep track of which author proposed which methodology. While LLMs are powerful reasoning engines, they face two key challenges in this specific context:
1.  **Scope**: They don't have access to the paper you just found five minutes ago.
2.  **Specifics**: General knowledge is great, but research requires specific, verifiable citations from the text in front of you.

This isn't a failure of the models; it's a gap in the workflow. I built **Scholar RAG** to bridge that gap, combining the reasoning power of modern LLMs with the specific, ground-truth data of academic papers.

Here is how I built a personal research assistant that searches, downloads, and chats with papers using Python, LangChain, and Streamlit.

---

## The Solution: Retrieval-Augmented Generation (RAG)

The concept is simple: instead of asking the LLM to memorize the world, we give it an "open book" exam. We retrieve the relevant pages from our PDFs and say, *"Using only these pages, answer the user's question."*

This approach, known as **Retrieval-Augmented Generation (RAG)**, transforms the LLM from a generalist into a specialist on your specific topic.

My implementation, Scholar RAG, follows this pipeline:
1.  **Search**: Query Google Scholar for the latest research.
2.  **Acquire**: Automatically identify and download Open Access PDFs.
3.  **Ingest**: Process the text and index it into a local vector database.
4.  **Interact**: Chat with the papers, getting answers grounded in the actual text.

## The Tech Stack
*   **Application Logic**: Python 3.9+
*   **Interface**: [Streamlit](https://streamlit.io/) (for a clean, fast UI)
*   **Data Source**: `scholarly` (to scrape Google Scholar)
*   **Orchestration**: [LangChain](https://www.langchain.com/) (for the RAG pipeline)
*   **Vector Database**: [ChromaDB](https://www.trychroma.com/) (local storage)
*   **LLM**: OpenAI `gpt-4o-mini` (for synthesis)

---

## Phase 1: The Search & Scrape
The first step is automating the discovery process. The `scholarly` library is a fantastic wrapper for Google Scholar. It allows us to search for keywords and retrieve metadata like authors, abstracts, and the `eprint_url` (the direct link to the PDF).

In `scholar_search.py`, I implemented a search function that filters for accessible papers:

```python
search_query = scholarly.search_pubs("Generative AI in Education")
# Iterate through results
paper = next(search_query)
pdf_url = paper.get('eprint_url')
```

This saves the tedious step of manually checking 20 different tabs for full-text access.

## Phase 2: Processing the PDFs
Once downloaded, the PDFs need to be converted into a format the AI can understand. I used `PyPDFLoader` from LangChain to extract raw text, but raw text is too long for a single prompt.

I used `RecursiveCharacterTextSplitter` to break the text into 1000-character chunks with a 200-character overlap.

```python
# snippet from pdf_processor.py
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,  # Critical for maintaining context
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
```

The overlap is crucial, it ensures that sentences or ideas aren't cut in half, preserving the semantic meaning across chunk boundaries.

## Phase 3: The Brain (Vector Store)
To find the right chunk for a question, we use **Embeddings**. These are vector representations of text where similar meanings are mathematically close.

I used **ChromaDB** to store these embeddings locally. When I ask, *"What are the limitations of the proposed method?"*, the system:
1.  Converts my question into a vector.
2.  Finds the 5 chunks that are most similar to my question vector.
3.  Passes those chunks to the LLM.

```python
# snippet from rag_engine.py
def query(self, question):
    # Retrieve top 5 most similar chunks
    retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
    
    # ... LangChain pipeline to pass context to LLM ...
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain.invoke({"input": question})
```

## Phase 4: The Interface
I wrapped everything in a **Streamlit** app (`app.py`). It provides a simple dashboard where I can:
1.  Search for a topic.
2.  Select interesting papers for analysis.
3.  Wait for the ingestion (usually seconds).
4.  Start chatting.

The result is a tool that feels like a conversation with a research assistant who has just read the papers for you.

---

## Challenges & Lessons Learned
1.  **PDF Parsing is Hard**: Not all PDFs are created equal. Some are scanned images, others have complex multi-column layouts. Using robust loaders is essential.
2.  **Rate Limiting**: Automated scraping requires being a "good citizen." I added delays between requests to respect Google Scholar's servers.
3.  **Context Relevance**: The quality of the answer depends entirely on the quality of the retrieved chunks. Tuning the chunk size and overlap was key to getting good results.

## Conclusion
Scholar RAG is a practical example of how we can build specialized tools on top of general-purpose LLMs. By connecting the reasoning capabilities of GPT-4o with a specific, curated dataset (the papers), we get the best of both worlds: checking facts against the source text while leveraging AI for synthesis.

The code is available on GitHub. Feel free to clone it and start building your own research assistant.

*Happy researching!*
