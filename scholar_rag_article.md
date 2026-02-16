# Scholar RAG: A Personal Academic Research Assistant

*Stop drowning in tabs. Start chatting with your research.*

---

In the age of information overload, academic research can feel like drinking from a firehose. You start with one paper, follow a citation, search for a related concept, and before you know it, you have 50 tabs open and no idea where you read that one specific statistic.

What if you could just **ask** your collection of papers a question?

Enter **Scholar RAG**, a project I built to streamline the literature review process using **Retrieval-Augmented Generation (RAG)**. This tool searches Google Scholar, downloads open-access PDFs, builds a local knowledge base, and lets you chat with the papers using an LLM.

Here is how it works and how I built it.

## The Problem: The "Ctrl+F" Limit
Traditional research involves downloading PDFs and using "Ctrl+F" to find keywords. This works for specific terms but fails for conceptual questions like, *"How do recent transformer models handle long-context dependencies?"* or *"Compare the results of Paper A and Paper B."*

To answer these, you need to read, synthesize, and connect dots—a perfect task for an LLM, provided it has access to the right data.

## The Solution: Scholar RAG
Scholar RAG is a Streamlit application that automates the research workflow:
1.  **Search**: It queries Google Scholar for your topic.
2.  **Acquire**: It identifies and downloads available PDFs (Open Access).
3.  **Digest**: It processes the text and embeds it into a vector database.
4.  **Interact**: It provides a chat interface to answer questions based *strictly* on the downloaded content.

## Under the Hood: The Tech Stack
The project relies on a modern Python AI stack:

*   **UI/UX**: [Streamlit](https://streamlit.io/) for a clean, interactive dashboard.
*   **Data Source**: `scholarly` library to scrape Google Scholar results.
*   **Orchestration**: [LangChain](https://www.langchain.com/) to manage the RAG pipeline.
*   **Vector Store**: [ChromaDB](https://www.trychroma.com/) for local, persistent embedding storage.
*   **Intelligence**: OpenAI's `gpt-4o-mini` for reasoning and `text-embedding-3-small` for semantic search.

### Step 1: sourcing the Data
We use the `scholarly` package to fetch metadata. It allows us to get titles, authors, abstracts, and most importantly, the `eprint_url` (the direct link to the PDF).

```python
# snippet from scholar_search.py
search_query = scholarly.search_pubs("Generative AI in Education")
paper = next(search_query)
pdf_url = paper.get('eprint_url')
```

### Step 2: The RAG Pipeline
Once we have the PDFs, the magic happens in `rag_engine.py`. We don't just feed the entire PDF to the LLM (which would be slow and expensive). Instead, we chunk the text and create **embeddings**—numerical representations of the text's meaning.

We use **ChromaDB** to store these embeddings locally. When you ask a question, the system:
1.  Converts your question into an embedding.
2.  Finds the most mathematically similar chunks in the database (Vector Search).
3.  Sends those chunks + your question to GPT-4o-mini.

```python
# snippet from rag_engine.py
def query(self, question):
    # Find relevant context
    retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Generate answer using LangChain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": question})
    return response["answer"]
```

## Why This Matters
Tools like this democratize advanced research capabilities. You do not need an enterprise license or a massive server farm. With a few hundred lines of Python code, you can build a personalized research assistant that lives on your laptop and grows with your interests.

## Future Improvements
The current version is a solid MVP, but there is always room to grow:
*   **Multi-Source Search**: Integrating ArXiv or PubMed APIs.
*   **Citation Tracking**: ensuring the LLM explicitly cites which paper a fact came from.
*   **Local LLMs**: Swapping OpenAI for a local model using Ollama for complete privacy.

## Conclusion
Scholar RAG allows you to turn a static folder of PDFs into an interactive knowledge base. It is a practical example of how RAG can solve real-world information retrieval problems.

*Check out the code on GitHub and start chatting with your research today!*
