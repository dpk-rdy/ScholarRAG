# How I Built a Personal AI Research Assistant to Stop Drowning in PDFs

*A deep dive into building a Retrieval-Augmented Generation (RAG) system with Python, and LangChain.*

---

## The 3 AM Realization
It was 3 AM on a Tuesday. I had 47 tabs open. My laptop fan was sounding like a jet engine preparing for takeoff. I was trying to synthesize three different papers on "Sparse Attention Mechanisms in Transformers," and I was failing.

I needed to find *one specific statistic* about memory complexity that I knew I had read somewhere in this digital pile. Ctrl+F was useless because I couldn't remember the exact phrasing. Was it "memory footprint"? "VRAM usage"? "Computational cost"?

I sat back and thought: **Why am I doing manual labor in the age of AI?**

We have Large Language Models (LLMs) that can pass the Bar Exam, yet here I was, playing "Where's Waldo" with academic PDFs. I didn't need a summarize-everything bot; I needed a *dialogue* with my reading list. I wanted to ask, "How does Paper A's approach to sparse attention differ from Paper B's?" and get an answer backed by the text.

So, I built **Scholar RAG**.

In this article, I’ll walk you through how I built a personal academic research assistant that searches Google Scholar, downloads papers, and lets you chat with them. We’ll cover the tech stack, the architecture, and the code that makes it tick.

---

## The Solution: Retrieval-Augmented Generation (RAG)

The core problem with using ChatGPT for research is **hallucination**. If you ask it about a paper published last week, it might not know it exists. If you paste the text, you hit context window limits.

**RAG** solves this. Instead of fine-tuning a model (which is expensive and slow), we keep the model frozen but give it a "cheat sheet" of relevant information for every question.

Here is the workflow I designed for Scholar RAG:
1.  **Search**: Query Google Scholar for the latest research.
2.  **Acquire**: Automatically identify and download Open Access PDFs.
3.  **Ingest**: Split the PDFs into bite-sized chunks and index them in a local vector database.
4.  **Retrieve**: When I ask a question, find the top 5 most relevant chunks.
5.  **Generate**: Send those chunks + the question to GPT-4o, instructing it to answer *only* using the provided context.

## The Tech Stack
*   **Application Logic**: Python 3.9+
*   **Interface**: [Streamlit](https://streamlit.io/) (for a clean, fast UI)
*   **Data Source**: `scholarly` (to scrape Google Scholar)
*   **Orchestration**: [LangChain](https://www.langchain.com/) (the glue holding it all together)
*   **Vector Database**: [ChromaDB](https://www.trychroma.com/) (runs locally, no signup needed)
*   **LLM**: OpenAI `gpt-4o-mini` (fast, cheap, and smart enough for synthesis)

---

## Phase 1: The Search & Scrape
The first challenge was getting the data. I used the `scholarly` library, which acts as a wrapper for Google Scholar. It allows us to search for keywords and retrieve metadata like authors, abstracts, and, crucially, the `eprint_url`—the direct link to the PDF.

In `scholar_search.py`, I implemented a simple search function:

```python
search_query = scholarly.search_pubs("Generative AI in Education")
# Iterate through results
paper = next(search_query)
pdf_url = paper.get('eprint_url')
```

**The Catch**: Not every paper is free. My script filters for results that actually have improved PDF links (Open Access or preprints like arXiv), saving me from the heartbreak of paywalls.

## Phase 2: Processing the PDFs
Downloading a PDF is easy. Making it understandable to an AI is harder.

PDFs are messy. They have headers, footers, and multi-column layouts. In `pdf_processor.py`, I used `PyPDFLoader` from LangChain to extract the raw text. But we can't just dump 50 pages of text into the database. We need to **chunk** it.

I used `RecursiveCharacterTextSplitter` with a chunk size of 1000 characters and an overlap of 200.

```python
# snippet from pdf_processor.py
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,  # Critical for maintaining context across cuts
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
```

**Why Overlap?** Imagine a sentence starts at character 990 and ends at 1020. Without overlap, that sentence gets cut in half, and the meaning is lost. Overlap ensures that the "connective tissue" of the text is preserved.

## Phase 3: The Brain (Vector Store)
This is where the magic happens. We need a way to search for "meaning," not just keywords.

I used **ChromaDB** to store the embeddings. An embedding is a list of numbers (a vector) that represents the semantic meaning of a text chunk. "King" and "Queen" will have vectors that are numerically close to each other, while "King" and "Banana" will be far apart.

In `rag_engine.py`, the system takes the user's question, converts it into an embedding, and queries the database for the nearest neighbors.

```python
# snippet from rag_engine.py
def query(self, question):
    # Retrieve top 5 most similar chunks
    retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
    
    # ... LangChain pipeline to pass context to LLM ...
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain.invoke({"input": question})
```

I chose `k=5` (retrieving 5 chunks) as a sweet spot. Too few, and you miss context. Too many, and you confuse the model (or rack up API costs).

## Phase 4: The Interface
I didn't want to run this in a terminal. I wanted a tool I could actually use. **Streamlit** was the obvious choice. It turns Python scripts into web apps in minutes.

The app (`app.py`) has two main modes:
1.  **Search Mode**: I type a topic, it finds papers, and I tick a box to "Analyze" the ones that look interesting.
2.  **Chat Mode**: Once analyzed, the papers are in the "brain." I can now ask, *"What are the limitations proposed in the third paper?"* and get an immediate answer.

---

## Challenges & Lessons Learned
Building this wasn't all smooth sailing.

1.  **PDF Quality**: Some PDFs are just images of text. `PyPDFLoader` fails on those. I had to accept that my tool works best with "born-digital" PDFs, not scanned ones.
2.  **Rate Limiting**: Google Scholar *hates* bots. I had to add random sleeps (`time.sleep(random.uniform(1, 3))`) in the search loop to avoid getting my IP banned.
3.  **Context Pollution**: Sometimes, the retrieved chunks would be from the bibliography or the copyright footer. This "noise" can confuse the LLM. I learned that data cleaning is 80% of the work in AI engineering.

## Conclusion: The Future of Research
Scholar RAG isn't just a cool weekend project; it's changed how I work. I no longer dread the literature review phase. I can synthesize information faster, find connections I would have missed, and actually *enjoy* the learning process.

The code is open source. You can clone it, plug in your OpenAI key, and start chatting with the collective knowledge of humanity (or at least, the Open Access part of it).

**What's next?** I'm thinking of adding support for local LLMs via Ollama so the whole system can run offline, completely private and free.

*Happy researching!*
