import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    def ingest(self, chunks):
        """
        Adds document chunks to the vector store.
        """
        if not chunks:
            return
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        self.vector_store.add_documents(documents=chunks)
        print("Ingestion complete.")
        
    def query(self, question):
        """
        Answers a question based on the indexed documents.
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        system_prompt = (
            "You are an assistant for request answering questions based on academic papers. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": question})
        return response["answer"]
        
    def clear_database(self):
        """
        Clears the vector store.
        """
        self.vector_store.delete_collection()
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print("Database cleared.")
