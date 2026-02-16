try:
    import streamlit
    import scholarly
    import langchain
    import langchain_community
    import langchain_openai
    import langchain_chroma
    import chromadb
    import pypdf
    import requests
    print("All dependencies installed successfully!")
except ImportError as e:
    print(f"Missing dependency: {e}")
except Exception as e:
    print(f"Error: {e}")
