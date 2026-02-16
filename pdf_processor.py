import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def download_pdf(url, output_dir="downloaded_papers"):
    """
    Downloads a PDF from a URL to the specified directory.
    Returns the path to the downloaded file, or None if failed.
    """
    if not url:
        return None
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = url.split("/")[-1]
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    
    # Clean filename
    filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).rstrip()
    filepath = os.path.join(output_dir, filename)
    
    # Check if already downloaded
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return filepath
        
    print(f"Downloading PDF from: {url}")
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def process_pdf(pdf_path):
    """
    Loads a PDF and splits it into chunks.
    Returns a list of Document objects.
    """
    print(f"Processing PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    # Test the processor with a dummy PDF if available
    # For now, just print a message
    print("Run this module to test PDF downloading and processing.")
