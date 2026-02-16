import time
import random
from scholarly import scholarly

def search_papers(query, limit=5):
    """
    Searches for papers on Google Scholar using the `scholarly` library.
    Returns a list of dictionaries with paper details.
    """
    print(f"Searching for: {query}")
    search_query = scholarly.search_pubs(query)
    papers = []
    
    for _ in range(limit):
        try:
            paper = next(search_query)
            
            # Extract relevant details
            title = paper.get('bib', {}).get('title', 'No Title')
            abstract = paper.get('bib', {}).get('abstract', 'No Abstract')
            authors = paper.get('bib', {}).get('author', [])
            pub_year = paper.get('bib', {}).get('pub_year', 'Unknown')
            eprint_url = paper.get('eprint_url') # Direct PDF link if available
            pub_url = paper.get('pub_url') # Publisher link
            
            paper_data = {
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'year': pub_year,
                'pdf_url': eprint_url,
                'pub_url': pub_url
            }
            papers.append(paper_data)
            print(f"Found: {title}")
            
            # Be polite to Google Scholar
            time.sleep(random.uniform(1, 3))
            
        except StopIteration:
            break
        except Exception as e:
            print(f"Error fetching paper: {e}")
            break
            
    return papers

if __name__ == "__main__":
    # Test the search
    results = search_papers("Generative AI in Education", limit=3)
    for p in results:
        print(f"Title: {p['title']}")
        print(f"PDF: {p['pdf_url']}")
        print("-" * 20)
