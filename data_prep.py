import os
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Set
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset
from document_store import DocumentStore

def scrape_web_content(url: str) -> Optional[str]:
    """
    Scrape content from a web URL
    
    Args:
        url: URL to scrape
        
    Returns:
        Scraped text content or None if scraping failed
    """
    try:
        response = requests.get(url)
        if response.status_code >= 300:
            return None
        soup = BeautifulSoup(response.text, "html.parser")
    except:
        return None
        
    # Try to find the main content
    text = soup.find("article")
    if text is None:
        text = soup.find('div', id='mplus-content')
    if text is None:
        return None
        
    # Clean up the text
    text = text.get_text()
    text = re.sub(r'\n+', '\n', text)

    # In case some webpage returned a 200 but is actually a 404
    if "404" in text:
        return None
    
    return text

def get_scraped_urls(content_dir: str) -> Set[str]:
    """
    Get set of URLs that have already been scraped
    
    Args:
        content_dir: Directory containing scraped content JSON files
        
    Returns:
        Set of URLs that have been scraped
    """
    content_path = Path(content_dir)
    scraped_urls = set()
    
    # Load URLs from each JSON file
    for json_file in content_path.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                content = json.load(f)
                scraped_urls.update(content.keys())
        except Exception as e:
            print(f"Error loading {json_file}: {str(e)}")
            
    return scraped_urls

def load_dataset_urls(
    dataset_name: str = "lavita/MedQuAD",
    cache_dir: str = "./dataset"
) -> Set[str]:
    """
    Load dataset and extract unique URLs
    
    Args:
        dataset_name: Name of the dataset to load
        cache_dir: Directory to cache the dataset
        
    Returns:
        Set of unique URLs from the dataset
    """
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    return set(dataset['train']['document_url'])

def get_next_batch_index(content_dir: str) -> int:
    """
    Get the next available batch index by finding the highest existing index
    
    Args:
        content_dir: Directory containing batch files
        
    Returns:
        Next available batch index
    """
    content_path = Path(content_dir)
    max_index = 0
    
    # Find the highest index in existing files
    for json_file in content_path.glob("*.json"):
        try:
            # Extract index from filename (remove .json extension)
            index = int(json_file.stem)
            max_index = max(max_index, index)
        except ValueError:
            continue
            
    return max_index + 1

def scrape_documents(
    urls: Set[str],
    content_dir: str,
    batch_size: int = 100,
    skip_existing: bool = True
) -> None:
    """
    Scrape content from URLs and save to files
    
    Args:
        urls: Set of URLs to scrape
        content_dir: Directory to save scraped content
        batch_size: Number of documents to process before saving
        skip_existing: Whether to skip URLs that have already been scraped
    """
    content_path = Path(content_dir)
    content_path.mkdir(parents=True, exist_ok=True)
    
    # Get already scraped URLs
    scraped_urls = get_scraped_urls(content_dir) if skip_existing else set()
    urls_to_scrape = urls - scraped_urls
    
    if not urls_to_scrape:
        print("All URLs have already been scraped")
        return
        
    print(f"Found {len(urls_to_scrape)} URLs to scrape")
    
    # Get the next available batch index
    next_batch_index = get_next_batch_index(content_dir)
    
    # Scrape content for each URL
    document_content: Dict[str, Any] = {}
    for i, url in tqdm(enumerate(urls_to_scrape), desc="Scraping documents"):
        content = scrape_web_content(url)
        document_content[url] = content
        
        # Save batch of documents
        if (i % batch_size == 0 and i != 0) or i == len(urls_to_scrape) - 1:
            batch_file = content_path / f"{next_batch_index}.json"
            with open(batch_file, "w") as f:
                json.dump(document_content, f)
            document_content = {}
            next_batch_index += 1

def load_or_scrape_documents(
    dataset_name: str = "lavita/MedQuAD",
    cache_dir: str = "dataset",
    content_dir: str = "document_content",
    batch_size: int = 100,
    skip_existing: bool = True
) -> Dataset:
    """
    Load dataset and scrape document content if needed
    
    Args:
        dataset_name: Name of the dataset to load
        cache_dir: Directory to cache the dataset
        content_dir: Directory to store scraped content
        batch_size: Number of documents to process before saving
        skip_existing: Whether to skip URLs that have already been scraped
        
    Returns:
        Dataset containing the loaded documents
    """
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)

    dataset['train'] = dataset['train'].filter(lambda x: x['answer'] is not None)
    
    # Load dataset URLs and convert to set
    urls = set(dataset['train']['document_url'])
    
    # Scrape documents if needed
    scrape_documents(urls, content_dir, batch_size, skip_existing)
    
    # load all scraped content
    document_content = {}
    for i in os.listdir('./document_content'):
        with open(f"./document_content/{i}", 'r') as f:
            document_content.update(json.load(f))
    
    # Add document content and generate unique IDs
    def modify_id_and_content(example, idx):
        """Add unique integer ID to example."""
        example['document_id'] = str(idx).zfill(7)  # 7-digit zero-padded ID
        example['document_content'] = document_content.get(example['document_url'])
        return example
    
    dataset['train'] = dataset['train'].map(
        modify_id_and_content,
        with_indices=True,
        desc="Adding unique IDs and scraped content"
    )
    
    # Filter out documents with no content
    dataset['train'] = dataset['train'].filter(lambda x: x['document_content'] is not None)

    # Initialize and return the processed dataset
    return dataset['train']

if __name__ == "__main__":
    # Load dataset and scrape documents
    dataset = load_or_scrape_documents()
    
    data_store = DocumentStore(
        collection_name="documents",
        document_content_dir="./document_content",
        embedding_model="intfloat/multilingual-e5-large",
        chroma_db_path="./chroma_db",
        bm25_index_path="./bm25_index"
    )

    data_store.add_documents_from_dataset(dataset)

    print("Done")