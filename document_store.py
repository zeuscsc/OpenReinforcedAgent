from typing import List, Dict, Any, Iterator, Tuple
import json
import os
import pickle
from pathlib import Path
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch import Tensor
import chromadb
from chromadb.utils import embedding_functions
from datasets import Dataset, load_from_disk
import datasets
from nltk.tokenize import word_tokenize, sent_tokenize
from rank_bm25 import BM25Okapi
import nltk
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK sentence tokenizer."""
    return [s.strip() for s in sent_tokenize(text) if s.strip()]

def get_chunk_context(**metadata) -> str:
    """Generate context from metadata fields.
    
    Supported metadata fields:
    - category (str): Document category
    - umls_semantic_group (str): UMLS semantic group
    - synonyms (List[str]): List of synonyms
    """
    context_parts = []
    
    if metadata.get('category'):
        context_parts.append(f"Category: {metadata['category']}")

    if metadata.get('umls_semantic_group'):
        context_parts.append(f"Semantic Group: {metadata['umls_semantic_group']}")
    
    if metadata.get('synonyms'):
        # Limit to first 3 synonyms to keep context concise
        syn_list = metadata['synonyms'][:3]
        context_parts.append(f"Synonyms: {', '.join(syn_list)}")
    
    return ' | '.join(context_parts)

def chunk_document(
    text: str,
    metadata: Dict[str, Any],
    tokenizer: Any,
    max_chunk_size: int = 512
) -> Iterator[Tuple[str, str]]:
    """Chunk document into sections with metadata-based context prefixes."""
    # Get context and its token length
    context = get_chunk_context(**metadata)
    context_tokens = len(tokenizer.encode(context))
    max_content_tokens = max_chunk_size - context_tokens
    
    # Split into sentences and batch tokenize
    sentences = split_into_sentences(text)
    sentence_tokens = tokenizer(sentences, padding=False, truncation=False)
    token_lengths = {
        sentence: len(tokens) 
        for sentence, tokens in zip(sentences, sentence_tokens['input_ids'])
    }
    
    # Process sentences
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_token_count = token_lengths[sentence]
        
        # Check if adding this sentence would exceed the limit
        if current_length + sentence_token_count > max_content_tokens and current_chunk:
            # Yield current chunk
            yield context, " ".join(current_chunk)
            # Start new chunk with current sentence
            current_chunk = [sentence]
            current_length = sentence_token_count
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_token_count
    
    # Handle remaining text
    if current_chunk:
        yield context, " ".join(current_chunk)

class E5EmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device='cpu'
    ):
        super().__init__()
        self.device = device
        self.model = SentenceTransformer(
            model_name, 
            device=device, 
            tokenizer_kwargs={"padding": True, "truncation": True}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def passage_embedding(self, texts: List[str]) -> List[List[float]]:
        processed_text = [f"passage: {text}" for text in texts]
        embeddings = self.model.encode(processed_text, batch_size=8, normalize_embeddings=True)
        
        return embeddings

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(texts, batch_size=8, normalize_embeddings=True)
        
        return embeddings

class DocumentStore:
    def __init__(
        self,
        collection_name: str,
        chroma_db_path: str,
        document_content_dir: str,
        embedding_model: str = "intfloat/multilingual-e5-large",
        device: str = 'cuda',
        semantic_weight: float = 0.8,
        bm25_weight: float = 0.2,
        bm25_index_path: str = "./bm25_index", 
    ):
        self.client = chromadb.PersistentClient(chroma_db_path)
        self.document_content_dir = Path(document_content_dir)
        self.embedding_function = E5EmbeddingFunction(embedding_model, device=device)
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.bm25_index_path = bm25_index_path
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "hnsw:space":"cosine"
            }
        )

        # Load BM25 index
        if os.path.exists(self.bm25_index_path):
            self.load_bm25_index(self.bm25_index_path)
        else:
            self.bm25_index = None

    def _extract_document_id(self, chunk_id: str) -> str:
        """Extract document ID from chunk ID."""
        return chunk_id.split('_chunk_')[0]

    def _initialize_bm25_index(self, documents: List[str], chunk_ids: List[str]) -> None:
        """Initialize BM25 index with tokenized documents."""
        # Tokenize documents
        tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.chunk_ids = chunk_ids

    def _combine_results(
        self,
        semantic_results: Dict[str, List],
        bm25_results: List[Tuple[int, float]],
        n_results: int
    ) -> Dict[str, List]:
        """Combine semantic and BM25 results using rank fusion."""
        # Get ranked chunk IDs from semantic search
        ranked_semantic_ids = [(doc_id, idx) for doc_id, idx in zip(
            semantic_results['ids'][0],
            range(len(semantic_results['ids'][0]))
        )]
        
        # Get ranked chunk IDs from BM25, converting indices to chunk IDs
        ranked_bm25_ids = [(self.chunk_ids[idx], rank) for rank, (idx, _) in enumerate(bm25_results)]
        # Combine results
        chunk_ids = list(set([x[0] for x in ranked_semantic_ids + ranked_bm25_ids]))
        chunk_id_to_score = {}
        
        # Calculate scores using weighted rank fusion
        for chunk_id in chunk_ids:
            score = 0
            # Add semantic search score
            semantic_idx = next((idx for id_, idx in ranked_semantic_ids if id_ == chunk_id), None)
            if semantic_idx is not None:
                score += self.semantic_weight * (1 / (semantic_idx + 1))
            
            # Add BM25 score
            bm25_idx = next((idx for id_, idx in ranked_bm25_ids if id_ == chunk_id), None)
            if bm25_idx is not None:
                score += self.bm25_weight * (1 / (bm25_idx + 1))
            
            chunk_id_to_score[chunk_id] = score
        
        # Sort chunks by score
        sorted_chunks = sorted(
            chunk_id_to_score.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_results]

        # Reconstruct results in ChromaDB format
        result_ids = [chunk_id for chunk_id, _ in sorted_chunks]
        result = self.collection.get(ids=result_ids)
        
        return result

    def add_documents_from_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 128
    ) -> None:
        """Add documents from the dataset to ChromaDB and BM25 index."""
        # Concatenate train and test splits
        _dataset = dataset
        
        def process_document(example):
            """Process a single document and return its chunks."""
            if not example['document_content']:
                return example
            
            # Get chunks for the document
            doc_chunks = []
            doc_chunk_ids = []
            
            for idx, (context, chunk_text) in enumerate(chunk_document(
                example['document_content'],
                example,
                self.embedding_function.tokenizer,
                509 # minus the 'passage: ' prefix
            )):
                # Combine context and chunk
                full_text = f"{context}\n\n{chunk_text}"
                chunk_id = f"{example['document_id']}_chunk_{idx}"
                
                # Store chunks and IDs in example
                doc_chunks.append(full_text)
                doc_chunk_ids.append(chunk_id)
            
            # Add chunks to example
            example['chunks'] = doc_chunks
            example['chunk_ids'] = doc_chunk_ids
            return example
        
        # First pass: Process all documents and add chunks to dataset
        _dataset = _dataset.map(
            process_document,
            desc="Processing documents",
        )
        
        # Initialize BM25 index with all chunks
        all_chunks = []
        all_chunk_ids = []
        
        for batch_idx in range(0, len(_dataset), batch_size):
            batch = _dataset[batch_idx:batch_idx + batch_size]
            all_chunks.extend([chunk for chunks in batch['chunks'] for chunk in chunks])
            all_chunk_ids.extend([chunk_id for chunk_ids in batch['chunk_ids'] for chunk_id in chunk_ids])
        
        # Find chunks that appear more than once
        chunk_counts = Counter(all_chunks)
        duplicated_chunks = {chunk for chunk, count in chunk_counts.items() if count > 1}
        
        # Remove all instances of duplicated chunks and their corresponding IDs
        unique_chunks = []
        unique_chunk_ids = []
        for chunk, chunk_id in tqdm(zip(all_chunks, all_chunk_ids), desc="Removing duplicated chunks"):
            if not chunk in duplicated_chunks:
                unique_chunks.append(chunk)
                unique_chunk_ids.append(chunk_id)
        
        all_chunks = unique_chunks
        all_chunk_ids = unique_chunk_ids
        
        # Filter dataset to keep only unique chunks
        def filter_unique_chunks(example):
            filtered_chunks = []
            filtered_chunk_ids = []
            for chunk, chunk_id in zip(example['chunks'], example['chunk_ids']):
                if chunk in unique_chunks:
                    filtered_chunks.append(chunk)
                    filtered_chunk_ids.append(chunk_id)
            example['chunks'] = filtered_chunks
            example['chunk_ids'] = filtered_chunk_ids
            return example
        
        # Update dataset with only unique chunks and remove empty documents
        _dataset = _dataset.map(filter_unique_chunks, desc="Filtering unique chunks", num_proc=8)
        _dataset = _dataset.filter(lambda x: len(x['chunks']) > 0)

        # Add unique chunks to ChromaDB
        for batch_idx in tqdm(range(0, len(_dataset), batch_size), desc="Adding chunks to ChromaDB"):
            self.collection.add(
                documents=all_chunks[batch_idx:batch_idx + batch_size],
                embeddings=self.embedding_function.passage_embedding(all_chunks[batch_idx:batch_idx + batch_size]),
                ids=all_chunk_ids[batch_idx:batch_idx + batch_size]
            )

        # Initialize BM25 index
        self._initialize_bm25_index(all_chunks, all_chunk_ids)
        self.save_bm25_index(self.bm25_index_path)

        _dataset = _dataset.train_test_split(0.1)
        _dataset.save_to_disk("./dataset_curated")
        

    def save_bm25_index(self, save_dir: str) -> None:
        """Save BM25 index and chunk mappings to disk.
        
        Args:
            save_dir: Directory to save the index and mappings
        """
        if self.bm25_index is None:
            raise ValueError("BM25 index has not been initialized")
            
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save BM25 index using pickle
        index_path = save_dir / "bm25_index.pkl"
        with open(index_path, 'wb') as f:
            pickle.dump(self.bm25_index, f)
        
        # Save chunk ID mapping
        chunk_ids_path = save_dir / "chunk_ids.json"
        with open(chunk_ids_path, 'w') as f:
            json.dump(self.chunk_ids, f)

    def load_bm25_index(self, load_dir: str) -> None:
        """Load BM25 index and chunk mappings from disk.
        
        Args:
            load_dir: Directory containing the saved index and mappings
        """
        load_dir = Path(load_dir)
        
        # Load BM25 index using pickle
        index_path = load_dir / "bm25_index.pkl"
        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {index_path}")
            
        with open(index_path, 'rb') as f:
            self.bm25_index = pickle.load(f)
        
        # Load chunk ID mapping
        chunk_ids_path = load_dir / "chunk_ids.json"
        if not chunk_ids_path.exists():
            raise FileNotFoundError(f"Chunk ID mapping not found at {chunk_ids_path}")
            
        with open(chunk_ids_path) as f:
            self.chunk_ids = json.load(f)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Dict[str, Any] = None
    ) -> Dict[str, List]:
        """Search using both semantic search and BM25, then combine results."""
        # Semantic search
        semantic_results = self.collection.query(
            query_texts=[f"query: {query}"],
            n_results=n_results,  # Get more results for better fusion
            where=where
        )
        
        # BM25 search
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_results = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True
        )[:n_results]  # Get more results for better fusion
            
        # Combine results
        return self._combine_results(semantic_results, bm25_results, n_results)     