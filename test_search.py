from document_store import DocumentStore
import json

def main():
    # Initialize DocumentStore with same parameters as in data_prep.py
    data_store = DocumentStore(
        collection_name="documents",
        chroma_db_path="./chroma_db",
        document_content_dir="./document_content",
        embedding_model="intfloat/multilingual-e5-large",
        semantic_weight=0.8,
        bm25_weight=0.2,
        bm25_index_path="./bm25_index",
        device='cuda'
    )

    # Test search functionality
    queries = [
        "What are the symptoms of diabetes?",
    ]

    for query in queries:
        # Debug: Print raw results from semantic and BM25 search
        print(f"\n===== DEBUGGING FOR QUERY: '{query}' =====")
        
        # Get semantic search results directly
        semantic_results = data_store.collection.query(
            query_texts=[f"query: {query}"],
            n_results=5
        )
        print(f"\nSemantic search returned {len(semantic_results['ids'][0])} results")
        print(f"Semantic IDs: {semantic_results['ids'][0]}")
        
        # Get BM25 results directly
        import nltk
        from nltk.tokenize import word_tokenize
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = data_store.bm25_index.get_scores(tokenized_query)
        bm25_results = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        print(f"\nBM25 search returned {len(bm25_results)} results")
        bm25_ids = [data_store.chunk_ids[idx] for idx, _ in bm25_results]
        print(f"BM25 IDs: {bm25_ids}")
        
        # Get combined results
        results = data_store.search(query, n_results=5)
        print(f"\nCombined search returned {len(results['ids'])} results out of requested {5}")
        print(f"Combined IDs: {results['ids']}")
        
        # Debug the _combine_results method
        # Get ranked chunk IDs from semantic search
        ranked_semantic_ids = [(doc_id, idx) for doc_id, idx in zip(
            semantic_results['ids'][0],
            range(len(semantic_results['ids'][0]))
        )]
        
        # Get ranked chunk IDs from BM25
        ranked_bm25_ids = [(data_store.chunk_ids[idx], rank) for rank, (idx, _) in enumerate(bm25_results)]
        
        # Combine results
        chunk_ids = list(set([x[0] for x in ranked_semantic_ids + ranked_bm25_ids]))
        print(f"\nUnique combined IDs before scoring: {len(chunk_ids)}")
        print(f"Unique IDs: {chunk_ids}")
        
        # Calculate scores
        chunk_id_to_score = {}
        for chunk_id in chunk_ids:
            score = 0
            semantic_idx = next((idx for id_, idx in ranked_semantic_ids if id_ == chunk_id), None)
            if semantic_idx is not None:
                score += data_store.semantic_weight * (1 / (semantic_idx + 1))
            
            bm25_idx = next((idx for id_, idx in ranked_bm25_ids if id_ == chunk_id), None)
            if bm25_idx is not None:
                score += data_store.bm25_weight * (1 / (bm25_idx + 1))
            
            chunk_id_to_score[chunk_id] = score
        
        # Sort chunks by score
        sorted_chunks = sorted(
            chunk_id_to_score.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        print(f"\nTop 5 sorted chunks by score:")
        for chunk_id, score in sorted_chunks:
            print(f"  {chunk_id}: {score}")
        
        result_ids = [chunk_id for chunk_id, _ in sorted_chunks]
        print(f"\nFinal result_ids to retrieve: {result_ids}")
        
        # Display normal search results
        print(f"\nSearch results for query '{query}':")
        for i, (doc_id, content) in enumerate(zip(
            results['ids'],
            results['documents']
        ), 1):
            print(f"\n{i}. Document ID: {doc_id}")
            print(f"   Content: {content[:200]}...")

if __name__ == "__main__":
    main()
