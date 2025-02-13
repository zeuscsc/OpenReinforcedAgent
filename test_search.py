from document_store import DocumentStore

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
        "What are the risk factors for heart disease?",
        "How is cancer treated?",
        "What are common mental health disorders?",
    ]

    for query in queries:
        results = data_store.search(query, n_results=5)
        print(f"\nSearch results for query '{query}':")
        for i, (doc_id, content) in enumerate(zip(
            results['ids'],
            results['documents']
        ), 1):
            print(f"\n{i}. Document ID: {doc_id}")
            print(f"   Content: {content[:500]}...")

if __name__ == "__main__":
    main()
