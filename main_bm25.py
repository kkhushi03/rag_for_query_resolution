from rag_pipeline_bm25.stage_01_populate_db import run_populate_db_bm25
from rag_pipeline_bm25.stage_02_query_data import run_query_rag
from utils.config import CONFIG

def main():
    print("\n===== STEP 1: Populating BM25 DB =====\n")
    run_populate_db_bm25(reset=False)

    print("\n===== STEP 2: Running BM25 RAG Query =====\n")
    query_text = CONFIG.get("QUERY_TEXT", "What is the main advantage of YieldNet over traditional single-crop yield prediction models using remote sensing data?")
    result = run_query_rag(query=query_text)

    print("\n===== RAG Output =====")
    print(f"\nQuery:\n{query_text}")

    if not isinstance(result, dict):
        print("\nLLM Answer:\nNo valid results returned.")
        print("\nContext Used:\n...")
        print("\nSources:\n[]")
        print("\nTop K Raw Results:\n[]")
        print("\nReranked Results:\n[]")
        print("\nGraded Results:\n[]")
        return

    print(f"\nLLM Answer:\n{result['llms_response']}")
    print(f"\nContext Used:\n{result['context'][:1000]}...")  # Truncated for readability
    print(f"\nSources:\n{result['sources']}")
    print(f"\nTop K Raw Results:\n{[(doc.metadata.get('chunk_id'), score) for doc, score in result['results']]}")
    print(f"\nReranked Results:\n{[(doc.metadata.get('chunk_id'), score) for doc, score in result['reranked_results']]}")
    print(f"\nGraded Results:\n{[(doc.metadata.get('chunk_id'), score, grade) for doc, score, grade in result['graded_results']]}")

if __name__ == "__main__":
    main()
