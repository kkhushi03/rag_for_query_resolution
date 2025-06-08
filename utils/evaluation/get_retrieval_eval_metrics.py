import math
from typing import List, Dict


def calc_mrr_score(retrieved_doc_ids: List[str], relevant_doc_ids: List[str]) -> Dict[str, float]:
    """
    Calculate Mean Reciprocal Rank (MRR)
    
    Args:
        retrieved_doc_ids: List of document IDs in retrieval order
        relevant_doc_ids: List of relevant document IDs (ground truth)
    
    Returns:
        Dictionary containing MRR score
    """
    if not retrieved_doc_ids or not relevant_doc_ids:
        return {'mrr': 0.0}
    
    # Convert to sets for faster lookup
    relevant_set = set(relevant_doc_ids)
    
    # Find the rank of the first relevant document
    for rank, doc_id in enumerate(retrieved_doc_ids, 1):
        if doc_id in relevant_set:
            return {'mrr': 1.0 / rank}
    
    # No relevant document found
    return {'mrr': 0.0}


def calc_ndcg_at_k_score(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> Dict[str, float]:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG@k)
    
    Args:
        retrieved_doc_ids: List of document IDs in retrieval order
        relevant_doc_ids: List of relevant document IDs (ground truth)
        k: Number of top documents to consider
    
    Returns:
        Dictionary containing nDCG@k score
    """
    if not retrieved_doc_ids or not relevant_doc_ids:
        return {f'ndcg_at_{k}': 0.0}
    
    # Convert to set for faster lookup
    relevant_set = set(relevant_doc_ids)
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        if doc_id in relevant_set:
            # Binary relevance: relevant=1, non-relevant=0
            # DCG formula: sum(rel_i / log2(i + 2)) for i from 0 to k-1
            dcg += 1.0 / math.log2(i + 2)
    
    # Calculate IDCG (Ideal DCG) - best possible DCG
    # This is DCG when all relevant docs are at the top
    num_relevant = min(len(relevant_doc_ids), k)
    idcg = 0.0
    for i in range(num_relevant):
        idcg += 1.0 / math.log2(i + 2)
    
    # Calculate nDCG
    if idcg == 0:
        ndcg = 0.0
    else:
        ndcg = dcg / idcg
    
    return {f'ndcg_at_{k}': ndcg}


def calc_precision_at_k_score(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> Dict[str, float]:
    """
    Calculate Precision@k
    
    Args:
        retrieved_doc_ids: List of document IDs in retrieval order
        relevant_doc_ids: List of relevant document IDs (ground truth)
        k: Number of top documents to consider
    
    Returns:
        Dictionary containing Precision@k score
    """
    if not retrieved_doc_ids or not relevant_doc_ids:
        return {f'precision_at_{k}': 0.0}
    
    # Convert to set for faster lookup
    relevant_set = set(relevant_doc_ids)
    
    # Count relevant documents in top-k retrieved documents
    top_k_retrieved = retrieved_doc_ids[:k]
    relevant_retrieved = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)
    
    # Precision@k = (relevant docs in top-k) / k
    precision_at_k = relevant_retrieved / k
    
    return {f'precision_at_{k}': precision_at_k}


def calc_recall_at_k_score(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> Dict[str, float]:
    """
    Calculate Recall@k
    
    Args:
        retrieved_doc_ids: List of document IDs in retrieval order
        relevant_doc_ids: List of relevant document IDs (ground truth)
        k: Number of top documents to consider
    
    Returns:
        Dictionary containing Recall@k score
    """
    if not retrieved_doc_ids or not relevant_doc_ids:
        return {f'recall_at_{k}': 0.0}
    
    # Convert to set for faster lookup
    relevant_set = set(relevant_doc_ids)
    
    # Count relevant documents in top-k retrieved documents
    top_k_retrieved = retrieved_doc_ids[:k]
    relevant_retrieved = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)
    
    # Recall@k = (relevant docs in top-k) / (total relevant docs)
    recall_at_k = relevant_retrieved / len(relevant_doc_ids)
    
    return {f'recall_at_{k}': recall_at_k}


def calc_all_retrieval_scores(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], at_k: int) -> Dict[str, float]:
    """
    Calculate all retrieval evaluation scores
    
    Args:
        retrieved_doc_ids: List of document IDs in retrieval order
        relevant_doc_ids: List of relevant document IDs (ground truth)
        k: Number of top documents to consider for @k metrics
    
    Returns:
        Dictionary containing all retrieval scores
    """
    scores = {}
    
    # MRR
    scores.update(calc_mrr_score(retrieved_doc_ids, relevant_doc_ids))
    
    # nDCG@k
    scores.update(calc_ndcg_at_k_score(retrieved_doc_ids, relevant_doc_ids, at_k))
    
    # Precision@k
    scores.update(calc_precision_at_k_score(retrieved_doc_ids, relevant_doc_ids, at_k))
    
    # Recall@k
    scores.update(calc_recall_at_k_score(retrieved_doc_ids, relevant_doc_ids, at_k))
    
    return scores


# # test the utils
# if __name__ == "__main__":
#     # Test all metrics
#     predicted = "The king can move one square in any direction horizontally, vertically, or diagonally."
#     ground_truth = "A king moves one square in any direction: horizontally, vertically, or diagonally."
#     context = "Chess piece movement rules: The king is the most important piece. A king moves one square in any direction - horizontally, vertically, or diagonally. The king cannot move into check."
#     retrieved_doc_ids = [
#         "doc_7", "doc_3", "doc_5", "doc_2", "doc_1"
#     ]
#     relevant_doc_ids = [
#         "doc_2", "doc_5", "doc_9"
#     ]
#     k = 5
    
#     print("=== Testing All Generation Metrics ===")
#     print(f"Predicted: {predicted}")
#     print(f"Ground Truth: {ground_truth}")
#     print(f"Context: {context[:100]}...")
#     print(f"Retrieved Docs: {retrieved_doc_ids}")
#     print(f"Relevant Docs: {relevant_doc_ids}")
#     print()
    
#     retrieval_scores = calc_all_retrieval_scores(retrieved_doc_ids, relevant_doc_ids, k=k)
#     for metric, score in retrieval_scores.items():
#         print(f"{metric}: {score:.3f}")