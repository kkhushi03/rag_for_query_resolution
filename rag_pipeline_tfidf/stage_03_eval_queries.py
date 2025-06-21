import os, json, traceback, time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from utils.config import CONFIG
from utils.logger import setup_logger
from rag_pipeline_tfidf.stage_02_query_data import query_rag
from utils.evaluation.get_retrieval_eval_metrics import calc_all_retrieval_scores
from utils.evaluation.get_generation_eval_metrics import calc_all_generation_scores
from utils.evaluation.save_n_update_results import append_new_results_to_csv


# configurations
LOG_PATH = Path(CONFIG["LOG_PATH"])
TFIDF_DB_DIR = Path(CONFIG["TFIDF_DB_DIR"])
TFIDF_DB_PATH = Path(CONFIG["TFIDF_DB_PATH"])
TFIDF_META_PATH = Path(CONFIG["TFIDF_META_PATH"])
EVALUATION_DATA_PATHS = CONFIG.get("EVALUATION_DATA_PATHS", [])
RESULTS_CSV_PATH = CONFIG.get("RESULTS_CSV_PATH", "evaluation_results.csv")

K = CONFIG["K"]
R = CONFIG["R"]
MAX_N = CONFIG["MAX_N"]

EMBEDDING_MODEL = CONFIG["EMBEDDING_MODEL"]
LOCAL_LLM = CONFIG["LOCAL_LLM"]

# Setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "stage_03_evaluate_rag.log")


def load_evaluation_data(file_path: str, logger) -> List[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        logger.info(f"[Stage 03, Part 01] Loaded evaluation data from {file_path}")
    except FileNotFoundError:
        logger.warning(f"[Stage 03, Part 01] Evaluation data file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"[Stage 03, Part 01] Error parsing JSON file: {e}")
        logger.debug(traceback.format_exc())
        return []

def evaluate_single_query(question: str, ground_truth: str, relevant_doc_ids: List[str], tfidf_db_dir: Path, tfidf_db_path: Path, tfidf_metadata_path: Path, at_k: int, at_r: int, max_n: int, logger) -> Dict[str, Any]:
    start_time = time.time()

    try:
        logger.info(f"[Stage 03, Part 02] Processing question: {question}")
        rag_result = query_rag(query_text=question, tfidf_db_dir=tfidf_db_dir, tfidf_db_path=tfidf_db_path, tfidf_metadata_path=tfidf_metadata_path, at_k=at_k, at_r=at_r, logger=logger)

        predicted_answer = rag_result.get('llms_response', '')
        context = rag_result.get('context', '')
        retrieved_docs = rag_result.get('results', []) 
        graded_results = rag_result.get('graded_results', []) # list of (doc, sim_score) 
        
        logger.info(f" [Stage 03, Part 02.1(a)] Predicted answer: {predicted_answer}")
        logger.debug(f" [Stage 03, Part 02.1(b)] Context: {context}")
        logger.info(f" [Stage 03, Part 02.1(c)] Retrieved docs: {len(retrieved_docs)}")
        logger.info(f" [Stage 03, Part 02.1(d)] Graded docs: {len(graded_results)}")
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        
        logger.info("[Stage 03, Part 02.2] Extracting retrieved graded document IDs in order......")
        graded_retrieved_doc_ids = [doc.metadata.get("chunk_id", "") for doc, _, _ in graded_results]
        logger.info(f"[Stage 03, Part 02.2] Retrieved graded doc IDs: {graded_retrieved_doc_ids}")
        
        logger.info(f"[Stage 03, Part 02.3] Calculating retrieval & generation scores for query: {question}")
        retrieval_scores = calc_all_retrieval_scores(graded_retrieved_doc_ids, relevant_doc_ids, at_r)
        generation_scores = calc_all_generation_scores(predicted_answer, ground_truth, context, max_n)
        
        result = {
            'question': question,
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'context': context[:200] + "..." if len(context) > 200 else context,  # Truncate for CSV
            'graded_retrieved_doc_ids': graded_retrieved_doc_ids[:5],  # Top 5 for CSV
            'relevant_doc_ids': relevant_doc_ids,
            'latency_ms': latency_ms,
            'success': True,
            'error': None
        }
        
        logger.info(f"[Stage 03, Part 02.3] Adding retrieval & generation scores to result......")
        result.update(retrieval_scores)
        result.update(generation_scores)
        
        logger.info(f"[Stage 03, Part 02.3] Successfully calculated retrieval & generation scores for query: {question}")
        return result
        
    except Exception as e:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        logger.error(f"[Stage 03, Part 02] Error: {e}")
        logger.error(f"Latency: {latency_ms} ms")
        logger.debug(traceback.format_exc())
        
        result = {
            'question': question,
            'predicted_answer': '',
            'ground_truth': ground_truth,
            'graded_retrieved_doc_ids': [],
            'relevant_doc_ids': relevant_doc_ids,
            'latency_ms': latency_ms,
            'success': False,
            'error': str(e)
        }
        
        # Add zero scores for all metrics
        zero_retrieval_scores = {
            'mrr': 0.0,
            'ndcg_at_5': 0.0,
            'precision_at_5': 0.0,
            'recall_at_5': 0.0
        }
        zero_generation_scores = {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'exact_match': 0.0,
            'rouge_l_precision': 0.0, 'rouge_l_recall': 0.0, 'rouge_l_f1': 0.0,
            'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0, 'bleu': 0.0,
            'faithfulness': 0.0
        }
        
        result.update(zero_retrieval_scores)
        result.update(zero_generation_scores)
        return result
        
    finally:
        logger.info(f"[Stage 03, Part 02] Finished processing question & generating eval. scores")
        # Log the result
        logger.info(json.dumps(result))


def run_evaluation(eval_dir=EVALUATION_DATA_PATHS, result_dir=RESULTS_CSV_PATH, tfidf_db_dir=TFIDF_DB_DIR, tfidf_db_path=TFIDF_DB_PATH, tfidf_metadata_path=TFIDF_META_PATH, at_k=K, at_r=R, max_n=MAX_N, embedding_model=EMBEDDING_MODEL, local_llm=LOCAL_LLM):
    logger = setup_logger("evaluation_logger", LOG_FILE)
    logger.info(" ")
    logger.info("++++++++[Pipeline 3] Starting RAG Evaluation.....")
    
    # if the questions are in a single file
    logger.info(f"[Stage 03] Loading evaluation data from {eval_dir}......")
    eval_data = load_evaluation_data(eval_dir, logger)
    if not eval_data:
        logger.error("[Stage 03] No evaluation data found. Exiting.")
        return
    
    logger.info(f"[Stage 03] Loaded {len(eval_data)} evaluation questions")
    
    # # if the questions are in multiple files
    # logger.info(f"[Stage 03] Loading & combining evaluation data from {eval_dir}......")
    # eval_data = []
    # for path in eval_dir:
    #     logger.info(f"[Stage 03] Loading evaluation data from {path}......")
    #     data = load_evaluation_data(path, logger)
    #     if not data:
    #         logger.warning(f"[Stage 03] No valid data found in {path}. Skipping.....")
    #         continue
    #     for item in data:
    #         item['source_file'] = os.path.basename(path)
    #         logger.debug("[Stage 03] Tracking source file for each question......")
    #     eval_data.extend(data)

    # if not eval_data:
    #     logger.error("[Stage 03] No evaluation data found from any source. Exiting.")
    #     return

    logger.info(f"[Stage 03] Total combined evaluation questions: {len(eval_data)}")
    logger.info("[Stage 03] Evaluating each question......")
    results = []
    for i, item in enumerate(eval_data):
        logger.info(f"[Stage 03] Evaluating question {i+1}/{len(eval_data)}: {item['question'][:50]}......")
        
        logger.info("[Stage 03] Get relevant_doc_ids from the evaluation data......")
        relevant_doc_ids = item.get('relevant_doc_ids', [])
        if not relevant_doc_ids:
            logger.warning(f"[Stage 03] No relevant_doc_ids found for question {i+1}. Using empty list.")
        
        result = evaluate_single_query(
            question=item['question'],
            ground_truth=item['ground_truth'],
            relevant_doc_ids=relevant_doc_ids,
            tfidf_db_dir=tfidf_db_dir,
            tfidf_db_path=tfidf_db_path,
            tfidf_metadata_path=tfidf_metadata_path,
            at_k=at_k,
            at_r=at_r,
            max_n=max_n,
            logger=logger
        )
        
        logger.info("[Stage 03] Adding metadata to the results......")
        result.update({
            'timestamp': datetime.now().isoformat(),
            'embedding_model_name': embedding_model,
            'gen_model_name': local_llm,
            'q_id': i,
            'difficulty': item.get('difficulty', 'unknown'),
            'correctness': item.get('correctness', 'unknown'),
        })
        results.append(result)
        logger.info("[Stage 03] Added metadata to the results.")
        
        logger.info(" ")
        logger.info("[Stage 03] Logging key metrics for this question......")
        logger.info(f"[Result A.1] Retrieval Scores -> "
                    f"MRR: {result.get('mrr', 0):.3f}, "
                    f"nDCG@R: {result.get('ndcg_at_r', 0):.3f}, "
                    f"P@R: {result.get('precision_at_r', 0):.3f}, "
                    f"R@R: {result.get('recall_at_r', 0):.3f}")
        
        logger.info(f"[Result A.2] Generation Scores -> "
                    f"F1: {result.get('f1', 0):.3f}, "
                    f"Precision: {result.get('precision', 0):.3f}, "
                    f"Recall: {result.get('recall', 0):.3f}, "
                    f"EM: {result.get('exact_match', 0):.3f}, "
                    f"ROUGE-L: {result.get('rouge_l_f1', 0):.3f}, "
                    f"BLEU: {result.get('bleu', 0):.3f}, "
                    f"BLEU-1: {result.get('bleu_1', 0):.3f}, "
                    f"BLEU-2: {result.get('bleu_2', 0):.3f}, "
                    f"BLEU-3: {result.get('bleu_3', 0):.3f}, "
                    f"BLEU-4: {result.get('bleu_4', 0):.3f}, "
                    f"Faithfulness: {result.get('faithfulness', 0):.3f},")
        
        logger.info(f"[Result A.3] Latency: {result['latency_ms']:.1f}ms")
        logger.info(" ")
    
    logger.info("[Stage 03] Calculating aggregated metrics.....")
    successful_results = [r for r in results if r['success']]
    if successful_results:
        logger.info("[Stage 03] Aggregating metrics for successful results.....")
        metrics_to_average = [
            # retrieval metrics
            'mrr', 'ndcg_at_r', 'precision_at_r', 'recall_at_r'
            # generation metrics
            'f1', 'precision', 'recall', 'exact_match',
            'rouge_l_f1', 'rouge_l_precision', 'rouge_l_recall',
            'bleu', 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
            'faithfulness', 'latency_ms',
        ]
        
        avg_metrics = {}
        for metric in metrics_to_average:
            logger.debug("[Stage 03] Check if metric exists in successful results")
            if metric in successful_results[0]:
                avg_metrics[f'avg_{metric}'] = sum(r[metric] for r in successful_results) / len(successful_results)
        
        logger.info(" ")
        logger.info("#--#--EVALUATION SUMMARY--#--#")
        logger.info(f" [*] Total Questions: {len(eval_data)}")
        logger.info(f" [*] Successful Evaluations: {len(successful_results)}")
        
        logger.info(" ")
        logger.info("AVERAGE RETRIEVAL METRICS:")
        logger.info(f" [01] Average MRR: {avg_metrics.get('avg_mrr', 0):.3f}")
        logger.info(f" [02] Average nDCG@R: {avg_metrics.get('avg_ndcg_at_r', 0):.3f}")
        logger.info(f" [03] Average Precision@R: {avg_metrics.get('avg_precision_at_r', 0):.3f}")
        logger.info(f" [04] Average Recall@R: {avg_metrics.get('avg_recall_at_r', 0):.3f}")
        
        logger.info(" ")
        logger.info("AVERAGE GENERATION METRICS:")
        logger.info(f" [05] Average F1 Score: {avg_metrics.get('avg_f1', 0):.3f}")
        logger.info(f" [06] Average Precision: {avg_metrics.get('avg_precision', 0):.3f}")
        logger.info(f" [07] Average Recall: {avg_metrics.get('avg_recall', 0):.3f}")
        logger.info(f" [08] Average Exact Match: {avg_metrics.get('avg_exact_match', 0):.3f}")
        logger.info(f" [09] Average ROUGE-L Precision: {avg_metrics.get('avg_rouge_l_precision', 0):.3f}")
        logger.info(f" [10] Average ROUGE-L Recall: {avg_metrics.get('avg_rouge_l_recall', 0):.3f}")
        logger.info(f" [11] Average ROUGE-L F1: {avg_metrics.get('avg_rouge_l_f1', 0):.3f}")
        logger.info(f" [12] Average BLEU Score: {avg_metrics.get('avg_bleu', 0):.3f}")
        logger.info(f" [13] Average BLEU-1: {avg_metrics.get('avg_bleu_1', 0):.3f}")
        logger.info(f" [14] Average BLEU-2: {avg_metrics.get('avg_bleu_2', 0):.3f}")
        logger.info(f" [15] Average BLEU-3: {avg_metrics.get('avg_bleu_3', 0):.3f}")
        logger.info(f" [16] Average BLEU-4: {avg_metrics.get('avg_bleu_4', 0):.3f}")
        logger.info(f" [17] Average Faithfulness: {avg_metrics.get('avg_faithfulness', 0):.3f}")
        
        logger.info(" ")
        logger.info("AVERAGE LATENCY (ms/query) :")
        logger.info(f" [18] Average Latency: {avg_metrics.get('avg_latency_ms', 0):.1f}ms")
    
    logger.info("[Stage 03] Saving results to CSV.....")
    append_new_results_to_csv(results, result_dir, logger)
    logger.info(f"[Stage 03] Saved results to CSV in {result_dir}")
    
    logger.info("++++++++RAG Evaluation Completed")
    return results

if __name__ == "__main__":
    run_evaluation()