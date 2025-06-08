import os
import json
import traceback
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from utils.config import CONFIG
from utils.logger import setup_logger
from rag_pipeline.stage_02_query_data import query_rag
from utils.evaluation.get_retrieval_eval_metrics import calc_all_retrieval_scores
from utils.evaluation.get_generation_eval_metrics import calc_all_generation_scores
from utils.evaluation.save_n_update_results import append_new_results_to_csv


# configurations
LOG_PATH = CONFIG["LOG_PATH"]
CHROMA_DB_PATH = CONFIG["CHROMA_DB_PATH"]
EVALUATION_DATA_PATHS = CONFIG.get("EVALUATION_DATA_PATHS", [])
RESULTS_CSV_PATH = CONFIG.get("RESULTS_CSV_PATH", "evaluation_results.csv")
K = CONFIG["K"]
R = CONFIG["R"]
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
        logger.info(f"[Part 01] Loaded evaluation data from {file_path}")
    except FileNotFoundError:
        logger.warning(f"[Part 01] Evaluation data file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"[Part 01] Error parsing JSON file: {e}")
        logger.debug(traceback.format_exc())
        return []

def evaluate_single_query(question: str, ground_truth: str, relevant_doc_ids: List[str], at_k, logger) -> Dict[str, Any]:
    start_time = time.time()

    try:
        logger.info(f"[Part 02] Processing question: {question}")
        rag_result = query_rag(query_text=question, chroma_db_dir=CHROMA_DB_PATH, at_k=K, at_r=R, logger=logger)
        
        predicted_answer = rag_result.get('llms_response', '')
        context = rag_result.get('context', '')
        retrieved_docs = rag_result.get('results', []) 
        graded_results = rag_result.get('graded_results', []) # list of (doc, sim_score) 
        
        logger.info(f" [Part 02.1(a)] Predicted answer: {predicted_answer}")
        logger.info(f" [Part 02.1(b)] Context: {context}")
        logger.info(f" [Part 02.1(c)] Retrieved docs: {retrieved_docs}")
        logger.info(f" [Part 02.1(d)] Graded results: {graded_results}")
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        
        logger.info("[Part 02.2] Extracting retrieved document IDs in order......")
        retrieved_doc_ids = [doc.metadata.get("chunk_id", "") for doc, _ in retrieved_docs]
        logger.info(f"[Part 02.2] Retrieved doc IDs: {retrieved_doc_ids}")
        
        logger.info(f"[Part 02.3] Calculating retrieval & generation scores for query: {question}")
        retrieval_scores = calc_all_retrieval_scores(retrieved_doc_ids, relevant_doc_ids, k=at_k)
        generation_scores = calc_all_generation_scores(predicted_answer, ground_truth, context)
        
        result = {
            'question': question,
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'context': context[:200] + "..." if len(context) > 200 else context,  # Truncate for CSV
            'retrieved_doc_ids': retrieved_doc_ids[:5],  # Top 5 for CSV
            'relevant_doc_ids': relevant_doc_ids,
            'latency_ms': latency_ms,
            'success': True,
            'error': None
        }
        
        logger.info(f"[Part 02.3] Adding retrieval & generation scores to result......")
        result.update(retrieval_scores)
        result.update(generation_scores)
        
        logger.info(f"[Part 02.3] Successfully calculated retrieval & generation scores for query: {question}")
        return result
        
    except Exception as e:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        logger.error(f"[Part 02] Error: {e}")
        logger.error(f"Latency: {latency_ms} ms")
        logger.debug(traceback.format_exc())
        
        result = {
            'question': question,
            'predicted_answer': '',
            'ground_truth': ground_truth,
            'retrieved_doc_ids': [],
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
        logger.info(f"[Part 02] Finished processing question & generating eval. scores")
        # Log the result
        logger.info(json.dumps(result))

# def save_results_to_csv(results: List[Dict], result_dir, logger):
#     try:
#         df = pd.DataFrame(results)
#         logger.info(f"[Part 03] Saving results to CSV: {result_dir}")
        
#         logger.info("[Part 03.1] Creating directory if not exists.....")
#         os.makedirs(os.path.dirname(result_dir), exist_ok=True)
        
#         logger.info("[Part 03.2] Checking if CSV exists to append or create new.....")
#         if os.path.exists(result_dir):
#             logger.info("[Part 03.3] CSV exists, appending results.....")
#             existing_df = pd.read_csv(result_dir)
#             combined_df = pd.concat([existing_df, df], ignore_index=True)
#             combined_df.to_csv(result_dir, index=False)
#             logger.info(f"[Part 03(a)] Appended {len(results)} results to existing CSV: {result_dir}")
#         else:
#             logger.info("[Part 03.4] CSV does not exist, creating new CSV.....")
#             df.to_csv(result_dir, index=False)
#             logger.info(f"[Part 03(b)] Created new CSV with {len(results)} results: {result_dir}")
            
#     except Exception as e:
#         logger.error(f"[Part 03] Error saving results to CSV: {e}")
#         logger.debug(traceback.format_exc())


def run_evaluation(eval_dir=EVALUATION_DATA_PATHS, result_dir=RESULTS_CSV_PATH, at_k=K,):
    logger = setup_logger("evaluation_logger", LOG_FILE)
    logger.info(" ")
    logger.info("++++++++[Pipeline 3] Starting RAG Evaluation.....")
    
    logger.info(f"Loading evaluation data from {eval_dir}......")
    eval_data = load_evaluation_data(eval_dir, logger)
    if not eval_data:
        logger.error("No evaluation data found. Exiting.")
        return
    
    logger.info(f"Loaded {len(eval_data)} evaluation questions")
    
    # logger.info(f"Loading & combining evaluation data from {eval_dir}......")
    # eval_data = []
    # for path in eval_dir:
    #     logger.info(f"Loading evaluation data from {path}......")
    #     data = load_evaluation_data(path, logger)
    #     if not data:
    #         logger.warning(f"No valid data found in {path}. Skipping.....")
    #         continue
    #     for item in data:
    #         item['source_file'] = os.path.basename(path)
    #         logger.debug("Tracking source file for each question......")
    #     eval_data.extend(data)

    # if not eval_data:
    #     logger.error("No evaluation data found from any source. Exiting.")
    #     return

    logger.info(f"Total combined evaluation questions: {len(eval_data)}")
    logger.info("Evaluating each question......")
    results = []
    for i, item in enumerate(eval_data):
        logger.info(f"Evaluating question {i+1}/{len(eval_data)}: {item['question'][:50]}......")
        
        logger.info("Get relevant_doc_ids from the evaluation data......")
        relevant_doc_ids = item.get('relevant_doc_ids', [])
        if not relevant_doc_ids:
            logger.warning(f"No relevant_doc_ids found for question {i+1}. Using empty list.")
        
        result = evaluate_single_query(
            question=item['question'],
            ground_truth=item['ground_truth'],
            relevant_doc_ids=relevant_doc_ids,
            at_k=at_k,
            logger=logger
        )
        
        logger.info("Adding metadata to the results......")
        result.update({
            'timestamp': datetime.now().isoformat(),
            'embedding_model_name': EMBEDDING_MODEL,
            'gen_model_name': LOCAL_LLM,
            'question_id': i,
            'category': item.get('category', 'unknown'),
            'difficulty': item.get('difficulty', 'unknown'),
            'source_file': item.get('source_file', 'unknown'),
        })
        results.append(result)
        
        logger.info(" ")
        logger.info("Logging key metrics for this question......")
        logger.info(f"[Result A.1] Retrieval Scores -> "
                    f"MRR: {result.get('mrr', 0):.3f}, "
                    f"nDCG@5: {result.get('ndcg_at_5', 0):.3f}, "
                    f"P@5: {result.get('precision_at_5', 0):.3f}, "
                    f"R@5: {result.get('recall_at_5', 0):.3f}")
        
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
    
    logger.info("Calculating aggregated metrics.....")
    successful_results = [r for r in results if r['success']]
    if successful_results:
        logger.info("Aggregating metrics for successful results.....")
        metrics_to_average = [
            # retrieval metrics
            'mrr', 'ndcg_at_5', 'precision_at_5', 'recall_at_5'
            # generation metrics
            'f1', 'precision', 'recall', 'exact_match',
            'rouge_l_f1', 'rouge_l_precision', 'rouge_l_recall',
            'bleu', 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
            'faithfulness', 'latency_ms',
        ]
        
        avg_metrics = {}
        for metric in metrics_to_average:
            logger.debug("Check if metric exists in successful results")
            if metric in successful_results[0]:
                avg_metrics[f'avg_{metric}'] = sum(r[metric] for r in successful_results) / len(successful_results)
        
        logger.info(" ")
        logger.info("#--#--EVALUATION SUMMARY--#--#")
        logger.info(f" [*] Total Questions: {len(eval_data)}")
        logger.info(f" [*] Successful Evaluations: {len(successful_results)}")
        
        logger.info(" ")
        logger.info("AVERAGE RETRIEVAL METRICS:")
        logger.info(f" [01] Average MRR: {avg_metrics.get('avg_mrr', 0):.3f}")
        logger.info(f" [02] Average nDCG@5: {avg_metrics.get('avg_ndcg_at_5', 0):.3f}")
        logger.info(f" [03] Average Precision@5: {avg_metrics.get('avg_precision_at_5', 0):.3f}")
        logger.info(f" [04] Average Recall@5: {avg_metrics.get('avg_recall_at_5', 0):.3f}")
        
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
    
    logger.info("Saving results to CSV.....")
    # save_results_to_csv(results, result_dir, logger)
    append_new_results_to_csv(results, result_dir, logger)
    logger.info(f"Saved results to CSV in {result_dir}")
    
    logger.info("++++++++RAG Evaluation Completed")
    return results

if __name__ == "__main__":
    run_evaluation()