from openai import OpenAI
import json
from datasets import load_from_disk
from document_store import DocumentStore, E5EmbeddingFunction
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm
import os
from functools import partial

# Initialize OpenAI client
client = OpenAI(
    base_url='http://localhost:8000/v1',
    api_key="EMPTY",
)

def get_mrr(condition, rollout_result):
    """Calculate Mean Reciprocal Rank for a single query.
    
    Args:
        condition (dict): Contains the ground truth information including correct_id
        rollout_result (list): List of messages containing retrieved_ids
    """
    retrieved_ids = list(
        map(
            lambda x: json.loads(x.get('content', None)).get('ids', None), 
            filter(
                lambda x: isinstance(x, dict) and x.get('role', None) == 'ipython', 
                rollout_result
            )
        )
    )
    
    if len(retrieved_ids) == 0:
        return 0.0
    
    retrieved_ids = [r.split("_chunk_")[0] for r in retrieved_ids[-1]]
    correct_id = condition.get('document_id')
        
    if correct_id in retrieved_ids:
        rank = retrieved_ids.index(correct_id) + 1
        return 1.0 / rank
    return 0.0

def get_answer_similarity(embedding_function, condition, rollout_result):
    """Calculate semantic similarity between model answer and ground truth.
    
    Args:
        condition (dict): Contains the ground truth and embedding function
        rollout_result (list): List of messages containing model's answer
    """ 
    model_answer = rollout_result[-1].content
    ground_truth = condition.get('ground_truth', '')
    
    
    # Get embeddings
    with torch.no_grad():
        embeddings = embedding_function([
            'passage: ' + model_answer,
            'passage: ' + ground_truth
        ])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]
    
    return max(0, similarity)

def get_format_reward(condition, rollout_result):
    """Calculate reward based on format compliance.
    
    Args:
        condition (dict): Contains format requirements and constraints
        rollout_result (list): List of messages to check for format compliance
    """
    retrieved_ids = list(
        map(
            lambda x: json.loads(x.get('content', None)).get('ids', None), 
            filter(
                lambda x: isinstance(x, dict) and x.get('role', None) == 'ipython', 
                rollout_result
            )
        )
    )
    
    # did not emit tool calls
    if len(retrieved_ids) == 0:
        return 0.0

    return 1.0

def calculate_rewards(condition, rollout_results, reward_functions, weight_scheme="uniform"):
    """Calculate rewards for a group of rollout results.
    
    Args:
        rollout_results: List of dictionaries containing metric scores
        reward_functions: Dictionary of reward functions in the form {name: function(metrics_dict)}
                         where metrics_dict contains the metrics from rollout_results
        weight_scheme: One of "uniform", "variance", or dict of {name: weight}
    
    Returns:
        List of reward values for each rollout
    
    Raises:
        ValueError: If reward_functions is empty or None
    """
    if not reward_functions:
        raise ValueError("reward_functions must be provided and cannot be empty")
    
    # Calculate rewards for each function
    rewards_by_function = {}
    for name, func in reward_functions.items():
        # Filter out 'chat_history' and any other non-metric fields
        rewards_by_function[name] = [
            func(condition=condition, rollout_result=result)
            for result in rollout_results
        ]
    
    # Calculate weights based on scheme
    if isinstance(weight_scheme, dict):
        weights = weight_scheme
    elif weight_scheme == "std":
        total_std = sum(np.std(rewards) for rewards in rewards_by_function.values())
        weights = {
            name: np.std(rewards) / total_std
            for name, rewards in rewards_by_function.items()
        }
    else:  # uniform
        weights = {name: 1.0/len(reward_functions) for name in reward_functions.keys()}
    
    # Calculate final weighted rewards
    final_rewards = np.zeros(len(rollout_results))
    for name, rewards in rewards_by_function.items():
        final_rewards += np.array(rewards) * weights[name]
    
    return {
        "rewards": final_rewards.tolist(),
        "weights": weights,
        "rewards_by_function": rewards_by_function,
        "std": np.std(final_rewards)
    }

def run_llm_rollout(
    dataset, 
    device, 
    max_num_rollout=64
):
    """Run LLM rollout on the dataset and evaluate performance."""
    
    # Initialize document store
    doc_store = DocumentStore(
        collection_name="documents",
        chroma_db_path="./chroma_db",
        bm25_index_path="./bm25_index",
        document_content_dir="./document_content",
        device=device
    )
    
    embedding_function = doc_store.embedding_function
    
    # Tool definition for the LLM
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search for documents using a query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        },
                    },
                    "required": ["query"],
                },
            }
        }
    ]
    
    def search_documents(query):
        """Search function that will be called by the LLM."""
        results = doc_store.search(query, n_results=10)
        return json.dumps({
            "documents": results["documents"],
            "ids": results["ids"]
        })
    
    grouped_rollout_results = []
    # Process test set
    for i in tqdm(range(0, num_examples)):
        example = dataset[i]
        rollout_results = []

        for j in range(0, max_num_rollout):
            # Metric Stroage
            # Construct the prompt
            messages = [{"role": "user", "content": "Use the functions defined above to answer the following question: " + example["question"]}]
            
            # First API call: Ask model to search for relevant documents
            response = client.chat.completions.create(
                model='/workspace/Llama-3.2-3B-Instruct',
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            
            response_message = response.choices[0].message
            messages.append(response_message)
            
            # Handle search function calls
            retrieved_ids = []
            max_retry = 3
            while messages[-1].tool_calls and max_retry > 0:
                for tool_call in messages[-1].tool_calls:
                    if tool_call.function.name == "search_documents":
                        function_args = json.loads(tool_call.function.arguments)
                        search_response = search_documents(function_args["query"])
                        
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "ipython",
                            "name": "search_documents",
                            "content": search_response,
                        })

                        # Second API call: Get final answer from model
                        next_response = client.chat.completions.create(
                            model='/workspace/Llama-3.2-3B-Instruct',
                            messages=messages,
                            tools=tools,
                        )

                        messages.append(next_response.choices[0].message)
                        max_retry -= 1

            rollout_results.append(messages)

        grouped_rollout_results.append({
            "data": example,
            "rollout_results": rollout_results
        })

    return grouped_rollout_results