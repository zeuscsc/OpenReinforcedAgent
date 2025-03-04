from openai import OpenAI
import json
from datasets import load_from_disk, Dataset
from document_store import DocumentStore, E5EmbeddingFunction
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm
import os
from functools import partial
import argparse
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import pickle

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
                lambda x: isinstance(x, dict) and (x.get('role', None) == 'ipython' or x.get('role', None) == 'tool'), 
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
    model_answer = rollout_result[-1].get('content', None)
    ground_truth = condition.get('answer', None)
    
    if model_answer is None:
        return 0.0
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
    
    return float(max(0, similarity))

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
                lambda x: isinstance(x, dict) and (x.get('role', None) == 'ipython' or x.get('role', None) == 'tool'), 
                rollout_result
            )
        )
    )
    
    # did not emit tool calls
    if len(retrieved_ids) == 0:
        return 0.0

    return 1.0

def get_reward_functions(embedding_function):
    """Get the reward functions with initialized embedding function.

    Returns:
        Dictionary of reward functions
    """
    return {
        "mrr": get_mrr,
        "answer_similarity": partial(get_answer_similarity, embedding_function),
        "format": get_format_reward
    }

def calculate_rewards(condition, rollout_results, reward_functions, weight_scheme="uniform"):
    """Calculate rewards for a group of rollout results.
    
    Args:
        rollout_results: List of dictionaries containing metric scores
        reward_functions: Dictionary of reward functions in the form {name: function(metrics_dict)}
                         where metrics_dict contains the metrics from rollout_results
        weight_scheme: One of "uniform", "std", "pca", or dict of {name: weight}
    
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
    elif weight_scheme == "pca":
        # Convert rewards_by_function to a matrix where rows are samples and columns are features (reward functions)
        reward_names = list(rewards_by_function.keys())
        reward_matrix = np.column_stack([rewards_by_function[name] for name in reward_names])
        
        # Standardize the data (mean=0, std=1) for each feature
        reward_matrix_std = (reward_matrix - np.mean(reward_matrix, axis=0)) / (np.std(reward_matrix, axis=0) + 1e-8)
        
        # Apply PCA to project to 1 dimension (maximum variance direction)
        pca = PCA(n_components=1)
        projected_rewards = pca.fit_transform(reward_matrix_std).flatten()
        
        # Use the projected values directly as rewards
        # No need for weights since we're returning the PCA result directly
        return {
            "advantages": ((projected_rewards - np.mean(projected_rewards)) / (np.std(projected_rewards) + 1e-8)).tolist(),
            "rewards": projected_rewards.tolist(),
            "weights": {name: float(component) for name, component in zip(reward_names, pca.components_[0])},
            "rewards_by_function": rewards_by_function,
            "std": float(np.std(projected_rewards))
        }
    else:  # uniform
        weights = {name: 1.0/len(reward_functions) for name in reward_functions.keys()}
    
    # Calculate final weighted rewards
    final_rewards = np.zeros(len(rollout_results))
    for name, rewards in rewards_by_function.items():
        final_rewards += np.array(rewards) * weights[name]
    
    return {
        "advantages": ((final_rewards - np.mean(final_rewards)) / (np.std(final_rewards) + 1e-8)).tolist(),
        "rewards": final_rewards.tolist(),
        "weights": weights,
        "rewards_by_function": rewards_by_function,
        "std": float(np.std(final_rewards))
    }

def parse_rollout(grouped_rollout, rewards_info):
    """Create a dataset from rollout results and rewards.
    
    Args:
        grouped_results: List of dictionaries containing example data and rollouts
        rewards_info: List of reward information from calculate_rewards
        
    Returns:
        HuggingFace Dataset containing messages, rewards, and metrics
    """
    all_messages = []
    all_advantages = []
    all_metrics = []
    
    for group, reward_info in zip(grouped_rollout, rewards_info):
        rollouts = group["rollouts"]
        all_messages.extend(rollouts)
        all_advantages.extend(reward_info["advantages"])
        
        # Store individual metric values for each rollout
        for i in range(len(rollouts)):
            metrics = {
                name: values[i] 
                for name, values in reward_info["rewards_by_function"].items()
            }
            metrics["std"] = reward_info["std"]
            all_metrics.append(metrics)
    
    return {
        'messages': all_messages,
        'advantages': all_advantages,
        'metrics': all_metrics
    }

def message_to_dict(message):
    if isinstance(message, dict):
        return message
    else:
        if message.tool_calls:
            return {
                'role':'assistant',
                'tool_calls':[
                    {
                        'function': {
                            'arguments': json.loads(tool_call.function.arguments),
                            'name': tool_call.function.name
                        }
                    } for tool_call in message.tool_calls
                ]
            }
        else:
            return {
                'role':'assistant',
                'content':message.content
            }


def run_llm_rollout(
    dataset,
    doc_store,
    num_rollouts=64,
    model_path="/workspace/Qwen2.5-7B-Instruct-qlora-vllm",
    vllm_port=8000
):
    """Run LLM rollout on the dataset.
    
    Args:
        dataset: HuggingFace dataset containing examples
        doc_store: Document store instance for search
        num_rollouts: Maximum number of rollouts per example
        model_path: Path to the model to use for rollouts
        vllm_port: Port number for vLLM server
    Returns:
        List of dictionaries containing example data and rollout results
    """

    client = OpenAI(
        base_url=f'http://localhost:{vllm_port}/v1',
        api_key="EMPTY",
    )
    
    tools = [{
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search documents in the document store",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query to search documents"},
                },
                "required": ["query"]
            }
        }
    }]

    # Process examples
    grouped_rollout = []
    
    num_examples = len(dataset)
    for i in tqdm(range(0, num_examples)):
        example = dataset[i]
        rollout = []
        
        # Run rollouts for this example
        for _ in tqdm(range(num_rollouts)):
            messages = [
                {
                    'role':'system',
                    'content':'You are an assistant that only answers questions about the medical document store. Use the search tool once and only once to find relevant documents to the question. Fomulate your query carefully and with details.'
                },
                {
                    'role':"user",
                    'content':example['question']
                }
            ]
            
            for _ in range(5):
                if isinstance(messages[-1], ChatCompletionMessage) and messages[-1].tool_calls:
                    messages.append({
                        'role':'tool',
                        'content':json.dumps(
                            doc_store.search(
                                **json.loads(messages[-1].tool_calls[0].function.arguments),
                                n_results=3
                            )
                        )
                    })
                    continue
                if isinstance(messages[-1], dict):
                    chat_completion = client.chat.completions.create(
                        model=model_path,
                        messages=messages,
                        tools=tools,
                    )
                    response = chat_completion.choices[0].message
                    messages.append(response)
                    
            # Store rollout result
            rollout.append([message_to_dict(x) for x in messages])
        
        # Store results for this example
        grouped_rollout.append({
            "example": example,
            "rollouts": rollout
        })
    
    return grouped_rollout

def main():
    parser = argparse.ArgumentParser(description='Run LLM rollouts')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--num_rollouts', type=int, default=64, help='Number of rollouts per example')
    parser.add_argument('--model', type=str, default='/workspace/Qwen2.5-7B-Instruct-qlora', help='Path to the model')
    parser.add_argument('--vllm_port', type=int, default=8000, help='Port to use for vLLM server')
    args = parser.parse_args()

    # Load dataset
    dataset = load_from_disk(args.dataset)
    
    # Initialize document store
    doc_store = DocumentStore(
        collection_name="documents",
        chroma_db_path="./chroma_db",
        bm25_index_path="./bm25_index",
        document_content_dir="./document_content",
        device=args.device
    )
    
    # 1. Run rollouts
    grouped_rollout = run_llm_rollout(
        dataset=dataset,
        doc_store=doc_store,
        num_rollouts=args.num_rollouts,
        model_path=args.model,
        vllm_port=args.vllm_port
    )
    
    # 2. Calculate rewards
    reward_functions = get_reward_functions(embedding_function=doc_store.embedding_function)
    rewards_info = [
        calculate_rewards(
            condition=group["example"],
            rollout_results=group["rollouts"],
            reward_functions=reward_functions,
            weight_scheme="std"
        )
        for group in grouped_rollout
    ]
    
    # 3. Prase results
    rollout_result = parse_rollout(grouped_rollout, rewards_info)
    
    # 4. Save results using pickle
    output_path = os.path.join(args.dataset, "rollout_results.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(rollout_result, f)

if __name__ == "__main__":
    main()