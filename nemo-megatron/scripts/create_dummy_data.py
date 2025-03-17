#!/usr/bin/env python3

"""
Script to create dummy preference data for GRPO training
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Optional

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def create_preference_pair(
    idx: int,
    prompt_templates: Optional[List[str]] = None,
    chosen_templates: Optional[List[str]] = None,
    rejected_templates: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Create a single preference pair
    
    Args:
        idx: Index for the pair
        prompt_templates: List of prompt templates to choose from
        chosen_templates: List of chosen response templates
        rejected_templates: List of rejected response templates
        
    Returns:
        Dictionary with prompt, chosen, and rejected fields
    """
    # Default templates if none provided
    if prompt_templates is None:
        prompt_templates = [
            "Explain the concept of {topic} in simple terms.",
            "What are the advantages of using {topic}?",
            "Write a short explanation about {topic}.",
            "How does {topic} work?",
            "Summarize the key points of {topic}."
        ]
    
    if chosen_templates is None:
        chosen_templates = [
            "Here's a clear explanation of {topic}: {topic} is a technique that helps machines learn from data. It works by {explanation}. This is useful because {reason}.",
            "{topic} is a powerful approach in AI that {explanation}. The main advantages include {advantages}. Many researchers consider it essential for {application}.",
            "Let me explain {topic} simply: it's a method that {explanation}. Unlike traditional approaches, it {difference}. This makes it particularly effective for {application}.",
            "The concept of {topic} refers to {explanation}. It was developed to address {problem} and has been successfully applied to {application}. The key insight is {insight}."
        ]
    
    if rejected_templates is None:
        rejected_templates = [
            "I'm not sure about {topic}, but I think it might be related to computers or something.",
            "{topic} is basically just a fancy term for {incorrect}. It's not that important really.",
            "I don't know much about {topic}, but I can make something up if you want.",
            "The concept of {topic} is too complex to explain simply. You would need advanced degrees to understand it properly."
        ]
    
    # Topics to use in the templates
    topics = [
        "reinforcement learning", "transformer models", "attention mechanisms",
        "neural networks", "deep learning", "natural language processing",
        "computer vision", "generative AI", "transfer learning", "fine-tuning",
        "parameter-efficient fine-tuning", "low-rank adaptation", "RLHF",
        "preference optimization", "language modeling", "multimodal learning"
    ]
    
    # Explanations to use in the templates
    explanations = [
        "analyzes patterns in data to make predictions",
        "processes information in parallel to improve efficiency",
        "focuses on the most important parts of the input",
        "simulates how human neurons process information",
        "learns hierarchical representations of data",
        "understands and generates human language",
        "interprets and analyzes visual information",
        "creates new content based on patterns in training data"
    ]
    
    # Advantages to use in the templates
    advantages = [
        "improved accuracy, faster processing, and better generalization",
        "reduced computational requirements while maintaining performance",
        "ability to handle complex relationships in data",
        "better transfer of knowledge across different tasks",
        "more efficient use of training data"
    ]
    
    # Applications to use in the templates
    applications = [
        "autonomous systems", "language translation", "content generation",
        "recommendation systems", "medical diagnosis", "financial forecasting",
        "scientific research", "creative applications"
    ]
    
    # Select templates and fill in the placeholders
    import random
    
    topic = random.choice(topics)
    explanation = random.choice(explanations)
    advantage = random.choice(advantages)
    application = random.choice(applications)
    incorrect = random.choice(topics)  # Intentionally using a different topic for incorrect explanations
    
    # Ensure incorrect is different from topic
    while incorrect == topic:
        incorrect = random.choice(topics)
    
    # Create the preference pair
    prompt_template = random.choice(prompt_templates)
    chosen_template = random.choice(chosen_templates)
    rejected_template = random.choice(rejected_templates)
    
    # Fill in the templates
    prompt = prompt_template.format(topic=topic)
    chosen = chosen_template.format(
        topic=topic,
        explanation=explanation,
        reason=f"it enables {application}",
        advantages=advantage,
        application=application,
        difference=f"is more efficient at {explanation}",
        problem=f"challenges in {application}",
        insight=f"that {explanation} leads to {advantage}"
    )
    rejected = rejected_template.format(
        topic=topic,
        incorrect=incorrect
    )
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

def create_preference_dataset(
    output_path: str,
    num_samples: int = 1000,
    prompt_templates: Optional[List[str]] = None,
    chosen_templates: Optional[List[str]] = None,
    rejected_templates: Optional[List[str]] = None
):
    """
    Create a preference dataset and save it as JSONL
    
    Args:
        output_path: Path to save the JSONL file
        num_samples: Number of preference pairs to generate
        prompt_templates: List of prompt templates
        chosen_templates: List of chosen response templates
        rejected_templates: List of rejected response templates
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate preference pairs
    data = []
    for i in range(num_samples):
        pair = create_preference_pair(
            i,
            prompt_templates=prompt_templates,
            chosen_templates=chosen_templates,
            rejected_templates=rejected_templates
        )
        data.append(pair)
    
    # Save to JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    logging.info(f"Created {num_samples} preference pairs at {output_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create dummy preference data for GRPO training")
    parser.add_argument("--output-dir", type=str, default="/home/xentropy/OpenReinforcedAgent/data",
                        help="Directory to save the generated data")
    parser.add_argument("--train-samples", type=int, default=1000,
                        help="Number of training samples to generate")
    parser.add_argument("--val-samples", type=int, default=200,
                        help="Number of validation samples to generate")
    parser.add_argument("--test-samples", type=int, default=100,
                        help="Number of test samples to generate")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate datasets
    create_preference_dataset(
        os.path.join(args.output_dir, "train_preferences.jsonl"),
        num_samples=args.train_samples
    )
    
    create_preference_dataset(
        os.path.join(args.output_dir, "val_preferences.jsonl"),
        num_samples=args.val_samples
    )
    
    create_preference_dataset(
        os.path.join(args.output_dir, "test_preferences.jsonl"),
        num_samples=args.test_samples
    )
    
    logging.info(f"All datasets created successfully in {args.output_dir}")

if __name__ == "__main__":
    main()
