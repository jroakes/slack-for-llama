"""Preprocess the data and save it to a file."""

import logging
from typing import Optional, Tuple
from transformers import AutoTokenizer
from lib.slack_extract import Slack
from datasets import Dataset

def extract_human_content(messages, slack):
    """Extract human-written content from messages."""
    extracted_content = []
    for message in messages:
        if 'text' in message and message['text'].strip():
            # Skip system messages and bot messages
            if message.get('subtype') or message.get('bot_id'):
                continue
                
            content = message['text'].strip()
            extracted_content.append(content)
    return extracted_content

def format_slack_data(data_dir: str = "data", output_file: str = "slack_conversations.txt") -> list:
    """Format Slack data into training examples."""
    logging.info("Loading Slack data...")
    slack = Slack(data_dir=data_dir)
    messages = slack.get_messages()
    
    # Extract human-written content
    training_texts = extract_human_content(messages, slack)
    
    logging.info(f"Extracted {len(training_texts)} messages")
    
    # Filter out very short messages
    training_texts = [text for text in training_texts if len(text) > 10]
    
    # Save raw texts for reference
    with open(output_file, "w", encoding='utf-8') as f:
        for text in training_texts:
            f.write(f"{text}\n")
            f.write("-" * 50 + "\n")
    
    logging.info(f"Saved {len(training_texts)} training examples to {output_file}")
    return training_texts

def tokenize_data(texts: list, tokenizer: AutoTokenizer, max_length: int = 512) -> Dataset:
    """Convert text data into tokenized dataset."""
    logging.info("Tokenizing data...")
    
    formatted_data = []
    
    for idx, text in enumerate(texts):
        if idx % 1000 == 0:
            logging.info(f"Tokenizing example {idx}/{len(texts)}")
            
        # Simple text format
        formatted_data.append({"text": text})
    
    # Convert to Dataset format
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split into train/test
    dataset_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    
    logging.info(f"Created dataset with {len(dataset_split['train'])} training and {len(dataset_split['test'])} test examples")
    
    return dataset_split

def preprocess_data(data_dir: str = "data", model: str = "meta-llama/Llama-3.2-3B") -> Tuple[Optional[Dataset], Optional[AutoTokenizer]]:
    """Preprocess the data for fine-tuning."""
    try:
        # Initialize tokenizer
        logging.info(f"Initializing tokenizer for model: {model}")
        tokenizer = AutoTokenizer.from_pretrained(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logging.info("Tokenizer initialized successfully")
        
        # Get the text data
        logging.info("Formatting Slack data...")
        texts = format_slack_data(data_dir, "slack_conversations.txt")
        
        if not texts:
            logging.error("No valid texts found in dataset")
            return None, None
        
        # Log sample data
        logging.info("\nSample texts:")
        for i in range(min(3, len(texts))):
            logging.info(f"\nExample {i+1}:\n{texts[i][:200]}...")
        
        # Convert to HuggingFace Dataset format
        logging.info("\nCreating dataset...")
        tokenized_dataset = tokenize_data(texts, tokenizer)
        logging.info("Dataset creation complete")
        
        return tokenized_dataset, tokenizer
        
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None, None