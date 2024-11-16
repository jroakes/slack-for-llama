"""Preprocess Slack conversations for Llama 3 fine-tuning."""
from lib.slack_extract import Slack
from datasets import Dataset
from typing import List, Dict, Optional
import re
from transformers import AutoTokenizer
import logging
from pathlib import Path

def get_user_real_name(slack_instance: Slack, user_id: str) -> str:
    """Get real name for a user ID."""
    if not user_id:
        return "Unknown User"
    for user in slack_instance.users:
        if user['id'] == user_id:
            return user['profile'].get('real_name', user_id)
    return user_id

def clean_text(text: str, slack_instance: Slack) -> str:
    """Minimal cleaning to preserve conversation context."""
    if not isinstance(text, str):
        return ""
    
    # Normalize Slack-specific formatting
    text = re.sub(r'<@([A-Za-z0-9]+)>', 
                  lambda m: f"@{get_user_real_name(slack_instance, m.group(1))}", 
                  text)
    text = re.sub(r'<#([A-Za-z0-9]+)\|([^>]+)>', r'#\2', text)
    
    # Keep URLs but standardize format
    text = re.sub(r'<(http\S+)>', r'\1', text)
    
    # Basic cleaning
    text = ' '.join(text.split())
    return text

def group_messages_into_conversations(
    messages: List[Dict],
    time_window: float = 7200,  # 2 hours
    min_messages: int = 2,
    max_messages: int = 40
) -> List[List[Dict]]:
    """Group messages into conversations with expanded context."""
    if not messages:
        return []
    
    conversations = []
    current_conv = []
    
    for msg in messages:
        if not current_conv:
            current_conv.append(msg)
            continue
        
        prev_msg = current_conv[-1]
        
        try:
            time_diff = float(msg['timestamp']) - float(prev_msg['timestamp'])
        except (ValueError, KeyError):
            continue
        
        # Check thread continuity and time window
        same_thread = (
            ('thread_ts' in msg and 'thread_ts' in prev_msg and 
             msg['thread_ts'] == prev_msg['thread_ts']) or
            ('thread_ts' not in msg and 'thread_ts' not in prev_msg)
        )
        
        if same_thread and time_diff < time_window and len(current_conv) < max_messages:
            current_conv.append(msg)
        else:
            if len(current_conv) >= min_messages:
                conversations.append(current_conv)
            current_conv = [msg]
    
    if len(current_conv) >= min_messages:
        conversations.append(current_conv)
    
    return conversations

def format_conversation_for_llama(
    conv: List[Dict], 
    slack_instance: Slack, 
    system_prompt: Optional[str] = None
) -> List[Dict]:
    """Format conversation for Llama chat format."""
    formatted_messages = []
    
    if system_prompt:
        formatted_messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    for msg in conv:
        cleaned_text = clean_text(msg.get('text', ''), slack_instance)
        if not cleaned_text:
            continue
        
        # Alternate between user/assistant roles
        user_msg = {
            "role": "user" if len(formatted_messages) % 2 == (1 if system_prompt else 0) else "assistant",
            "content": cleaned_text
        }
        formatted_messages.append(user_msg)
    
    return formatted_messages

def prepare_conversation_pairs(formatted_data: List[Dict]) -> List[Dict]:
    """Create training examples from conversations."""
    training_examples = []
    
    for conv_data in formatted_data:
        messages = conv_data["messages"]
        if len(messages) < 2:
            continue
        
        # Process each dialogue turn
        for i in range(1, len(messages)-1, 2):
            context_messages = messages[:i+1]
            response = messages[i+1]["content"]
            
            if not response.strip():
                continue
            
            example = {
                "text": str(context_messages) + "\n\nResponse: " + response,
                "context_length": len(context_messages)
            }
            training_examples.append(example)
    
    return training_examples

def create_dataset(
    conversation_data: List[Dict],
    tokenizer: AutoTokenizer,
    max_length: int = 2048
) -> Dataset:
    """Create dataset with length validation."""
    training_examples = prepare_conversation_pairs(conversation_data)
    
    filtered_examples = []
    skipped_count = 0
    
    for example in training_examples:
        tokens = tokenizer.encode(example["text"])
        if len(tokens) <= max_length:
            filtered_examples.append(example)
        else:
            skipped_count += 1
    
    if skipped_count > 0:
        logging.warning(f"Skipped {skipped_count} examples due to length constraints")
    
    dataset = Dataset.from_dict({
        "text": [ex["text"] for ex in filtered_examples],
        "context_length": [ex["context_length"] for ex in filtered_examples]
    })
    
    # Training/test split with logging
    dataset = dataset.train_test_split(test_size=0.1)
    logging.info(f"Created dataset with {len(dataset['train'])} training and {len(dataset['test'])} test examples")
    
    return dataset

def preprocess_data(config: dict) -> Optional[Dataset]:
    """Preprocess Slack data for fine-tuning using configuration."""
    try:
        # Extract configuration values
        data_dir = config['data']['data_dir']
        model_name = config['model']['name']
        max_length = config['data'].get('max_seq_length', 2048)
        system_prompt = config['prompts'].get('training')
        
        logging.info(f"Starting preprocessing with model: {model_name}")
        logging.info(f"Using data directory: {data_dir}")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logging.info("Set pad_token to eos_token for tokenizer")
        
        # Verify data directory exists
        data_path = Path(data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        # Initialize Slack data processor
        slack = Slack(data_dir=str(data_path))
        messages = slack.get_messages()
        
        if not messages:
            logging.error("No messages found in data directory")
            return None
            
        logging.info(f"Found {len(messages)} messages")
        
        # Format messages for processing
        human_content = [
            {
                'text': msg['text'],
                'user': get_user_real_name(slack, msg.get('user', '')),
                'timestamp': msg.get('ts', ''),
                'thread_ts': msg.get('thread_ts', None)
            }
            for msg in messages
            if isinstance(msg.get('text', ''), str) and msg.get('text', '').strip()
        ]
        
        # Sort by timestamp
        human_content.sort(key=lambda x: float(x['timestamp']) if x['timestamp'] else 0)
        
        # Group messages into conversations
        conversations = group_messages_into_conversations(
            human_content,
            time_window=config['data'].get('conversation_window', 7200),
            min_messages=config['data'].get('min_messages', 2),
            max_messages=config['data'].get('max_messages', 40)
        )
        
        logging.info(f"Grouped messages into {len(conversations)} conversations")
        
        # Format conversations for training
        formatted_data = []
        for conv in conversations:
            formatted_conv = format_conversation_for_llama(conv, slack, system_prompt)
            if len(formatted_conv) >= 2:  # Ensure meaningful conversations
                formatted_data.append({"messages": formatted_conv})
        
        # Create dataset
        dataset = create_dataset(formatted_data, tokenizer, max_length)
        
        # Log preprocessing completion
        if dataset:
            logging.info("Preprocessing completed successfully")
        
        # Save sample to log file for verification
        sample_size = min(5, len(dataset['train']))
        log_path = Path(config['model'].get('output_dir', 'logs')) / 'dataset_sample.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("Dataset Sample:\n\n")
            for i in range(sample_size):
                f.write(f"Example {i+1}:\n")
                f.write(f"{dataset['train'][i]['text']}\n")
                f.write("-" * 80 + "\n")
        
        logging.info(f"Saved dataset sample to {log_path}")
        
        return dataset
        
    except Exception as e:
        logging.error(f"Preprocessing error: {str(e)}")
        return None