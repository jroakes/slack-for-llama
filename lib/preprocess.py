"""Preprocess Slack conversations for Llama 3 fine-tuning."""
from lib.slack_extract import Slack
from datasets import Dataset
from typing import List, Dict, Optional
import re
import logging
from pathlib import Path
from lib.model_util import setup_tokenizer

def get_user_real_name(slack_instance: Slack, user_id: str) -> str:
    """Get real name for a user ID."""
    if not user_id:
        return "Unknown User"
    for user in slack_instance.users:
        if user['id'] == user_id:
            return user['profile'].get('real_name', user_id)
    return user_id

def clean_text(text: str, slack_instance: Slack) -> str:
    """Clean and normalize text while preserving context."""
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
) -> Dict:
    """Format conversation into messages list."""
    formatted_messages = []
    
    # Add system prompt if provided
    if system_prompt:
        formatted_messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Format conversation messages
    for msg in conv:
        cleaned_text = clean_text(msg.get('text', ''), slack_instance)
        if not cleaned_text:
            continue
        
        # Alternate between user/assistant roles
        role = "user" if len(formatted_messages) % 2 == (1 if system_prompt else 0) else "assistant"
        formatted_messages.append({
            "role": role,
            "content": cleaned_text
        })
    
    return {"messages": formatted_messages}

def create_dataset(
    conversation_data: List[Dict],
    tokenizer,
    max_length: int = 2048
) -> Dataset:
    """Create dataset with length validation."""
    filtered_examples = []
    skipped_count = 0
    
    # Process each conversation
    for conv_data in conversation_data:
        # Format using tokenizer's template
        try:
            text = tokenizer.apply_chat_template(
                conv_data["messages"],
                tokenize=False
            )
            
            # Validate length
            tokens = tokenizer.encode(text)
            if len(tokens) <= max_length:
                filtered_examples.append({
                    "text": text,
                    "messages": conv_data["messages"],
                    "length": len(conv_data["messages"])
                })
            else:
                skipped_count += 1
                
        except Exception as e:
            logging.warning(f"Error processing conversation: {str(e)}")
            skipped_count += 1
            continue
    
    if skipped_count > 0:
        logging.warning(f"Skipped {skipped_count} examples due to length constraints or processing errors")
    
    if not filtered_examples:
        raise ValueError("No valid examples found after filtering")
    
    # Create dataset
    dataset = Dataset.from_dict({
        "text": [ex["text"] for ex in filtered_examples],
        "messages": [ex["messages"] for ex in filtered_examples],
        "length": [ex["length"] for ex in filtered_examples]
    })
    
    # Split and log
    dataset = dataset.train_test_split(test_size=0.1)
    logging.info(f"Created dataset with {len(dataset['train'])} training and {len(dataset['test'])} test examples")
    
    return dataset

def preprocess_data(config: dict) -> Optional[Dataset]:
    """Preprocess Slack data for fine-tuning."""
    try:
        # Extract configuration values
        data_dir = config['data']['data_dir']
        model_name = config['model']['name']
        max_length = config['data'].get('max_seq_length', 2048)
        system_prompt = config['prompts'].get('training')
        
        logging.info(f"Starting preprocessing with model: {model_name}")
        logging.info(f"Using data directory: {data_dir}")
        
        # Use consolidated tokenizer setup
        tokenizer = setup_tokenizer(model_name, config)
        
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
            if len(formatted_conv["messages"]) >= 2:  # Ensure meaningful conversations
                formatted_data.append(formatted_conv)
        
        logging.info(f"Formatted {len(formatted_data)} conversations")
        
        # Create dataset
        dataset = create_dataset(formatted_data, tokenizer, max_length)
        
        if dataset:
            logging.info("Preprocessing completed successfully")
            
            # Save sample for verification
            sample_size = min(5, len(dataset['train']))
            log_path = Path(config['model'].get('output_dir', 'llama3_finetuned')) / 'dataset_sample.log'
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