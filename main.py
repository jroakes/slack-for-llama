"""Main entry point for Slack conversation model training and testing."""
import logging
import argparse
import sys
import os
from typing import Optional
import yaml
from yaml import SafeLoader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from lib.preprocess import preprocess_data
from lib.model_util import load_fine_tuned_model, get_model_info
from lib.training import fine_tune_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)


# Custom stopping criteria for chat
class ChatStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stops=None):
        super().__init__()
        self.stops = stops or ["Human:", "human:", "Assistant:", "assistant:"]
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids is None or input_ids.shape[1] == 0:
            return False

        # Check if input_ids is too short to decode
        if input_ids.shape[0] < 3:
            return False
        
        # Decode the last part of the input_ids
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Check for any of the stop strings at the end of the decoded sequence
        for stop in self.stops:
            if decoded.endswith(stop):
                return True
        return False



def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=SafeLoader)
        
        # Required base sections
        required_sections = ['model', 'data', 'training', 'prompts']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate critical subsections
        if not config['model'].get('name'):
            raise ValueError("Model name not specified in config")
        if not config['data'].get('data_dir'):
            raise ValueError("Data directory not specified in config")
            
        # Set defaults for optional sections
        if 'chat' not in config:
            config['chat'] = {
                'max_new_tokens': 512,
                'temperature': 0.7,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'no_repeat_ngram_size': 3
            }
        
        return config
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        raise

def train(config: dict) -> Optional[str]:
    """Run the training workflow."""
    try:
        # Get model configuration
        model_name = config['model']['name']
        model_dir = config['model'].get('output_dir', "llama3_finetuned")
        
        # Log key configuration settings
        logging.info(f"Starting training pipeline:")
        logging.info(f"Model: {model_name}")
        logging.info(f"Output directory: {model_dir}")
        logging.info(f"Batch size: {config['training'].get('batch_size', 4)}")
        logging.info(f"Training epochs: {config['training'].get('num_train_epochs', 3)}")
        logging.info(f"Max sequence length: {config['data'].get('max_seq_length', 2048)}")
        if config['model'].get('quantization', {}).get('enabled', True):
            logging.info("Quantization: Enabled")
            logging.info(f"Quantization type: {config['model'].get('quantization', {}).get('load_in_4bit', True) and '4-bit' or '8-bit'}")
        else:
            logging.info("Quantization: Disabled")
        
        # Preprocess data
        dataset = preprocess_data(config)
        if not dataset:
            logging.error("Preprocessing failed")
            return None
        
        # Add dataset info to config for model card
        config['dataset_info'] = {
            'train_samples': len(dataset['train']),
            'eval_samples': len(dataset['test'])
        }
        
        # Run training
        logging.info("Starting model fine-tuning...")
        output_dir = fine_tune_model(
            dataset=dataset,
            output_dir=model_dir,
            config=config
        )
        
        if output_dir:
            # Get and log model info after training
            model_info = get_model_info(output_dir)
            if model_info:
                logging.info("Training completed successfully:")
                logging.info(f"Model size: {model_info.get('model_size_gb', 0):.2f} GB")
                logging.info(f"Number of shards: {model_info.get('num_shards', 0)}")
            
        return output_dir
        
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        return None


def chat_loop(model_path: str, config: dict):
    """Interactive chat loop with fine-tuned model."""
    try:
        # Load model with inference settings
        model, tokenizer = load_fine_tuned_model(
            model_path,
            inference_mode=True
        )

        if not model or not tokenizer:
            logging.error("Failed to load model or tokenizer")
            return

        # Get chat configuration
        chat_config = config.get('chat', {})
        system_prompt = config.get('prompts', {}).get('chat', '')

        print("\nChat session started. Commands:")
        print("- 'exit': End the session")
        print("- 'new': Start a new conversation")
        print("- 'clear_history': Clear conversation history")
        print("- 'info': Show model information")
        
        conversation_history = [{
            "role": "system", 
            "content": system_prompt
        }]
        
        # Initialize stopping criteria with tokenizer
        stopping_criteria = StoppingCriteriaList([ChatStoppingCriteria(tokenizer)])
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'new':
                conversation_history = [conversation_history[0]]
                print("\nStarting new conversation...")
                continue
            elif user_input.lower() == 'clear_history':
                conversation_history = [conversation_history[0]]
                print("\nConversation history cleared...")
                continue
            elif user_input.lower() == 'info':
                model_info = get_model_info(model_path)
                if model_info:
                    print("\nModel Information:")
                    print(f"Base model: {model_info.get('model_card', {}).get('base_model', 'unknown')}")
                    print(f"Size: {model_info.get('model_size_gb', 0):.2f} GB")
                    print(f"Training samples: {model_info.get('model_card', {}).get('dataset_info', {}).get('train_samples', 0)}")
                continue
            
            # Add user input to conversation
            conversation_history.append({"role": "user", "content": user_input})
            
            # Generate response
            inputs = tokenizer.apply_chat_template(
                conversation_history,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            attention_mask = torch.ones_like(inputs)
            attention_mask[inputs == tokenizer.pad_token_id] = 0
            
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=chat_config.get('max_new_tokens', 512),
                temperature=chat_config.get('temperature', 0.7),
                top_p=chat_config.get('top_p', 0.9),
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=chat_config.get('repetition_penalty', 1.1),
                no_repeat_ngram_size=chat_config.get('no_repeat_ngram_size', 3),
                stopping_criteria=stopping_criteria
            )
            
            response = tokenizer.decode(
                outputs[0][inputs.shape[-1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()
            
            # Clean up response
            print(f"\n{response}")
            conversation_history.append({"role": "assistant", "content": response})
            
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        print("\nError occurred. Chat session ended.")



def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Slack conversation model training and chat')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--chat', action='store_true', help='Start chat session')
    parser.add_argument('--config', type=str, default='llmconfig.yaml', help='Path to config file')
    parser.add_argument('--model-dir', type=str, help='Model directory for chat (overrides config)')
    
    args = parser.parse_args()
    
    if not (args.train or args.chat):
        parser.print_help()
        sys.exit(1)
    
    try:
        # Load config
        config = load_config(args.config)
        
        if args.train:
            output = train(config)
            if output:
                print(f"Training completed. Model saved to: {output}")
            else:
                print("Training failed")
                sys.exit(1)
        
        if args.chat:
            model_dir = args.model_dir or config['model']['output_dir']
            chat_loop(model_dir, config)
            
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()