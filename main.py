# main.py

import logging
import argparse
import signal
import sys
from typing import Optional
from lib.preprocess import preprocess_data
from lib.training import fine_tune_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from trl import setup_chat_format  

def setup_signal_handlers():
    def signal_handler(sig, frame):
        print("\nGracefully exiting...")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

def train() -> Optional[str]:
    """Run the training workflow."""
    try:
        # Updated to use Llama 3.2 3B model with proper configuration
        model = "meta-llama/Llama-3.2-3B"
        model_dir = "llama3_slack_finetuned"

        logging.info("Starting data preprocessing...")
        # Preprocess the data with more detailed logging
        dataset, tokenizer = preprocess_data(model=model)

        if not dataset:
            logging.error("Preprocessing failed - no valid dataset created")
            return None

        logging.info(f"Dataset created successfully - {len(dataset['train'])} training examples")
        
        logging.info("Starting model fine-tuning...")
        # Updated training configuration
        output_dir = fine_tune_model(
            dataset=dataset,
            tokenizer=tokenizer,
            output_dir=model_dir,
            model_name=model
        )

        return output_dir

    except Exception as e:
        logging.error(f"Error in training execution: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None


def chat_loop(model_path: str):
    """Run an interactive chat loop with the fine-tuned model."""
    try:
        logging.info("Loading model and tokenizer...")
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_path)  

        # Load model with the tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            tokenizer=tokenizer  # Pass the tokenizer here
        )

        # No setup_chat_format needed since we saved these configs during training
        
        logging.info("Starting chat session... (Type 'exit' to end the session)")
        print("\nChat session started. You can start chatting with the model.")
        print("Type 'exit' to end the session.")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                print("\nEnding chat session...")
                break
            
            if not user_input:
                continue
            
            messages = [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = tokenizer(
                prompt, 
                return_tensors='pt', 
                padding=True, 
                truncation=True
            ).to("cuda")
            
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("\nAssistant:", response)

    except Exception as e:
        logging.error(f"Error in chat session: {e}")
        print("\nAn error occurred. Ending chat session...")
        print(f"Error details: {str(e)}")
    
    print("\nChat session ended.")

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Slack conversation model training and testing')
    parser.add_argument('--train', action='store_true', help='Run the training workflow')
    parser.add_argument('--test', action='store_true', help='Start an interactive chat session with the trained model')
    parser.add_argument('--model-dir', type=str, default='llama3_slack_finetuned',
                      help='Directory containing the fine-tuned model (for testing)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Setup signal handlers for graceful exit
    setup_signal_handlers()

    if not (args.train or args.test):
        parser.print_help()
        sys.exit(1)

    try:
        if args.train:
            logging.info("Starting training workflow...")
            output_dir = train()

            if output_dir:
                logging.info(f"Training completed successfully. Model saved to: {output_dir}")
            else:
                logging.error("Training failed")
                sys.exit(1)

        if args.test:
            logging.info(f"Starting chat session with model from: {args.model_dir}")
            chat_loop(args.model_dir)

    except KeyboardInterrupt:
        print("\nGracefully exiting...")
        sys.exit(0)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()