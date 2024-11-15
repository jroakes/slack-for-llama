"""Main File for the application."""

import logging
import argparse
import signal
import sys
from typing import Optional, Union
from lib.preprocess import preprocess_data
from lib.training import fine_tune_model
from datasets import Dataset

def setup_signal_handlers():
    def signal_handler(sig, frame):
        print("\nGracefully exiting...")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

def train(output_dataset_path: Optional[str] = None, preprocess_only: bool = False) -> Optional[Union[str, Dataset]]:
    """Run the training workflow.

    Args:
        output_dataset_path (Optional[str]): Path to save the preprocessed dataset.
        preprocess_only (bool): If True, only preprocess the data and return the dataset.

    Returns:
        Optional[Union[str, Dataset]]: Path to the output model directory if successful, None otherwise
                           or the preprocessed dataset if output_dataset_path and preprocess_only are True.
    """
    try:
        # Updated to use Llama 3.2 3B model
        model = "meta-llama/Llama-3.2-3B"
        model_dir = "llama3_slack_finetuned"

        logging.info("Starting data preprocessing...")
        # Preprocess the data
        dataset = preprocess_data(model=model)

        if not dataset:
            logging.error("Preprocessing failed")
            return None

        if output_dataset_path:
            logging.info(f"Saving preprocessed dataset to {output_dataset_path}")
            with open(output_dataset_path, "w", encoding="utf-8") as f:
                for item in dataset["train"]:  # Save only the training split
                    f.write(item["text"] + "\n")
            if preprocess_only:
                return dataset

        if preprocess_only:
            return dataset


        logging.info("Starting model fine-tuning...")
        # Fine-tune the model
        output_dir = fine_tune_model(
            dataset=dataset,
            output_dir=model_dir,
            model_name=model
        )

        return output_dir

    except Exception as e:
        logging.error(f"Error in training execution: {e}")
        return None

def chat_loop(model_path: str):
    """Run an interactive chat loop with the fine-tuned model.

    Args:
        model_path: Path to the fine-tuned model directory
    """
    # ... (rest of the chat_loop function remains unchanged)

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Slack conversation model training and testing')
    parser.add_argument('--train', action='store_true', help='Run the training workflow')
    parser.add_argument('--test', action='store_true', help='Start an interactive chat session with the trained model')
    parser.add_argument('--model-dir', type=str, default='llama3_slack_finetuned',
                      help='Directory containing the fine-tuned model (for testing)')
    parser.add_argument('--output-dataset', type=str, help='Path to save the preprocessed dataset')
    parser.add_argument('--preprocess-only', action='store_true', help='Only preprocess the data and save it to the output dataset path.')


    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Setup signal handlers for graceful exit
    setup_signal_handlers()

    if not (args.train or args.test or args.preprocess_only):
        parser.print_help()
        sys.exit(1)

    try:
        if args.train or args.preprocess_only:
            logging.info("Starting training workflow...")
            output = train(output_dataset_path=args.output_dataset, preprocess_only=args.preprocess_only)
            if isinstance(output, str):
                logging.info(f"Training completed successfully. Model saved to: {output}")
            elif isinstance(output, Dataset):
                logging.info("Preprocessing completed successfully. Dataset saved to specified path.")
            else:
                logging.error("Training/Preprocessing failed")
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