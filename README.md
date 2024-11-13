# Slack Conversation Fine-tuning for Llama 3

This project implements fine-tuning of the Llama 3.2-3B model on Slack conversation data using QLoRA (Quantized Low-Rank Adaptation) and PEFT (Parameter-Efficient Fine-Tuning) optimizations.

## Features

- Data preprocessing for Slack conversation exports
- QLoRA fine-tuning implementation
- Interactive chat interface for testing
- Support for both training and inference modes
- Memory-efficient training using 4-bit quantization


## Project Structure

```
.
├── README.md
├── main.py
├── lib/
│   ├── preprocess.py
│   ├── slack_extract.py
│   └── training.py
└── data/
    ├── users.json
    ├── channels.json
    └── [channel_directories]/
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Export your Slack workspace data
2. Place the exported data in the `data/` directory
3. Ensure you have `users.json` and `channels.json` in the root of the `data/` directory

### Training

To train the model:
```bash
python main.py --train
```

The fine-tuned model will be saved in `llama3_slack_finetuned/`.

### Testing

To start an interactive chat session with the trained model:
```bash
python main.py --test
```

To use a specific model directory:
```bash
python main.py --test --model-dir path/to/model
```

### Chat Commands

- Type your message and press Enter to get a response
- Type 'exit' to end the chat session
- Use Ctrl+C for graceful exit

## Implementation Details

### Preprocessing
- Extracts human-written messages from Slack export
- Formats conversations for model training
- Creates train/test splits

### Training
- Uses QLoRA for memory-efficient fine-tuning
- Implements PEFT for parameter-efficient training
- Uses 4-bit quantization for reduced memory usage


## License

MIT

## Acknowledgments

- Thanks to Meta AI for the Llama 3 model
