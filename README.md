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
- https://www.kaggle.com/code/mengqizou/llama-3b-qlora



## Example Training WOrkflow


# Dataset Format & Preprocessing

1. Data Structure
```python
{
    "messages": [
        {
            "role": "system",
            "content": "You are an AI assistant that embodies our team's Slack communication style and domain knowledge..."
        },
        {
            "role": "user", 
            "content": "<cleaned slack message>"
        },
        {
            "role": "assistant",
            "content": "<next message in thread>"
        }
        # ... alternating user/assistant roles
    ]
}
```

2. Key Preprocessing Steps:
- Group messages into coherent conversations using:
  - Thread IDs
  - Time windows (e.g. 2 hours between messages)
  - Minimum 2 messages per conversation
  - Maximum ~40 messages per conversation to fit context window
- Clean messages while preserving domain knowledge:
  - Resolve @mentions to real names
  - Keep URLs but standardize format
  - Preserve code blocks and formatting
  - Maintain emoji/reactions that provide context
- Filter out:
  - Bot messages
  - System messages
  - Empty messages
  - Messages exceeding max length after tokenization

# Fine-tuning Strategy

1. Model Loading:
- Use 4-bit quantization (QLoRA)
- Enable Flash Attention 2 where supported
- Use bfloat16 precision where supported, fallback to float16
- Enable gradient checkpointing

2. LoRA Configuration:
- Target all key model components:
  - Query/Key/Value projections
  - Output projections
  - Gate projections
  - Up/Down projections
  - LM head
- Optimal settings:
  - r=64 (higher rank for better knowledge retention)
  - alpha=128 (stronger adaptation)
  - dropout=0.05
  - Init weights with gaussian distribution

3. Training Configuration:
- Batch size: Based on GPU memory (start with 1-2)
- Gradient accumulation: 2-4 steps
- Learning rate: 2e-4 with cosine schedule
- Warmup ratio: 0.1
- Max gradient norm: 0.3-0.5
- Use paged_adamw_32bit optimizer
- Enable sequence packing for efficiency
- Train for 3-5 epochs

4. Memory Optimization:
- CPU offloading if needed
- Use FSDP for distributed training
- Implement efficient gradient checkpointing
- Clear cache between epochs

5. Post-Training:
- Option to merge LoRA weights for deployment
- Save in safe serialization format
- Shard large models appropriately

# Key Considerations

1. Quality:
- Validate conversation grouping logic extensively
- Monitor loss curves for overfitting
- Evaluate on held-out conversations
- Test domain knowledge retention

2. Performance:
- Balance between token context length and batch size
- Monitor GPU memory usage
- Consider CPU RAM requirements for merging

3. Production:
- Save checkpoints frequently
- Enable early stopping
- Implement robust error handling
- Log key metrics




Other Research:
- Finetuning your own ME model: https://github.com/andrewvassili/finetuning-fun/blob/7a07ca783183583f86eb914efadcf58a21237469/finetuning-foundation-me.ipynb#L72
- SFT Trainer Info: https://huggingface.co/docs/trl/en/sft_trainer
