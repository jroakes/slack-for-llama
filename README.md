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

```
# Data Imports & Processing

import csv
from datasets import Dataset

dataset = []

with open('/kaggle/input/financial-news/nasdaq_news_educational_values.csv', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        features = row["Text"]
        dataset.append(features)
        if len(dataset) == 1000:  # filtering out the top 1000 data with educaitonal values
            break

# Convert the list to a Hugging Face Dataset
dataset = Dataset.from_dict({"text": dataset})

# Split the dataset into training and testing sets
dataset = dataset.train_test_split(test_size=0.1)

dataset

# # Model Selection

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

from huggingface_hub import login

#Get ENV variables
hf_token = os.getenv("HUGGINGFACE_TOKEN")
wb_token = os.getenv("WANDB_TOKEN")

login(token = hf_token)
wandb.login(key=wb_token)

run = wandb.init(
    project='Fine-tune Llama 3.2-3B on nasdaq news Dataset', 
    job_type="training", 
    anonymous="allow"
)

base_model = "meta-llama/Llama-3.2-3B"
new_model = "llama-3.2-3B-nasdaq-news"

# # Training Techniques

torch_dtype = torch.float16
attn_implementation = "eager"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
model, tokenizer = setup_chat_format(model, tokenizer)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)

training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length=512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

trainer.train()

wandb.finish()
model.config.use_cache = True

messages = [
    {
        "role": "user",
        "content": "Hi, can you extract the financial data from the content here: Capital Account Statement for Genesis Growth Partners IV As of the period ending on September 30, 2024, the market value of the fund was reported at $1,250,000. The total commitment agreed upon is $2,000,000, with an unfunded commitment remaining at $750,000. The amount of capital called up to date amounts to $1,250,000. This statement was issued on October 10, 2024."
    },
    {
        "role": "assistant",
        "content": '''Extract the following financial data from the provided statement:
                Period End Date,
                Market Value,
                Commitment,
                Unfunded Commitment,
                Capital Called,
                Fund Name,
                Document Issue Date'''
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, 
                                       add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors='pt', padding=True, 
                   truncation=True).to("cuda")

outputs = model.generate(**inputs, max_length=300, 
                         num_return_sequences=1)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(text)


trainer.model.save_pretrained(new_model)
trainer.model.push_to_hub(new_model, use_temp_dir=False)
```


Other Research:
- Finetuning your own ME model: https://github.com/andrewvassili/finetuning-fun/blob/7a07ca783183583f86eb914efadcf58a21237469/finetuning-foundation-me.ipynb#L72
- SFT Trainer Info: https://huggingface.co/docs/trl/en/sft_trainer
