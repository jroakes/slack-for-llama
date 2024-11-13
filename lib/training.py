# training.py

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer, setup_chat_format, SFTConfig
import torch
from typing import Optional
import os
from datasets import Dataset

def fine_tune_model(
    dataset: Dataset, 
    tokenizer: AutoTokenizer,  # Add tokenizer as an argument
    output_dir: str, 
    model_name: str = "meta-llama/Llama-3.2-3B",
) -> Optional[str]:
    """
    Fine-tune a Llama 3.2 model using QLoRA and PEFT optimizations
    """
    try:
        torch_dtype = torch.float16
        attn_implementation = "eager"

        # QLoRA configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

        # Load base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        
        # Use the provided tokenizer
        model, _ = setup_chat_format(model, tokenizer)  # Use the passed tokenizer
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id

        # LoRA configuration
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                'up_proj', 'down_proj', 'gate_proj', 
                'k_proj', 'q_proj', 'v_proj', 'o_proj'
            ]
        )
        
        # Apply LoRA
        model = get_peft_model(model, peft_config)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=5,
            per_device_eval_batch_size=5,
            gradient_accumulation_steps=2,
            optim="paged_adamw_32bit",
            num_train_epochs=1,
            eval_strategy="steps",  # Updated from evaluation_strategy
            eval_steps=0.2,
            logging_steps=1,
            warmup_steps=10,
            learning_rate=2e-4,
            fp16=False,
            bf16=False,
            group_by_length=True,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            logging_dir=os.path.join(output_dir, "logs")
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            peft_config=peft_config,
            max_seq_length=512,
            dataset_text_field="text",
            tokenizer=tokenizer,
            args=training_args,
            packing=False
        )

        # Start training
        trainer.train()
        
        # Save the model
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)  # Save the tokenizer
        print(f"Model fine-tuned and saved to {output_dir}")
        
        return output_dir
        
    except Exception as e:
        print(f"Error during training: {e}")
        return None