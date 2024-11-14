# training.py

import os
import torch
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from trl import SFTTrainer
from datasets import Dataset
import logging

def fine_tune_model(
    dataset: Dataset, 
    tokenizer: AutoTokenizer,
    output_dir: str, 
    model_name: str = "meta-llama/Llama-3.2-3B",
) -> Optional[str]:
    """
    Fine-tune a model using QLoRA and PEFT optimizations.
    
    Args:
        dataset: HuggingFace dataset with 'train' and 'test' splits
        tokenizer: Pretrained tokenizer
        output_dir: Directory to save the model
        model_name: Name or path of the base model
        
    Returns:
        Optional[str]: Path to saved model or None if training fails
    """
    try:
        logging.info("Initializing model configuration...")
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        logging.info(f"Loading base model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            use_cache=False,  # Disable KV cache for training
            torch_dtype=torch.float16
        )
        
        # Enable gradient checkpointing for memory efficiency
        logging.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        
        # Configure tokenizer padding
        if tokenizer.pad_token is None:
            logging.info("Setting pad token...")
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id

        logging.info("Configuring LoRA...")
        # LoRA configuration
        peft_config = LoraConfig(
            r=16,               # Rank
            lora_alpha=32,      # Alpha parameter for LoRA scaling
            lora_dropout=0.05,  # Dropout probability for LoRA layers
            bias="none",        # Bias type
            task_type="CAUSAL_LM",
            target_modules=[
                'up_proj',
                'down_proj',
                'gate_proj',
                'k_proj',
                'q_proj',
                'v_proj',
                'o_proj'
            ]
        )
        
        # Apply LoRA to model
        logging.info("Applying LoRA to model...")
        model = get_peft_model(model, peft_config)
        
        logging.info("Setting up training arguments...")
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            learning_rate=2e-4,
            fp16=True,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            optim="paged_adamw_32bit",
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            group_by_length=True,
            save_safetensors=True,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False,
            report_to="none"  # Disable wandb logging
        )

        logging.info("Initializing trainer...")
        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=training_args,
            tokenizer=tokenizer,
            max_seq_length=512,
            dataset_text_field="text",
            packing=False
        )

        logging.info("Starting training...")
        trainer.train()
        
        logging.info("Saving model...")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logging.info(f"Training completed successfully. Model saved to: {output_dir}")
        return output_dir
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None