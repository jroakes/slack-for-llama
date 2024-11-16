"""Training module optimized for Llama 3 fine-tuning on conversational data."""
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    PeftModel,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, SFTConfig
import torch
from typing import Optional, Dict, Tuple, Union
import os
from datasets import Dataset
import logging
from pathlib import Path
import json
import gc
import shutil


def setup_model_and_tokenizer(
    model_name: str,
    config: dict,
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Setup model with optimal configurations for conversation fine-tuning."""
    try:
        # Determine optimal torch dtype
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Get quantization settings from config
        quant_config = config.get('model', {}).get('quantization', {})
        
        # Optimized 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.get('load_in_4bit', True),
            bnb_4bit_quant_type=quant_config.get('quant_type', "nf4"),
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=quant_config.get('use_double_quant', True)
        )

        # Load model with optimal settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa",
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Load and configure tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="right",  # Changed to right padding for stability
            truncation_side="left",
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model.config.pad_token_id = tokenizer.pad_token_id
        
        logging.info(f"Successfully loaded model and tokenizer: {model_name}")
        return model, tokenizer
        
    except Exception as e:
        logging.error(f"Error in setup_model_and_tokenizer: {str(e)}")
        raise

def get_lora_config(model: AutoModelForCausalLM, config: dict) -> LoraConfig:
    """Get LoRA config optimized for conversational knowledge transfer."""
    try:
        # Get LoRA settings from config
        lora_config = config.get('training', {}).get('lora', {})
        
        # Default target modules for Llama models
        default_target_modules = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj'
        ]
        
        # Create LoRA configuration
        return LoraConfig(
            r=lora_config.get('rank', 32),
            lora_alpha=lora_config.get('alpha', 64),
            target_modules=lora_config.get('target_modules', default_target_modules),
            lora_dropout=lora_config.get('dropout', 0.1),
            bias=lora_config.get('bias', "none"),
            task_type="CAUSAL_LM",
            inference_mode=False,
            init_lora_weights=lora_config.get('init_weights', "gaussian")
        )
    except Exception as e:
        logging.error(f"Error in get_lora_config: {str(e)}")
        raise

def save_training_config(output_dir: str, config: dict):
    """Save training configuration for reproducibility."""
    config_path = os.path.join(output_dir, "training_config.json")
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logging.info(f"Saved training configuration to {config_path}")
    except Exception as e:
        logging.warning(f"Failed to save training configuration: {str(e)}")

def fine_tune_model(
    dataset: Dataset, 
    output_dir: str, 
    config: dict
) -> Optional[str]:
    """Fine-tune model optimized for conversational data transfer."""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training configuration
        save_training_config(output_dir, config)
        
        # Setup model and tokenizer
        model_name = config['model']['name']
        model, tokenizer = setup_model_and_tokenizer(model_name, config)
        
        # Get LoRA configuration
        lora_config = get_lora_config(model, config)
        model = get_peft_model(model, lora_config)
        
        # Get training settings
        training_cfg = config['training']
        
        # Initialize trainer with early stopping callback
        early_stopping_patience = training_cfg.get('early_stopping', {}).get('patience', 3)
        early_stopping_threshold = training_cfg.get('early_stopping', {}).get('threshold', 0.01)

        # Create SFTConfig
        sft_config = SFTConfig(
            max_seq_length=config['data'].get('max_seq_length', 2048),
            dataset_text_field="text",
            packing=True
        )

        # Create training arguments with distinct run_name
        training_args = TrainingArguments(
            output_dir=output_dir,
            run_name=f"llama3_training_{Path(output_dir).name}",  # Distinct run name
            num_train_epochs=training_cfg.get('num_train_epochs', 3),
            per_device_train_batch_size=training_cfg.get('batch_size', 4),
            gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 2),
            learning_rate=training_cfg.get('learning_rate', 2e-4),
            lr_scheduler_type=training_cfg.get('lr_scheduler', "cosine"),
            warmup_ratio=training_cfg.get('warmup_ratio', 0.1),
            max_grad_norm=training_cfg.get('max_grad_norm', 0.3),
            logging_steps=training_cfg.get('logging_steps', 10),
            save_strategy="steps",
            evaluation_strategy="steps",
            eval_steps=training_cfg.get('evaluation_steps', 100),
            save_steps=training_cfg.get('save_steps', 100),
            bf16=torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            group_by_length=True,
            optim=training_cfg.get('optimizer', "paged_adamw_32bit"),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=training_args,
            peft_config=lora_config,
            tokenizer=tokenizer,
            config=sft_config,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_threshold=early_stopping_threshold
                )
            ]
        )
        
        # Training
        logging.info("Starting training...")
        trainer.train()
        
        # Save and merge model
        final_model_path = os.path.join(output_dir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        
        logging.info("Merging and saving model...")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(final_model_path, safe_serialization=True)
        tokenizer.save_pretrained(final_model_path)
        
        # Save model card with training details
        model_card = {
            "base_model": model_name,
            "training_config": config,
            "dataset_info": {
                "train_samples": len(dataset["train"]),
                "eval_samples": len(dataset["test"])
            }
        }
        
        with open(os.path.join(final_model_path, "model_card.json"), 'w') as f:
            json.dump(model_card, f, indent=2)
        
        # Cleanup
        del trainer, model, merged_model
        torch.cuda.empty_cache()
        gc.collect()
        
        logging.info(f"Training completed. Model saved to: {final_model_path}")
        return final_model_path
        
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        return None

def load_fine_tuned_model(
    model_path: str,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None
) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """Load fine-tuned model for inference with optimal settings."""
    try:
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
            
        # Load model card if available
        model_card_path = os.path.join(model_path, "model_card.json")
        if os.path.exists(model_card_path):
            with open(model_card_path, 'r') as f:
                model_card = json.load(f)
            logging.info(f"Loaded model card: {model_card.get('base_model', 'unknown')}")
        
        # Determine optimal torch dtype
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Load model with optimal settings
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa",
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logging.info(f"Successfully loaded model from: {model_path}")
        return model, tokenizer
        
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None, None

def get_model_info(model_path: str) -> Optional[Dict]:
    """Get information about a trained model."""
    try:
        info = {}
        
        # Load model card
        model_card_path = os.path.join(model_path, "model_card.json")
        if os.path.exists(model_card_path):
            with open(model_card_path, 'r') as f:
                info["model_card"] = json.load(f)
        
        # Get model size
        model_size = 0
        for path, _, files in os.walk(model_path):
            for f in files:
                fp = os.path.join(path, f)
                model_size += os.path.getsize(fp)
        info["model_size_gb"] = model_size / (1024**3)
        
        # Get training config if available
        config_path = os.path.join(model_path, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                info["training_config"] = json.load(f)
        
        return info
        
    except Exception as e:
        logging.error(f"Error getting model info: {str(e)}")
        return None