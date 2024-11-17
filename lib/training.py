"""Training module optimized for Llama 3 fine-tuning on conversational data."""
from transformers import (
    TrainingArguments,
    EarlyStoppingCallback
)
from trl import SFTTrainer, SFTConfig
import torch
from typing import Optional, Dict, Tuple
import os
from datasets import Dataset
import logging
from pathlib import Path
import json
import gc
import wandb
from lib.model_util import setup_model_and_tokenizer, get_lora_config, get_peft_model

def save_training_config(output_dir: str, config: dict):
    """Save training configuration for reproducibility."""
    config_path = os.path.join(output_dir, "training_config.json")
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logging.info(f"Saved training configuration to {config_path}")
    except Exception as e:
        logging.warning(f"Failed to save training configuration: {str(e)}")

def get_training_args(output_dir: str, training_cfg: dict, data_cfg: dict) -> TrainingArguments:
    """Create training arguments from configuration."""
    
    # Check for multi-GPU setup
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Adjust batch size for multi-GPU
    base_batch_size = int(training_cfg.get('batch_size', 4))
    batch_size = base_batch_size // world_size if world_size > 1 else base_batch_size

    return TrainingArguments(
        output_dir=output_dir,
        max_steps=int(training_cfg.get('max_steps', -1)),
        num_train_epochs=float(training_cfg.get('num_train_epochs', 3)),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=int(training_cfg.get('gradient_accumulation_steps', 2)),
        learning_rate=float(training_cfg.get('learning_rate', 2e-4)),
        lr_scheduler_type=training_cfg.get('lr_scheduler', "cosine"),
        warmup_ratio=float(training_cfg.get('warmup_ratio', 0.1)),
        max_grad_norm=float(training_cfg.get('max_grad_norm', 0.3)),
        logging_steps=int(training_cfg.get('logging_steps', 10)),
        save_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=int(training_cfg.get('evaluation_steps', 100)),
        save_steps=int(training_cfg.get('save_steps', 100)),
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        group_by_length=True,
        optim=training_cfg.get('optimizer', "paged_adamw_32bit"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=True if world_size > 1 else False,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        report_to=["tensorboard", "wandb"],
        save_total_limit=3,
        ddp_backend="nccl" if world_size > 1 else None,
        local_rank=local_rank if world_size > 1 else -1
    )

def setup_trainer(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    dataset: Dataset,
    training_args: TrainingArguments,
    lora_config: dict,
    config: dict
) -> SFTTrainer:
    """Initialize and configure the trainer."""
    early_stopping_patience = int(config['training'].get('early_stopping', {}).get('patience', 3))
    early_stopping_threshold = float(config['training'].get('early_stopping', {}).get('threshold', 0.01))

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        args=training_args,
        peft_config=lora_config,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold
            )
        ],
        packing=False,
        max_seq_length=int(config['data'].get('max_seq_length', 2048))
    )

def save_model_and_tokenizer(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    output_dir: str,
    config: dict,
    train_result: dict
) -> str:
    """Save the model, tokenizer, and training details."""
    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)

    logging.info("Merging and saving model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(
        final_model_path,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    tokenizer.save_pretrained(final_model_path)

    # Get dataset info with type checking
    dataset_info = config.get('dataset_info', {})
    train_samples = int(dataset_info.get('train_samples', 0))
    eval_samples = int(dataset_info.get('eval_samples', 0))

    # Extract training metrics safely
    train_metrics = train_result.metrics if hasattr(train_result, 'metrics') else {}
    training_loss = float(train_result.training_loss) if hasattr(train_result, 'training_loss') else None
    
    # Save model card with validated data
    model_card = {
        "base_model": config['model']['name'],
        "training_config": config,
        "dataset_info": {
            "train_samples": train_samples,
            "eval_samples": eval_samples
        },
        "training_results": {
            "train_loss": training_loss,
            "train_runtime": train_metrics.get("train_runtime"),
            "train_samples_per_second": train_metrics.get("train_samples_per_second")
        }
    }

    with open(os.path.join(final_model_path, "model_card.json"), 'w') as f:
        json.dump(model_card, f, indent=2)

    return final_model_path

def fine_tune_model(
    dataset: Dataset, 
    output_dir: str, 
    config: dict
) -> Optional[str]:
    """Fine-tune model optimized for conversational data transfer."""
    try:
        # Initial cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        # Initialize wandb
        wandb.init(
            project=config.get('wandb', {}).get('project', 'llama-finetuning'),
            name=config.get('wandb', {}).get('run_name', f"run_{Path(output_dir).name}"),
            config=config
        )
        
        # Create output directory and save config
        os.makedirs(output_dir, exist_ok=True)
        save_training_config(output_dir, config)
        
        # Setup model and tokenizer
        model_name = config['model']['name']
        model, tokenizer = setup_model_and_tokenizer(model_name, config)
        model.config.use_cache = False
        
        # Get LoRA configuration and apply
        lora_config = get_lora_config(model, config)
        model = get_peft_model(model, lora_config)
        
        # Setup training arguments
        training_args = get_training_args(output_dir, config['training'], config['data'])
        
        # Initialize trainer with preprocessed dataset
        trainer = setup_trainer(
            model,
            tokenizer,
            dataset,
            training_args,
            lora_config,
            config
        )
        
        # Training
        logging.info("Starting training...")
        train_result = trainer.train()
        logging.info(f"Training completed. Results: {train_result}")
        
        # Save trainer state
        trainer.save_state()
        
        # Cleanup before model saving
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        # Save model and artifacts
        final_model_path = save_model_and_tokenizer(
            model,
            tokenizer,
            output_dir,
            config,
            train_result
        )
        
        # Final cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        # Close wandb run
        wandb.finish()
        
        logging.info(f"Training completed. Model saved to: {final_model_path}")
        return final_model_path
        
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        # Ensure wandb run is closed even if there's an error
        wandb.finish()
        return None