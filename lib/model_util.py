"""Model utilities for training and inference."""
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    PeftModel,
    prepare_model_for_kbit_training
)
import torch
from typing import Optional, Dict, Tuple, Union
import os
import logging
from pathlib import Path
import json

# Enable TF32 for matrix multiplications
torch.backends.cuda.matmul.allow_tf32 = True

def has_multiple_gpus() -> bool:
    """Check if multiple GPUs are available."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logging.info(f"Found {gpu_count} GPUs")
        return gpu_count > 1
    return False

def setup_tokenizer(model_name: str, config: dict = None) -> AutoTokenizer:
    """Initialize tokenizer with consistent settings."""
    logging.info(f"Initializing tokenizer for {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        truncation_side="left",
        trust_remote_code=True
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set pad_token to eos_token")
    
    # Set chat template if not set
    try:
        if not tokenizer.chat_template:
            base_template = """<|begin_of_text|>{% for message in messages %}{% if message.role == 'system' %}System: {{ message.content }}\\n\\n{% elif message.role == 'user' %}Human: {{ message.content }}\\n{% elif message.role == 'assistant' %}Assistant: {{ message.content }}\\n\\n{% endif %}{% endfor %}"""
            tokenizer.chat_template = base_template
            logging.info("Set base model chat template")
    except Exception as e:
        logging.warning(f"Error setting chat template: {str(e)}")
    
    return tokenizer

def get_compute_dtype(use_bf16: Optional[bool] = None) -> torch.dtype:
    """Determine optimal compute dtype based on hardware capabilities."""
    if use_bf16 is True:
        return torch.bfloat16
    elif use_bf16 is False:
        return torch.float16
        
    # Auto-detect if not specified
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def get_quantization_config(config: dict, compute_dtype: torch.dtype) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytes configuration based on settings."""
    quant_config = config.get('model', {}).get('quantization', {})
    
    # Check for multiple GPUs and disable quantization if found
    if has_multiple_gpus():
        logging.info("Multiple GPUs detected - disabling quantization")
        return None
    
    # Return None if quantization is disabled
    if not quant_config.get('enabled', True):
        logging.info("Quantization disabled in config")
        return None
        
    # Determine quantization mode
    load_in_4bit = quant_config.get('load_in_4bit', True)
    load_in_8bit = quant_config.get('load_in_8bit', False) if not load_in_4bit else False
    
    if load_in_4bit:
        logging.info("Quantization enabled: Loading in 4-bit")
    elif load_in_8bit:
        logging.info("Quantization enabled: Loading in 8-bit")
    else:
        logging.info("No valid quantization mode specified (4-bit or 8-bit)")
        return None
        
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_quant_type=quant_config.get('quant_type', "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_config.get('use_double_quant', True)
    )

def setup_model_and_tokenizer(
    model_name: str,
    config: dict,
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Setup model with optimal configurations for conversation fine-tuning."""
    try:
        # Determine compute dtype
        compute_dtype = get_compute_dtype(config.get('model', {}).get('use_bf16', None))
        torch_dtype = torch_dtype or compute_dtype
        
        # Get quantization configuration
        bnb_config = get_quantization_config(config, compute_dtype)
        
        # Log configuration details
        logging.info(f"Loading model {model_name}")
        logging.info(f"Compute dtype: {compute_dtype}")
        logging.info(f"Quantization config: {'Enabled' if bnb_config else 'Disabled'}")

        # Load model with optimal settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa",
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )

        # Enable Flash Attention SDP
        torch.backends.cuda.enable_flash_sdp(True)
        
        # Prepare model for k-bit training if using quantization
        if bnb_config:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=config.get('training', {}).get('gradient_checkpointing', True)
            )
        
        # Use consolidated tokenizer setup
        tokenizer = setup_tokenizer(model_name, config)

        # Set pad_token_id if it's not already set
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        
        logging.info(f"Successfully loaded model and tokenizer")
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
        
        target_modules = lora_config.get('target_modules', default_target_modules)
        logging.info(f"Configuring LoRA with target modules: {target_modules}")
        
        # Create LoRA configuration
        return LoraConfig(
            r=lora_config.get('rank', 32),
            lora_alpha=lora_config.get('alpha', 64),
            target_modules=target_modules,
            lora_dropout=lora_config.get('dropout', 0.1),
            bias=lora_config.get('bias', "none"),
            task_type="CAUSAL_LM",
            inference_mode=False,
            init_lora_weights=lora_config.get('init_weights', "gaussian")
        )
    except Exception as e:
        logging.error(f"Error in get_lora_config: {str(e)}")
        raise

def load_fine_tuned_model(
    model_path: str,
    device_map: str = "auto",
    inference_mode: bool = True,
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
        
        # Determine optimal torch dtype if not specified
        if torch_dtype is None:
            torch_dtype = get_compute_dtype()
        
        # Load model with optimal inference settings
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa",
            trust_remote_code=True
        )
        
        if inference_mode:
            model.eval()
        
        # Use consolidated tokenizer setup
        tokenizer = setup_tokenizer(model_path)
        
        # Log model configuration
        logging.info(f"Model loaded with device map: {device_map}")
        logging.info(f"Model dtype: {torch_dtype}")
        
        return model, tokenizer
        
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None, None

def get_model_info(model_path: str) -> Optional[Dict]:
    """Get information about a trained model."""
    try:
        info = {}
        
        # Load model card if available
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
        
        # Get model shards info
        shards = [f for f in os.listdir(model_path) if f.startswith('pytorch_model') and f.endswith('.bin')]
        info["num_shards"] = len(shards)
        
        return info
        
    except Exception as e:
        logging.error(f"Error getting model info: {str(e)}")
        return None