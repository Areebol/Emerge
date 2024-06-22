# -*- coding: UTF-8 -*-
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import logging
import torch
import os
from transformers import (
    AutoModel,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    AutoModelForCausalLM,
    LlamaForCausalLM,
)
from peft import PeftModel, PeftConfig

def elements_in_path(path,elements):
    """
    path: 
    elements: 
    """
    for element in elements:
        if element in path:
            return True
    return False

def load_model_tokenizer(model_config=None,half_models=['12b','13b','14b','32b','34b','70b','72b']):
    """
    model: [model_name, model_path, model_family, model_param_size]
    """
    tokenizer = AutoTokenizer.from_pretrained(model_config[1], fast_tokenizer=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_config[1], output_attentions=True, attn_implementation="eager", trust_remote_code=True)
    if elements_in_path(model_config[0],half_models):
        logging.info(f"Loading model [{model_config[0]}] in half mode")
        model = AutoModelForCausalLM.from_pretrained(model_config[1], device_map="auto", torch_dtype=torch.float16, config=config, trust_remote_code=True)
        # model = AutoModelForCausalLM.from_pretrained(model_config[1], device_map="auto", config=config).half() # half load
    else:
        logging.info(f"Loading model [{model_config[0]}] in full mode")
        model = AutoModelForCausalLM.from_pretrained(model_config[1], device_map="auto", config=config,trust_remote_code=True)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    return model,tokenizer

def load_merge_model_tokenizer(base_model_path = '/U_20240603_ZSH_SMIL/LLM/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1',
                     lora_model_base_dir  = '/U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-llama2-13b-epoch10-v1',
                     lora_model_name = 'checkpoint-1000',
                     model_type = 'auto'):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, fast_tokenizer=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(base_model_path, output_attentions=True, attn_implementation="eager", trust_remote_code=True)
    model = merge_lora_model(base_model_path,lora_model_base_dir,lora_model_name,model_type=model_type,config=config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    return model,tokenizer

def merge_lora_model(base_model_path = '/U_20240603_ZSH_SMIL/LLM/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1',
                     lora_model_base_dir  = '/U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-llama2-13b-epoch10-v1',
                     lora_model_name = 'checkpoint-1000',
                     model_type = 'llama',
                     config=None):
    MODEL_CLASSES = {
        "bloom": (BloomForCausalLM, BloomTokenizerFast),
        "chatglm": (AutoModel, AutoTokenizer),
        "llama": (LlamaForCausalLM, AutoTokenizer),
        "baichuan": (AutoModelForCausalLM, AutoTokenizer),
        "auto": (AutoModelForCausalLM, AutoTokenizer),
    }
    model_class, _ = MODEL_CLASSES[model_type]
    print("Loading LoRA for causal language model")
    base_model = model_class.from_pretrained(
        base_model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        config = config,
    )
    new_model: PeftModel = PeftModel.from_pretrained(
        base_model,
        os.path.join(lora_model_base_dir, lora_model_name),
        device_map="auto",
        torch_dtype=torch.float16,
    )

    return new_model.merge_and_unload()