from utils.load_model import *
from utils.load_config import load_config
import data_loader
import data_processor
from pipeline.pipeline import Pipeline
import logging
import argparse
import gc
import torch 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config/qwen1_5.yaml", help="config file path")
    parser.add_argument("--start", default = 0, type=int, help="config file path")
    parser.add_argument("--model_cfg", default="./config/models_pz.yaml", help="model config file path")
    args = parser.parse_args()

    log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
    # logging.basicConfig(level="DEBUG", format=log_f)
    logging.basicConfig(level="INFO", format=log_f)
    
    # load config 
    config = load_config(args.cfg)
    model_cfg = load_config(args.model_cfg)
    for step in range(15,30):
        model_config = [f'llama2_13B_ft_{step}','xxx','llama2',step]
        print(f"Loading model: checkpoint-{step}000")
        model, tokenizer = load_merge_model_tokenizer(lora_model_name = f'checkpoint-{step}000')
        # data loaders + data processors
        data_loaders = [getattr(data_loader,loader_name)() for loader_name in config['data_loaders']]
        data_processors = [getattr(data_processor,processor_name)(model, tokenizer, model_config) for processor_name in config['data_processors']]
        # init pipeline
        pipeline = Pipeline(model,tokenizer,model_config,data_loaders,data_processors)
        # run
        pipeline.run()
        