from utils.load_model import load_model_tokenizer
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
    parser.add_argument("--model_cfg", default="./config/models_jq.yaml", help="model config file path")
    args = parser.parse_args()

    log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
    # logging.basicConfig(level="DEBUG", format=log_f)
    logging.basicConfig(level="INFO", format=log_f)
    
    # load config 
    config = load_config(args.cfg)
    model_cfg = load_config(args.model_cfg)

    model_familys = config['model_familys']
    model_configs = []
    for key in model_familys:
        model_configs += model_cfg[f"paths_{key}"]
        
    # models
    for model_config in model_configs[args.start:args.start+1]:
        model, tokenizer = load_model_tokenizer(model_config=model_config)
        # data loaders + data processors
        data_loaders = [getattr(data_loader,loader_name)() for loader_name in config['data_loaders']]
        data_processors = [getattr(data_processor,processor_name)(model, tokenizer, model_config) for processor_name in config['data_processors']]
        # init pipeline
        pipeline = Pipeline(model,tokenizer,model_config,data_loaders,data_processors)
        # run
        pipeline.run()
        