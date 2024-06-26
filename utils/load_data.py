import csv
import ast
from .meter import AverageMeter
import re

def load_token_levels(model_configs,data_type,token_type="Mean"):
    print(f"token_type == {token_type}")
    exp_dir = "./exp"
    load_dir = exp_dir + f"/{token_type}" + f"/{data_type}"
    token_levels = []
    exist_model_configs = []
    for model_config in model_configs:
        load_file = f"{load_dir}/{model_config[0]}.csv"
        token_entropy = []
        try:
            with open(load_file, newline='') as csvfile:
                print(f"loading file {load_file}")
                exist_model_configs.append(model_config)
                csvreader = csv.reader(csvfile)
                # 逐行读取CSV文件的内容
                for row in csvreader:
                    token_entropy.append(float(row[1]))
            token_levels.append(token_entropy)
        except FileNotFoundError:
            print(f"{model_config[0]} not token_level")
    return (token_levels,exist_model_configs)

def load_sentence_levels(model_configs,data_type,sentence_type="Softmax"):
    print(f"sentence_type == {sentence_type}")
    exp_dir = "./exp"
    load_dir = exp_dir + f"/{sentence_type}" + f"/{data_type}"
        
    sentence_levels = []
    exist_model_configs = []
    for model_config in model_configs:
        load_file = f"{load_dir}/{model_config[0]}.csv"
        sentence_entropy = []
        try:
            with open(load_file, newline='') as csvfile:
                print(f"loading file {load_file}")
                csvreader = csv.reader(csvfile)
                # 逐行读取CSV文件的内容
                for row in csvreader:
                    sentence_entropys = ast.literal_eval(row[1])
                    sentence_entropy.append(sum(sentence_entropys[1:])/(len(sentence_entropys)-1))
            exist_model_configs.append(model_config)
            sentence_levels.append(sentence_entropy)
        except FileNotFoundError:
            print(f"{model_config[0]} not sentence_level")
        except ValueError:
            print(f"{model_config[0]} appear nan value")
    return (sentence_levels,exist_model_configs)
