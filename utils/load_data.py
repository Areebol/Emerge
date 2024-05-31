import csv
import ast
from .meter import AverageMeter

def load_token_levels(model_configs,data_type):
    load_dir = "./exp" + "/MeanEntropy" + f"/{data_type}"
    token_levels = []
    exist_model_configs = []
    for model_config in model_configs:
        load_file = f"{load_dir}/{model_config[0]}.csv"
        token_entropy = AverageMeter()
        try:
            with open(load_file, newline='') as csvfile:
                exist_model_configs.append(model_config)
                csvreader = csv.reader(csvfile)
                # 逐行读取CSV文件的内容
                for row in csvreader:
                    token_entropy.update(float(row[1]))
            token_levels.append(token_entropy.avg)
        except FileNotFoundError:
            print(f"{model_config[0]} not token_level")
    return exist_model_configs, token_levels

def load_sentence_levels(model_configs,data_type):
    load_dir = "./exp" + "/SentenceEntropy" + f"/{data_type}"
    sentence_levels = []
    exist_model_configs = []
    for model_config in model_configs:
        load_file = f"{load_dir}/{model_config[0]}.csv"
        sentence_entropy = AverageMeter()
        try:
            with open(load_file, newline='') as csvfile:
                print(f"loading file {load_file}")
                exist_model_configs.append(model_config)
                csvreader = csv.reader(csvfile)
                # 逐行读取CSV文件的内容
                for row in csvreader:
                    sentence_entropys = ast.literal_eval(row[1])
                    sentence_entropy.update(sum(sentence_entropys)/len(sentence_entropys))
            sentence_levels.append(sentence_entropy.avg)
        except FileNotFoundError:
            print(f"{model_config[0]} not sentence_level")
    return exist_model_configs, sentence_levels