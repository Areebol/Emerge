import matplotlib.pyplot as plt
from .meter import AverageMeter
import os
from utils.load_config import load_config
from utils.load_data import load_token_levels, load_sentence_levels

def _plot_sentence_token_level(title, model_configs, token_levels, sentence_levels,save_path=None):
    plt.figure(figsize=(16, 8))
    for model_config,token_level,sentence_level in zip(model_configs, token_levels, sentence_levels):
        plt.scatter(model_config[3], token_level,label=f"{model_config[0]}_token_level")
        plt.text(model_config[3],token_level,f"{model_config[0]}_token_level",fontsize=8,ha='left',va='bottom')
        
        plt.scatter(model_config[3], sentence_level,label=f"{model_config[0]}_sentence_level")
        plt.text(model_config[3],sentence_level,f"{model_config[0]}_sentence_level",fontsize=8,ha='left',va='bottom')
    plt.legend()
    plt.xlabel("params size")
    plt.suptitle(title,fontsize=32)
    if save_path:
        plt.savefig(f"{save_path}/{title}.png")
    plt.show()
    plt.close()

def plot_sentence_token_level(title, model_configs, token_levels, sentence_levels,save_path=None):
    plt.figure(figsize=(16, 8))
    for model_config,token_level,sentence_level in zip(model_configs, token_levels, sentence_levels):
        fraction = AverageMeter()
        for avg_token_levl,avg_sentence_level in zip(token_level,sentence_level):
            fraction.update((avg_token_levl/avg_sentence_level))
        plt.scatter(model_config[3], fraction.avg,label=model_config[0])
        plt.text(model_config[3],fraction.avg,model_config[0],fontsize=8,ha='left',va='bottom')
    plt.legend()
    plt.xlabel("params size")
    plt.suptitle(title,fontsize=32)
    if save_path:
        plt.savefig(f"{save_path}/{title}_fraction.png")
    plt.show()
    plt.close()

    # _plot_sentence_token_level(title, model_configs, token_levels, sentence_levels,save_path)

    
def plot_level(title, model_configs,levels,save_path=None):
    plt.figure(figsize=(16, 8))
    for model_config,level in zip(model_configs, levels):
        avg_level = sum(level) / len(level)
        plt.scatter(model_config[3], avg_level,label=model_config[0])
        plt.text(model_config[3],avg_level,model_config[0],fontsize=8,ha='left',va='bottom')
    plt.legend()
    plt.xlabel("params size")
    plt.suptitle(title,fontsize=32)
    if save_path:
        plt.savefig(f"{save_path}/{title}.png")
    plt.show()
    plt.close()
        
        
def plot_sentence_entropy(title, model_configs,token_levels,sentence_levels):
    for sentence_level in sentence_levels:
        plt.figure(figsize=(16, 8))
        plt.plot(range(len(sentence_level)),sentence_level,label="sentence_entropy")
        plt.plot(range(len(sentence_level)),[sum(sentence_level)/len(sentence_level) for _ in range(len(sentence_level))],label="sentence_level", linestyle='--')
        
        plt.legend()
        plt.xlabel("index")
        plt.suptitle(f"{title}",fontsize=32)
        plt.show()
        plt.close()
        
def plot_family_data(model_familys=["llama_2"],data_type="xsum_examples"):
    model_cfg = "./config/models_jq.yaml"
    # 加载模型s
    model_configs = []
    for key in model_familys:
        model_configs += load_config(model_cfg)[f"paths_{key}"]
    # 读取数据
    _, token_levels = load_token_levels(model_configs,data_type)
    models_configs, sentence_levels = load_sentence_levels(model_configs,data_type)
    if models_configs:
        save_path = f"./picture/tmp/{data_type}/{model_familys}"
        os.makedirs(save_path,exist_ok=True)
        # 绘图
        plot_sentence_token_level("token_level_sentence_level",models_configs,token_levels,sentence_levels,save_path)
        plot_level("token_level",model_configs,token_levels,save_path)
        plot_level("sentence_level",model_configs,sentence_levels,save_path)