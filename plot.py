from utils.plotter import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name",type=str)
parser.add_argument("--num_lora_model", type=int)
args = parser.parse_args()

# model_familys = [["llama_2"]]
data_types = ["xsum_examples","Cot_examples","hc3_human_examples","poem_sentiment_classification"]
# for model_family in model_familys:
#     for data_type in data_types:
#         plot_family_data(model_family,data_type=data_type)
for data_type in data_types:
    plot_ft_data(model_name=args.model_name, num_lora_model=args.num_lora_model, data_type=data_type)