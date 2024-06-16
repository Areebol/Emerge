from utils.plotter import plot_family_data

model_familys = [["qwen_1.5"]]
data_types = ["xsum_examples","Cot_examples","hc3_human_examples","poem_sentiment_classification"]
for model_family in model_familys:
    for data_type in data_types:
        plot_family_data(model_family,data_type=data_type)