import os
# 使用实验室clash代理
# os.environ['http_proxy']='127.0.0.1:7890'
# os.environ['https_proxy']='127.0.0.1:7890'

# 使用国内镜像
os.environ['HF_ENDPOINT']='https://hf-mirror.com'

from huggingface_hub import snapshot_download
import huggingface_hub

# huggingface_hub.login('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')
huggingface_hub.login('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')

if __name__ == "__main__":
    cache_dir = "/U_20240603_ZSH_SMIL/LLM"
    # April 30th  
    # snapshot_download(repo_id="Qwen/Qwen1.5-0.5B",cache_dir=cache_dir)
    # snapshot_download(repo_id="Qwen/Qwen1.5-0.5B-Chat",cache_dir=cache_dir)
    # snapshot_download(repo_id="Qwen/Qwen1.5-1.8B",cache_dir=cache_dir)
    # snapshot_download(repo_id="Qwen/Qwen1.5-1.8B-Chat",cache_dir=cache_dir)
    # snapshot_download(repo_id="Qwen/Qwen1.5-4B",cache_dir=cache_dir)
    # snapshot_download(repo_id="Qwen/Qwen1.5-4B-Chat",cache_dir=cache_dir)
    # snapshot_download(repo_id="Qwen/Qwen1.5-7B",cache_dir=cache_dir)
    # snapshot_download(repo_id="Qwen/Qwen1.5-7B-Chat",cache_dir=cache_dir)
    # snapshot_download(repo_id="Qwen/Qwen1.5-14B",cache_dir=cache_dir)
    # snapshot_download(repo_id="Qwen/Qwen1.5-14B-Chat",cache_dir=cache_dir)
    # snapshot_download(repo_id="Qwen/Qwen1.5-32B",cache_dir=cache_dir)
    # snapshot_download(repo_id="Qwen/Qwen1.5-32B-Chat",cache_dir=cache_dir)
    # snapshot_download(repo_id="Qwen/Qwen1.5-72B",cache_dir=cache_dir)
    # snapshot_download(repo_id="Qwen/Qwen1.5-72B-Chat",cache_dir=cache_dir)    
    
    # May 29th
    # qwen 1 
    repo_ids = [
                "meta-llama/CodeLlama-7b-hf",
                "meta-llama/CodeLlama-7b-Python-hf",
                "meta-llama/CodeLlama-7b-Instruct-hf",
                "meta-llama/CodeLlama-13b-hf",
                "meta-llama/CodeLlama-13b-Python-hf",
                "meta-llama/CodeLlama-13b-Instruct-hf",
                "meta-llama/CodeLlama-34b-hf",
                "meta-llama/CodeLlama-34b-Python-hf",
                "meta-llama/CodeLlama-34b-Instruct-hf",
                # "Qwen/Qwen-1_8B-Chat",
                # "Qwen/Qwen-1_8B",
                # "Qwen/Qwen-7B-Chat",
                # "Qwen/Qwen-7B",
                # "Qwen/Qwen-14B-Chat",
                # "Qwen/Qwen-14B",
                # "Qwen/Qwen-72B-Chat",
                # "Qwen/Qwen-72B",
                # "meta-llama/Llama-2-7b-hf",
                # "meta-llama/Llama-2-7b-chat-hf",
                # "meta-llama/Llama-2-13b-hf",
                # "meta-llama/Llama-2-13b-chat-hf",
                # "meta-llama/Llama-2-70b-hf",
                # "meta-llama/Llama-2-70b-chat-hf",
                ]
    for repo_id in repo_ids:
        print(f"repod_id:{repo_id}")
        snapshot_download(repo_id=repo_id,cache_dir=cache_dir)
    