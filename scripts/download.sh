# 启动Clash脚本命令
# /share/home/zhangshuhai/ogsp/emerge/clash/clash-linux-amd64
# export http_proxy=127.0.0.1:7890 && export https_proxy=127.0.0.1:7890 && curl www.google.com
export HF_ENDPOINT=https://hf-mirror.com
duration=3600  # 1hr

num_iterations=100

for ((i=1; i<=$num_iterations; i++)); do
    python scripts/model_download.py 
    # program_pid=$!  # 获取程序的进程 ID
    # # 等待固定时间
    # sleep $duration

    # # 杀死程序进程
    # kill $program_pid
    # wait $program_pid 2>/dev/null  # 等待进程被杀死
done