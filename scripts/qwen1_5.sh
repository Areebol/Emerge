CFG=./config/qwen1_5.yaml
for ((i=8;i<12;i++))
do 
    python main.py --cfg $CFG --start $i
done