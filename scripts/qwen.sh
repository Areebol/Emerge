CFG=./config/qwen.yaml
for ((i=0;i<1;i++))
do 
    python main.py --cfg $CFG --start $i
done
