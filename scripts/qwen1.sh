CFG=./config/qwen1.yaml
for ((i=0;i<8;i++))
do 
    python main.py --cfg $CFG --start $i
done
