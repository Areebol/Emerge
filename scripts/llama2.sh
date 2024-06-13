CFG=./config/llama2.yaml
for ((i=0;i<6;i++))
do 
    python main.py --cfg $CFG --start $i
done
