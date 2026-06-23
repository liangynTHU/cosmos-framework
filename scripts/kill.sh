EXP_IP_LIST=$(sed 's/:8//g' <<< "$NODE_IP_LIST")
echo $EXP_IP_LIST

# pdsh -w $EXP_IP_LIST "pkill torchrun; pkill pt_elastic; sleep 10"
pdsh -w $EXP_IP_LIST "pkill -f 'cosmos_framework.scripts.train'"


