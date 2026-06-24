EXP_IP_LIST=$(sed 's/:8//g' <<< "$NODE_IP_LIST")
echo $EXP_IP_LIST

# pdsh -w $EXP_IP_LIST "pkill torchrun; pkill pt_elastic; sleep 10"
# pdsh -w $EXP_IP_LIST "pkill -TERM -f 'run_train_robotwin_cosmos3nano.sh'"

pdsh -w "$EXP_IP_LIST" \
'pkill -9 -u root -f "[/]jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python.*-m [c]osmos_framework.scripts.train" || true'