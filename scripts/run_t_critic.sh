set -ex

DATA="toxicity"

# MODEL="text-davinci-003"
# MODEL="gpt-3.5-turbo"
MODEL="glm-4-flash"
START=0
SPLIT="test"
NUM_SAMPLING=3
TEMPERATURE=0.9
CRITIC_TYPE="critic"
USE_TOOL=true

# CRITIC_TYPE=critic_v1_no-tool
# USE_TOOL=false

mkdir -p logs/$MODEL/$DATA



# END=$(expr 100 + $START)
END=-1

RUN_CMD="python -m debugpy --listen 7550 --wait-for-client" 

cmd="${RUN_CMD} src/toxicity/critic.py \
    --model $MODEL \
    --data $DATA \
    --split $SPLIT \
    --critic_type $CRITIC_TYPE \
    --use_tool $USE_TOOL \
    --seed 0 \
    --start $START \
    --end $END \
    --temperature $TEMPERATURE \
    --num_sampling $NUM_SAMPLING"

echo "cmd: $cmd"

$cmd > logs/$MODEL/$DATA/${SPLIT}_critic_tools-${USE_TOOL}_s${START}_e${END}_t${TEMPERATURE}.log 2>&1&

