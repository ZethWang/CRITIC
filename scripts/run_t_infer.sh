set -ex

DATA="toxicity"

# MODEL="text-davinci-003"
# MODEL="gpt-3.5-turbo"
MODEL="glm-4-flash"
SPLIT="test"
NUM_SAMPLING=3
START=0
END=-1
TEMPERATURE=0.9
# 创建日志目录以保存输出日志
mkdir -p logs/$MODEL/$DATA


# 设置 Python 调试命令
RUN_CMD="python -m debugpy --listen 7550 --wait-for-client"
# 构建完整命令
cmd="${RUN_CMD} src/toxicity/inference.py \
   --model $MODEL \
    --data $DATA \
    --split $SPLIT \
    --seed 0 \
    --start $START \
    --end $END \
    --temperature $TEMPERATURE \
    --num_sampling $NUM_SAMPLING"

# 打印并执行命令，将输出重定向到日志文件

echo " cmd: $cmd "
$cmd > logs/$MODEL/$DATA/${SPLIT}_s${START}_e${END}_t${TEMPERATURE}.log 2>&1&
