export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DTU_TRAINING=" " # your dtu training path
DTU_TRAINLIST="lists/dtu/train.txt"
DTU_TESTLIST="lists/dtu/test.txt"

exp=$1
PY_ARGS=${@:2}

DTU_LOG_DIR="./checkpoints/dtu"$exp
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi

python -m torch.distributed.launch --master_port 12345 --nproc_per_node=8 train_dtu.py --logdir $DTU_LOG_DIR --dataset=dtu_yao4 --batch_size=2 --trainpath=$DTU_TRAINING --summary_freq 100 \
        --depth_inter_r 0.5,0.5,0.5,0.5 --attn_temp 2 --ndepths 8,8,4,4 --trainlist $DTU_TRAINLIST --testlist $DTU_TESTLIST $PY_ARGS | tee -a $DTU_LOG_DIR/log.txt