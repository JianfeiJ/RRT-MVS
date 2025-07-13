export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
BLD_TRAINING=" " # your blendedmvs low-res path
BLD_TRAINLIST="lists/blendedmvs/train.txt"
BLD_TESTLIST="lists/blendedmvs/val.txt"
BLD_CKPT_FILE="checkpoints/dtu/dtu_no_pe.ckpt"  # dtu pretrained model without pe, can eval with ndepths 16,8,8,4 on tnt-adv
#BLD_CKPT_FILE="checkpoints/dtu/dtu_best.ckpt"  # dtu pretrained model, only eval with ndepths 8,8,4,4 on tnt-adv

exp=$1
PY_ARGS=${@:2}

BLD_LOG_DIR="./checkpoints/bld"$exp
if [ ! -d $BLD_LOG_DIR ]; then
    mkdir -p $BLD_LOG_DIR
fi

python -m torch.distributed.launch --master_port 12345 --nproc_per_node=8 train_bld.py --logdir $BLD_LOG_DIR --dataset=blendedmvs --batch_size=2 --trainpath=$BLD_TRAINING --summary_freq 100 \
        --loadckpt $BLD_CKPT_FILE --ndepths 8,8,4,4 --depth_inter_r 0.5,0.5,0.5,0.5 --reg_mode reg2d_hybrid --group_cor --inverse_depth --rt --lr 0.001 --attn_temp 2 --trainlist $BLD_TRAINLIST --testlist $BLD_TESTLIST  $PY_ARGS | tee -a $BLD_LOG_DIR/log.txt
