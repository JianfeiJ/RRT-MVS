export CUDA_VISIBLE_DEVICES=0
DTU_TESTPATH=" " # your dtu test path
DTU_TESTLIST="lists/dtu/test.txt"
DTU_CKPT_FILE="checkpoints/dtu/dtu_best.ckpt"  # dtu pretrained model

exp=$1
PY_ARGS=${@:2}

DTU_LOG_DIR="outputs/dtu/pcd"$exp
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi
DTU_OUT_DIR="outputs/dtu/pcd"$exp
if [ ! -d $DTU_OUT_DIR ]; then
    mkdir -p $DTU_OUT_DIR
fi

python test_dtu_pcd.py --dataset=general_eval4 --batch_size=1 --testpath=$DTU_TESTPATH  --testlist=$DTU_TESTLIST --loadckpt $DTU_CKPT_FILE --interval_scale 1.06 --outdir $DTU_OUT_DIR\
             --thres_view 2 --conf 0.60 --filter_method normal --num_view 5 --attn_temp 2 $PY_ARGS | tee -a $DTU_LOG_DIR/log_test.txt
