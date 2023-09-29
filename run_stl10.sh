gpu=$1;
log='secu_size_stl10_stage_1';
CUDA_VISIBLE_DEVICES=${gpu} python main_final.py \
    -j 8 \
    -p 100 \
    --epochs 801 \
    --clr 0.8 \
    --min-crop 0.2 \
    --secu-num-ins 105000 \
    --secu-tw 1 \
    --data-name 'stl10' \
    --secu-cst 'size' \
    --log ${log} \
    --dist-url 'tcp://localhost:1234' --multiprocessing-distributed --world-size 1 --rank 0 \
    /path/to/stl10/ | tee log/${log}.log;