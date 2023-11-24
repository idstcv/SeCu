log='secu_size_imagenet';
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_imagenet.py \
    --secu-cst 'size' \
    --log ${log} \
    --dist-url 'tcp://localhost:1234' --multiprocessing-distributed --world-size 1 --rank 0 \
    /path/to/imagenet/ | tee log/${log}.log;