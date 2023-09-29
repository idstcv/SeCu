gpu=$1;
log='secu_size_cifar100_20';
CUDA_VISIBLE_DEVICES=${gpu} python main.py \
    -j 8 \
    -p 100 \
    --epochs 801 \
    --clr 0.8 \
    --min-crop 0.2 \
    --secu-k 20 40 60 80 100 120 140 160 180 200 \
    --data-name 'cifar100' \
    --secu-cst 'size' \
    --log ${log} \
    --dist-url 'tcp://localhost:1234' --multiprocessing-distributed --world-size 1 --rank 0 \
    /path/to/cifar100/ | tee log/${log}.log;