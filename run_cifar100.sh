gpu=$1;
log='secu_size_cifar100';
CUDA_VISIBLE_DEVICES=${gpu} python main.py \
    -j 8 \
    -p 100 \
    --epochs 801 \
    --clr 0.01 \
    --min-crop 0.2 \
    --secu-k 100 200 300 400 500 600 700 800 900 1000 \
    --data-name 'cifar100' \
    --secu-cst 'size' \
    --log ${log} \
    --dist-url 'tcp://localhost:1234' --multiprocessing-distributed --world-size 1 --rank 0 \
    /path/to/cifar100/ | tee log/${log}.log;