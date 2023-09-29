gpu=$1;
log='secu_entropy_cifar10';
CUDA_VISIBLE_DEVICES=${gpu} python main.py \
    -j 8 \
    -p 100 \
    --epochs 401 \
    --clr 1.2 \
    --min-crop 0.3 \
    --data-name 'cifar10' \
    --secu-alpha 6000 \
    --secu-cst 'entropy' \
    --log ${log} \
    --dist-url 'tcp://localhost:1234' --multiprocessing-distributed --world-size 1 --rank 0 \
    /path/to/cifar10/ | tee log/${log}.log;