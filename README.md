# Accumulative Poisoning Attacks on Real-time data

The code for the paper '[Accumulative Poisoning Attacks on Real-time data](https://arxiv.org/pdf/2106.09993.pdf)'.

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 18.04.4
- GPU: Geforce 2080 Ti or Tesla P100
- Cuda: 10.1, Cudnn: v7.6
- Python: 3.6
- PyTorch: >= 1.6.0
- Torchvision: >= 0.6.0

## Running commands

### Burn-in phase
Below we provide running commands for burn-in phase
```python
python train_cifar.py
```
### Accumulative poisoning attacks in online learning cases
Below we provide running commands for accumulative phase + poisoned trigger(controlled by `--use_advtrigger`) + online poisoned trigger (controlled by `--use_online_advtrigger`):
```python
python online_accu_train.py \
                  --batch_size 100 --epoch 100 --test_batch_size 500 --log_name log_test_online.txt\
                  --resume checkpoints_base_bn --use_bn --model_name epoch40.pth \
                  --mode 'eval' --onlinemode 'train' --lr 1e-1 --momentum 0.9 \
                  --beta 1. --only_reg --threshold 0.18 --use_advtrigger
```
### Accumulative poisoning attacks in federated learning cases
Below we provide running commands for accumulative phase (controlled by `--feder_lambda`, `--epoch`) + poisoned trigger (controlled by `--poisoned_trigger_step`):
```python
python feder_accu_train.py \
                  --batch_size 100 --epoch 1000 --test_batch_size 500 --log_name log_test_feder.txt\
                  --resume checkpoints_base_bn --use_bn --model_name epoch40.pth \
                  --mode 'train' --onlinemode 'train' --feder_lambda 8e-2 --lr 1e-1 --momentum 0.9 \
                  --poisoned_trigger_step 0.01 \
                  --clip_gradnorm --clipvalue 10
```
Here we also activate the gradient norm clipping operations by the FLAGs `--clip_gradnorm` and `--clipvalue`.
