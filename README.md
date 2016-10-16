Various kinds of deep residual networks for CIFAR.

# Requirements
- Python 3.5+
- TensorFlow 0.10+
- numpy
    
 ```

 ```

# Models & Commands
## original redisual network
```
--model resnet 
```
[He, Kaiming, et al. "Deep residual learning for image recognition." arXiv preprint arXiv:1512.03385 (2015).](https://arxiv.org/abs/1512.03385)

```
--model resnet --residual_type 0
```
Bottleneck design

## pre-activation residual network
```
--model resnet --unit_type 1
```
[He, Kaiming, et al. "Identity mappings in deep residual networks." arXiv preprint arXiv:1603.05027 (2016).](https://arxiv.org/abs/1603.05027)

## wide residual network
```
--model [any model] --wide_factor k
```
[Zagoruyko, Sergey, and Nikos Komodakis. "Wide Residual Networks." arXiv preprint arXiv:1605.07146 (2016).](https://arxiv.org/abs/1605.07146)

## RoR
```
--model ror 
```
[Zhang, Ke, et al. "Residual Networks of Residual Networks: Multilevel Residual Networks." arXiv preprint arXiv:1608.02908 (2016).](https://arxiv.org/pdf/1608.02908.pdf)

## DenseNet
```
--model densenet
```
[Huang, Gao, Zhuang Liu, and Kilian Q. Weinberger. "Densely Connected Convolutional Networks." arXiv preprint arXiv:1608.06993 (2016).](https://arxiv.org/abs/1608.06993)

## residual network with shared weights
```
--model resrnn
```
[Liao, Qianli, and Tomaso Poggio. "Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex." arXiv preprint arXiv:1604.03640 (2016).](https://arxiv.org/abs/1604.03640)

# Running Examples
```bash
python -m train --data_dir ~/work/vrn/data/ --outputs_dir ~/work/vrn/outputs/ --dataset cifar-10 --model resnet
python -m train --data_dir ~/work/vrn/data/ --outputs_dir ~/work/vrn/outputs/ --dataset cifar-10 --model ror
python -m train --data_dir ~/work/vrn/data/ --outputs_dir ~/work/vrn/outputs/ --dataset cifar-10 --model resnet --print_step 3 --summary_step 40 --checkpoint_step 300
```

# License
MIT
