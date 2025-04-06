<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                Enhancing Fast Adversarial Training with Momentum-Driven Initialization and Max Norm Regularization for Robust Deep Learning Models </h1>
<p align='left' style="text-align:left;font-size:1.2em;">
</p>

## Introduction

<p align="center">
In Fast Adversarial Training (FAT) improve the adversarial example quality with (a) Momentum-driven initialization and (b) Max-norm regularization to enhance robustness.
</p>


> In this work, we conduct an extensive evaluation of the Momentum-Initialized Max-Norm Regularization (MIMR) method in terms of adversarial robustness and computational efficiency. We revisit the effectiveness of MIMR in mitigating Catastrophic Overfitting by leveraging momentum-driven initialization and applying max-norm regularization to stabilize model training. Additionally, we explore the impact of incorporating momentum in the initialization process and regularization, which improves the model’s resilience against adversarial perturbations while maintaining training stability. We further investigate the role of advanced weight normalization strategies and propose a novel integration of MIMR with these methods for enhanced model robustness. By combining these techniques, we introduce the MIMR-based adversarial training method, which efficiently strengthens the model's resistance to adversarial attacks. Experimental results on several benchmark datasets demonstrate that the MIMR approach outperforms state-of-the-art adversarial training methods in terms of both adversarial accuracy and training efficiency, establishing it as a promising method for improving adversarial robustness

## Train for MIMR
```python3 train_MIMR_CIFAR10.py  --out_dir ./output/ --data-dir cifar-10
python3 train_MIMR_CIFAR100.py  --out_dir ./output/ --data-dir cifar-100
python3 train_Tiny_Imagenet.py --out_dir ./output/ --data-dir tiny-imagenet-200
```
## Test
```python3.6 test_cifar10.py --model_path model.pth --out_dir ./output/ --data-dir cifar-10
python3.6 test_cifar100.py --model_path model.pth --out_dir ./output/ --data-dir cifar-100
python3.6 test_Tiny_imageNet.py --model_path model.pth --out_dir ./output/ --data-dir tiny-imagenet-200
```

## Trained Models
> The Trained models will be available soon
