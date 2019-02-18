# Residual networks

---

## Original ResNets 

We consider networks implementation available in [`torchvision`](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py).

["Deep Residual Learning for Image Recognition" Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](https://arxiv.org/pdf/1512.03385.pdf)

### Overview

A ResNet is composed of 5 parts that produce image features that can be further classified with a fully-connected layer:

```
--[Prep]--->[L1]->[L2]->[L3]->[L4]--->[Pool]->[FC]---
```

The first part is a preparation block composed of 
```
---[Conv]-[BN]-[ReLU]-[MaxPool]---

Conv = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
BN = BatchNorm2d(64)
MaxPool = MaxPool2d(kernel_size=3, stride=2, padding=1)
```
and helps to adapt the input data.

#### Residual blocks

Parts noted as `L1`, ..., `L4` are composed of a residual blocks. A residual block can be of type:

- basic block for shallow networks
- bottleneck block for deeper networks

and contains a "shortcut": 

- identity 
- projection

```
Li : ---[Projection/Identity]--[Identity]--[Identity]--...--[Identity]---
```

##### Basic block

1) Identity shortcut

```
        M->M                    M->M 
---.--[Conv3x3]-[BN]-[ReLU]--[Conv3x3]-[BN]--[+]--[ReLU]---
   |                                          | 
   |------------------------------------------|
```


2) Projection shortcut

```
        M->K                   K->K
---.--[Conv3x3]-[BN]-[ReLU]--[Conv3x3]-[BN]--[+]--[ReLU]---
   |   stride=2                               |
   |                                          | 
   |    M->K                                  | 
   |--[Conv1x1]-[BN]--------------------------|
       stride=2
```

##### Bottleneck block

1) Identity shortcut

```
       M -> M/4               M/4->M/4               M/4 -> M
---.--[Conv3x3]-[BN]-[ReLU]--[Conv3x3]-[BN]-[ReLU]--[Conv3x3]-[BN]--[+]--[ReLU]---
   |                                                                 | 
   |-----------------------------------------------------------------|
```


2) Projection shortcut

```
       M->K                    K->K                  K->K*4
---.--[Conv1x1]-[BN]-[ReLU]--[Conv3x3]-[BN]-[ReLU]--[Conv1x1]-[BN]--[+]--[ReLU]---
   |   stride=1 or 2                                                 |
   |                                                                 | 
   |   M->K*4                                                        | 
   |--[Conv1x1]-[BN]-------------------------------------------------|
       stride=1 or 2  
```


---

## Wide Residual Networks

We explore here the pytorch implementation from [szagoruyko/wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py).

### Overview

A Wide Residual Network is composed of 4 parts that produce image features that can be further classified with a fully-connected layer:
```
---[Conv3x3]-->[G1]-->[G2]-->[G3][BN][ReLU]--->[Pool]-[FC]---
```

A WRN model is configured by two parameters: depth, width (`WRN-d-w`).
Depth has to be `6 x n + 4`, where `n` defines the number blocks in the groups.
Width defines the width of residual blocks.

#### Pre-act residual block

A group `Gi` is composed of `n` blocks (where `n = (depth - 4) // 6`):
```
---[A1]--[B2]--...--[Bn]---
```

The first group `A1` can have a stride equals 1 or 2 in the convolutions. Other blocks `B1`, ..., `Bn`
have stride equal 1 

a) Identity shortcut
```
                   stride=1               stride=1
---.--[BN]-[ReLU]-[Conv3x3]--[BN]-[ReLU]-[Conv3x3]--[+]---
   |                                                 | 
   |-------------------------------------------------|

```
b) Projection shortcut
```
                   stride=2                stride=1
---[BN]-[ReLU]--.--[Conv3x3]--[BN]-[ReLU]-[Conv3x3]--[+]---
                |                                     | 
                |--[Conv1x1]--------------------------|
                   stride=2
```


["Wide Residual Networks" Sergey Zagoruyko, Nikos Komodakis](https://arxiv.org/pdf/1605.07146.pdf)