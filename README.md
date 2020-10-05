# AGIF: An Adaptive Graph-Interactive Framework for Joint MultipleIntent Detection and Slot Filling

This repository contains the PyTorch implementation of the paper: 

**AGIF: An Adaptive Graph-Interactive Framework for Joint MultipleIntent Detection and Slot Filling**. [Libo Qin](http://ir.hit.edu.cn/~lbqin/), [Xiao Xu](https://looperxx.github.io/), [Wanxiang Che](http://ir.hit.edu.cn/~car/chinese.htm), [Ting Liu](http://ir.hit.edu.cn/~liuting/). ***EMNLP 2020 Accept-Findings***. [[PDF]](https://www.aclweb.org/anthology/)

If you use any source codes or the datasets included in this toolkit in your work, please cite the following paper. The bibtex are listed below:

<pre>
...
</pre>
![example](img/example.png)

In the following, we will guide you how to use this repository step by step.

## Architecture
![framework](img/framework.png)

## Preparation

Our code is based on PyTorch 1.2 Required python packages:

-   numpy==1.18.1
-   tqdm==4.32.1
-   pytorch==1.2.0
-   python==3.7.3
-   cudatoolkit==9.2

We highly suggest you using [Anaconda](https://www.anaconda.com/) to manage your python environment.

## How to Run it

The script **train.py** acts as a main function to the project, you can run the experiments by the following commands.

```Shell
# MixATIS dataset
python train.py -g -bs=16 -ne=100 -dd=./data/MixATIS -lod=./log/MixATIS -sd=./save/MixATIS -nld=2 -nh=4 -wed=32 -sed=128 -ied=64 -sdhd=64 -dghd=64 -ln=MixATIS.txt

# MixSNIPS dataset
python train.py -g -bs=64 -ne=50 -dd=./data/MixSNIPS -lod=./log/MixSNIPS -sd=./save/MixSNIPS -nld=2 -nh=8 -wed=32 -ied=64 -sdhd=64 -ln=MixSNIPS.txt

# ATIS dataset
python train.py -g -bs=16 -ne=300 -dd=./data/ATIS -lod=./log/ATIS -sd=./save/ATIS -nld=2 -nh=4 -wed=64 -ied=128 -sdhd=128 -ln=ATIS.txt

# SNIPS dataset
python train.py -g -bs=16 -ne=200 -dd=./data/SNIPS -lod=./log/SNIPS -sd=./save/SNIPS -nld=2 -nh=8 -wed=64 -ied=64 -sdhd=64 -ln=SNIPS.txt 
```

We also provide our reported model parameters in the `save/best` directory, you can run the following command to evaluate them and so on.

```SHELL
# MixATIS dataset
python train.py -g -bs=16 -ne=0 -dd=./data/MixATIS -lod=./log/MixATIS -sd=./save/best/MixATIS -ld=./save/best/MixATIS -nld=2 -nh=4 -wed=32 -sed=128 -ied=64 -sdhd=64 -dghd=64 -ln=MixATIS.txt

# MixSNIPS dataset
python train.py -g -bs=64 -ne=0 -dd=./data/MixSNIPS -lod=./log/MixSNIPS -sd=./save/best/MixSNIPS -ld=./save/best/MixSNIPS -nld=2 -nh=8 -wed=32 -ied=64 -sdhd=64 -ln=MixSNIPS.txt

# ATIS dataset
python train.py -g -bs=16 -ne=0 -dd=./data/ATIS -lod=./log/ATIS -sd=./save/best/ATIS -ld=./save/best/ATIS -nld=2 -nh=4 -wed=64 -ied=128 -sdhd=128 -ln=ATIS.txt

# SNIPS dataset
python train.py -g -bs=16 -ne=0 -dd=./data/SNIPS -lod=./log/SNIPS -sd=./save/best/SNIPS -ld=./save/best/SNIPS -nld=2 -nh=8 -wed=64 -ied=64 -sdhd=64 -ln=SNIPS.txt 
```

Due to some stochastic factors(e.g., GPU and environment), it maybe need to slightly tune the hyper-parameters using grid search to reproduce the results reported in our paper. All the hyper-parameters are in the `utils/config.py` and here are the suggested hyper-parameter settings:

-   Batch size [16, 32, 64]
-   Number of attention heads [4, 8]
-   Intent Embedding Dim [64, 128]
-   Word Embedding Dim [32, 64, 128]

If you have any question, please issue the project or [email](mailto:xxu@ir.hit.edu.cn) me and we will reply you soon.

## Acknowledgement

A Stack-Propagation Framework with Token-Level Intent Detection for Spoken Language Understanding. Libo Qin,Wanxiang Che, Yangming Li, Haoyang Wen and Ting Liu. *(EMNLP 2019)*. Long paper. [[pdf](https://www.aclweb.org/anthology/D19-1214/)] [[code](https://github.com/LeePleased/StackPropagation-SLU)]

>   We are highly grateful for the public code of Stack-Propagation!