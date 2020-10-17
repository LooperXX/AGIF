# AGIF: An Adaptive Graph-Interactive Framework for Joint MultipleIntent Detection and Slot Filling

This repository contains the PyTorch implementation of the paper: 

**AGIF: An Adaptive Graph-Interactive Framework for Joint MultipleIntent Detection and Slot Filling**. [Libo Qin](http://ir.hit.edu.cn/~lbqin/), [Xiao Xu](https://looperxx.github.io/), [Wanxiang Che](http://ir.hit.edu.cn/~car/chinese.htm), [Ting Liu](http://ir.hit.edu.cn/~liuting/). ***EMNLP 2020 Accept-Findings***. [[PDF(Arxiv)]](https://arxiv.org/pdf/2004.10087.pdf) [[PDF]](https://www.aclweb.org/anthology/)

If you use any source codes or the datasets included in this toolkit in your work, please cite the following paper. The bibtex are listed below:

<pre>
...
</pre>
![example](img/example.png)

In the following, we will guide you how to use this repository step by step.

## Architecture
![framework](img/framework.png)

## Results
![result_multi](img/result_multi.png)

![result_single](img/result_single.png)

> Tips: We find some repeated sentences in the `MixATIS` and `MixSNIPS` datasets so that we clean these two datasets and name them `MixATIS_clean` and `MixSNIPS_clean`. We rerun all the experiments and the results are as follows:

|                         |  MixATIS_clean  |             |              |               | MixSNIPS_clean  |             |              |               |
| :---------------------: | :-------: | :---------: | :----------: | :-----------: | :-------: | :---------: | :----------: | :-----------: |
|         **Model**       | Slot (F1) | Intent (F1) | Intent (Acc) | Overall (Acc) | Slot (F1) | Intent (F1) | Intent (Acc) | Overall (Acc) |
|     Attention BiRNN     |   86.4    |      -      |     74.6     |     39.1      |   89.4    |      -      |     95.4     |     59.5      |
|     Slot-gated Full     |   86.9    |      -      |     64.4     |     35.5      |   87.5    |      -      |     94.6     |     53.1      |
|    Slot-gated Intent    |   87.7    |      -      |     63.9     |     35.5      |   87.9    |      -      |     94.6     |     55.4      |
|        Bi-Model         |   83.9    |      -      |     70.3     |     34.4      |   90.7    |      -      |     95.6     |     63.4      |
|          SF-ID          |   87.4    |      -      |     66.2     |     34.9      |   90.6    |      -      |     95.0     |     59.9      |
|    Stack-Propagation    |   86.9    |      -      |   **76.3**   |     40.8      |   93.9    |      -      |   **96.5**   |     73.9      |
| Stack-Propagation-Multi |   86.9    |    83.0     |     71.1     |     38.9      |   93.9    |    98.4     |     96.1     |     73.4      |
|  Joint Multiple ID&SL   |   84.9    |    83.6     |     73.9     |     32.4      |   91.2    |    98.0     |     95.1     |     65.8      |
|          AGIF           | **88.0**  |  **84.6**   |     76.1     |   **44.1**    | **94.6**  |  **98.5**   |     96.1     |   **75.4**    |

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
python train.py -g -bs=16 -ne=100 -dd=./data/MixATIS -lod=./log/MixATIS -sd=./save/MixATIS -nh=4 -wed=32 -sed=128 -ied=64 -sdhd=64 -dghd=64 -ln=MixATIS.txt

# MixATIS_clean dataset
python train.py -g -bs=16 -ne=200 -dd=./data/MixATIS_clean -lod=./log/MixATIS_clean -sd=./save/MixATIS_clean -nh=4 -wed=32 -sed=128 -ied=128 -sdhd=128 -dghd=64 -ln=MixSNIPS_clean.txt 

# MixSNIPS dataset
python train.py -g -bs=64 -ne=50 -dd=./data/MixSNIPS -lod=./log/MixSNIPS -sd=./save/MixSNIPS -nh=8 -wed=32 -ied=64 -sdhd=64 -ln=MixSNIPS.txt

# MixSNIPS_clean dataset
python train.py -g -bs=64 -ne=100 -dd=./data/MixSNIPS_clean -lod=./log/MixSNIPS_clean -sd=./save/MixSNIPS_clean -nh=8 -wed=32 -ied=64 -sdhd=64 -ln=MixSNIPS_clean.txt

# ATIS dataset
python train.py -g -bs=16 -ne=300 -dd=./data/ATIS -lod=./log/ATIS -sd=./save/ATIS -nh=4 -wed=64 -ied=128 -sdhd=128 -ln=ATIS.txt

# SNIPS dataset
python train.py -g -bs=16 -ne=200 -dd=./data/SNIPS -lod=./log/SNIPS -sd=./save/SNIPS -nh=8 -wed=64 -ied=64 -sdhd=64 -ln=SNIPS.txt 
```

We also provide our reported model parameters in the `save/best` directory, you can run the following command to evaluate them and so on.

```SHELL
# MixATIS dataset
python train.py -g -bs=16 -ne=0 -dd=./data/MixATIS -lod=./log/MixATIS -sd=./save/best/MixATIS -ld=./save/best/MixATIS -nh=4 -wed=32 -sed=128 -ied=64 -sdhd=64 -dghd=64 -ln=MixATIS.txt

# MixATIS_clean dataset
python train.py -g -bs=16 -ne=0 -dd=./data/MixATIS_clean -lod=./log/MixATIS_clean -sd=./save/best/MixATIS_clean -ld=./save/best/MixATIS_clean -nh=4 -wed=32 -sed=128 -ied=128 -sdhd=128 -dghd=64 -ln=MixSNIPS_clean.txt 

# MixSNIPS dataset
python train.py -g -bs=64 -ne=0 -dd=./data/MixSNIPS -lod=./log/MixSNIPS -sd=./save/best/MixSNIPS -ld=./save/best/MixSNIPS -nh=8 -wed=32 -ied=64 -sdhd=64 -ln=MixSNIPS.txt

# MixSNIPS_clean dataset
python train.py -g -bs=64 -ne=0 -dd=./data/MixSNIPS_clean -lod=./log/MixSNIPS_clean -sd=./save/best/MixSNIPS_clean -ld=./save/best/MixSNIPS_clean -nh=8 -wed=32 -ied=64 -sdhd=64 -ln=MixSNIPS_clean.txt

# ATIS dataset
python train.py -g -bs=16 -ne=0 -dd=./data/ATIS -lod=./log/ATIS -sd=./save/best/ATIS -ld=./save/best/ATIS -nh=4 -wed=64 -ied=128 -sdhd=128 -ln=ATIS.txt

# SNIPS dataset
python train.py -g -bs=16 -ne=0 -dd=./data/SNIPS -lod=./log/SNIPS -sd=./save/best/SNIPS -ld=./save/best/SNIPS -nh=8 -wed=64 -ied=64 -sdhd=64 -ln=SNIPS.txt 
```

Due to some stochastic factors(*e.g*., GPU and environment), it maybe need to slightly tune the hyper-parameters using grid search to reproduce the results reported in our paper. All the hyper-parameters are in the `utils/config.py` and here are the suggested hyper-parameter settings:

-   Number of attention heads [4, 8]
-   Intent Embedding Dim [64, 128]
-   Word Embedding Dim [32, 64]
-   Slot Embedding Dim [32, 64, 128]
-   Decoder Gat Hidden Dim [16, 32, 64]
-   Batch size [16, 32, 64]
-   `Intent Embedding Dim` must equal to `Slot Decoder Hidden Dim`

> P.S. We just slightly tune the hyper-parameters.


If you have any question, please issue the project or [email](mailto:xxu@ir.hit.edu.cn) me and we will reply you soon.

## Acknowledgement

A Stack-Propagation Framework with Token-Level Intent Detection for Spoken Language Understanding. Libo Qin,Wanxiang Che, Yangming Li, Haoyang Wen and Ting Liu. *(EMNLP 2019)*. Long paper. [[pdf]](https://www.aclweb.org/anthology/D19-1214/) [[code]](https://github.com/LeePleased/StackPropagation-SLU)

>   We are highly grateful for the public code of Stack-Propagation!