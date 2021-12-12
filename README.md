**Token**: Token-based Representation for Image Retrieval
========
PyTorch training code for **Token**-based Representation for Image Retrieval.
We propose a joint local feature learning and aggregation framework, obtaining **82.3 mAP** on ROxf with Medium evaluation protocols. Inference in 50 lines of PyTorch.

![Token](Figure/framework.png)

**What it is**. Given an image, **Token** first uses a CNN and a Local Feature Self-Attention (LFSA) module to extract local features $F_c$. Then, they are tokenized into $L$ visual tokens with spatial attention. Further, a refinement
block is introduced to enhance the obtained visual tokens with self-attention and cross-attention. Finally, **Token** concatenates all the
visual tokens to form a compact global representation $f_g$ and reduce its dimension. The aggreegated global feature is discriminative and efficient.

**About the code**. 
**Token** is very simple to implement and experiment with.
Training code follows this idea - it is not a library,
but simply a [train.py](train.py) importing model and criterion
definitions with standard training loops.

# mAP performance of the proposed model
We provide results of **Token**.
mAP is computed with Medium and Hard evaluation protocols.
model will come soon.

![Token](Figure/result.png)

# Requirements
- Python 3
- cuda 11.0
- [PyTorch](https://pytorch.org/get-started/locally/) tested on 1.8.0, torchvision 0.9.0
- numpy
- matplotlib


# Usage - Representation learning
There are no extra compiled components in **Token** and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via conda.
Install PyTorch 1.8.0 and torchvision 0.9.0:
```
conda install -c pytorch pytorch torchvision
```

## Data preparation
Before going further, please check out [Google landmarkv2 github](https://github.com/cvdfoundation/google-landmark). We use their training images. If you use this code in your research, please also cite their work!

Download and extract Google landmarkv2 train and val images with annotations from
[https://github.com/cvdfoundation/google-landmark](https://github.com/cvdfoundation/google-landmark).

Download [ROxf](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings) and [RPar](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings) datastes with [annotations](http://cmp.felk.cvut.cz/revisitop/).
We expect the directory structure to be the following:
```
/data/
  ├─ Google-landmark-v2 # train images
  │   ├─ train.csv
  │   ├─ train_clean.csv
  │   ├─ GLDv2-clean-train-split.pkl
  │   ├─ GLDv2-clean-val-split.pkl
  |   └─ train
  └─test # test images
      ├─ roxford5k
      |   ├─ jpg
      |   └─ gnd_roxford5k.pkl
      └─ rparis6k
          ├─ jpg
          └─ gnd_rparis6k.pkl
```

## Training
To train **Token** on a single node with 4 gpus for 30 epochs run:
```
sh experiment.sh
```
A single epoch takes 2.5 hours, so 30 epoch training
takes around 3 days on a single machine with 4 3090Ti cards.

We train **Token** with SGD setting learning rate to 0.01.
The refinement block is trained with dropout of 0.1, and linearly decaying scheduler is adopted to gradually decay the learning rate to 0 when the desired number of steps is reached.

## Evaluation
To evaluate on Roxf and Rparis with a single GPU run:
```
python test.py
```
and get results as below 
```
>> Test Dataset: roxford5k *** local aggregation >>
>> mAP Medium: 82.28, Hard: 66.57

>> Test Dataset: rparis6k *** local aggregation >>
>> mAP Medium: 89.34, Hard: 78.56
```
We found that there is a change in performance when the test environment is different, for example, when the environment is GeForce RTX 2080Ti with cuda 10.2, pytorch 1.7.1 and torchvision 0.8.2, the test performance is
```
>> Test Dataset: roxford5k *** local aggregation >>
>> mAP Medium: 81.36, Hard: 62.09

>> Test Dataset: rparis6k *** local aggregation >>
>> mAP Medium: 90.19, Hard: 80.16
```

# Qualitative examples
Selected qualitative examples of different methods. Top-11 results are shown in the figure. The image with green denotes the true positives and the red bounding boxes are false positives.

![Token](Figure/examples.png)
