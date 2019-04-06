# CNN_PyTorch

__Looking for training using Keras? switch to branch [master](https://github.com/skyduy/CNN_keras/tree/master).__


## Introduction
[Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) using [PyTorch](https://pytorch.org/) for [CAPTCHA](https://en.wikipedia.org/wiki/CAPTCHA) recognition.

Recognition for this type of CAPTCHA (generated from repo [CAPTCHA_generator](https://github.com/skyduy/CAPTCHA_generator)):

![1](https://github.com/skyduy/CNN_keras/blob/pytorch/samples/AHVE_3fe8.jpg)
![2](https://github.com/skyduy/CNN_keras/blob/pytorch/samples/VGDU_200c.jpg)
![3](https://github.com/skyduy/CNN_keras/blob/pytorch/samples/YUNX_d03a.jpg)

In branch [master](https://github.com/skyduy/CNN_keras/tree/master), 
we must firstly split the whole image into single letters, and then use letters 
as training data. So, for different CAPTCHA, we have to find a specific
method to split it. The benefits is that while splitting the picture, 
we also expand the training data samples and reduce the problem size.

Method used in this branch is that we see the whole picture as one training sample 
with four labels, so it becomes a multilabel-classification problems.
See model graph below for detail info.


## Achievement

Using 4.8k training set:

![4.8k](https://github.com/skyduy/CNN_keras/blob/pytorch/achievements/4.8k.png)

Using 10k training set:

![10k](https://github.com/skyduy/CNN_keras/blob/pytorch/achievements/10k.png)

Using 20k training set:

![20k](https://github.com/skyduy/CNN_keras/blob/pytorch/achievements/20k.png)


## Model Graph 

```Sorry for ugly painting, waiting for your pull requests to make it better :)```


![model](https://github.com/skyduy/CNN_keras/blob/pytorch/achievements/model.png)


## Others

- If you have any questions, feel free to post a issue.

