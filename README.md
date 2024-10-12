
#  GF-GAN

The code of "Guided Fusion of Infrared and Visible Images using Gradient-Based Attentive Generative Adversarial Networks"
## Network Architecture
![image](https://github.com/catjsjsj/GF-GAN/tree/master/Network_structure/GF-GAN.jpg)
GF-GAN overall structure.
## To Train

Run "**CUDA_VISIBLE_DEVICES=0 python train.py**" to train your model.
The training data are used [MSRS](https://github.com/Linfeng-Tang/MSRS "MSRS"): Multi-Spectral Road Scenarios for Practical Infrared and Visible Image Fusion. (L. Tang, J. Yuan, and J. Ma, “Image fusion in the loop of high-level vision tasks: A semanticaware real-time infrared and visible image fusion network,” Information Fusion 82, 28–42 (2022)). For convenient training, users can download the training dataset from [here](https://pan.baidu.com/s/1xueuKYvYp7uPObzvywdgyA), in which the extraction code is: **bvfl**.

## To Test

Run "**CUDA_VISIBLE_DEVICES=0 python test.py**" to test the model.


## Recommended Environment

 - [ ] torch  1.7.1
 - [ ] torchvision 0.8.2
 - [ ] numpy 1.19.2
 - [ ] pillow  8.0.1



# Visualizing the Attention Map
![](https://github.com/catjsjsj/GF-GAN/blob/main/GF-GAN_model/tzt/tsetval_pred_temp/1.jpg) 
![](https://github.com/catjsjsj/GF-GAN/blob/main/GF-GAN_model/tzt/tsetval_pred_temp/4.jpg) 
