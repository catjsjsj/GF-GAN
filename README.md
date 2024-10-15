
#  GF-GAN

The code of "Guided Fusion of Infrared and Visible Images using Gradient-Based Attentive Generative Adversarial Networks". Our paper has been submitted to a journal and the **DOI** of the paper will be updated after our preprint is approved.
## Network Architecture
![image](https://github.com/catjsjsj/GF-GAN/blob/master/Network_structure/GF-GAN.jpg)
GF-GAN overall structure.
## Novel gradient attention mechanism
![image](https://github.com/catjsjsj/GF-GAN/blob/master/Network_structure/Gradient%20attention.jpg)
Detailed structure of the gradient attention mechanism.
## To Train

Run "**CUDA_VISIBLE_DEVICES=0 python train.py**" to train your model.
The training data are used [MSRS](https://github.com/Linfeng-Tang/MSRS "MSRS"): Multi-Spectral Road Scenarios for Practical Infrared and Visible Image Fusion. (L. Tang, J. Yuan, and J. Ma, “Image fusion in the loop of high-level vision tasks: A semanticaware real-time infrared and visible image fusion network,” Information Fusion 82, 28–42 (2022)). For convenient training, users can download the training dataset from [here](https://pan.baidu.com/s/1xueuKYvYp7uPObzvywdgyA), in which the extraction code is: **bvfl**.

## To Test

Run "**CUDA_VISIBLE_DEVICES=0 python test.py**" to test the model. If you want to reproduce the results of our paper, you can download the TNO dataset and the LLVIP dataset and test them with our pre-trained weights. The TNO dataset can be downloaded [here](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029), and the LLVIP dataset can be downloaded [here](https://bupt-ai-cz.github.io/LLVIP/).


## Recommended Environment

 - [ ] torch  1.7.1
 - [ ] torchvision 0.8.2
 - [ ] numpy 1.19.2
 - [ ] pillow  8.0.1



# Fusion Example
<img src="https://github.com/catjsjsj/GF-GAN/blob/master/Fusion_results/TNO41_9.png" width="210px"><img src="https://github.com/catjsjsj/GF-GAN/blob/master/Fusion_results/TNO41_11.png" width="210px">
<img src="https://github.com/catjsjsj/GF-GAN/blob/master/Fusion_results/TNO41_12.png" width="210px"><img src="https://github.com/catjsjsj/GF-GAN/blob/master/Fusion_results/TNO41_13.png" width="210px">

Qualitative comparison of GF-GAN from the TNO dataset.

