```english
---
layout: post
title:  "Does saliency play a role in V4 neurons of monkeys?"
date:   2021-01-25
description: "Investigating V4 neurons and visualizing to what patterns they react most"
tags: [V4, saliency, neural prediction]
comments: false
share: true
authors: Katharina Anderer
typora-copy-images-to: ../media/katharina/
typora-root-url: ../
---
```

# Summary

It is proposed f.e in [1] that the the visual area V4 has an important role in selective attentional processes, that is the enhanced processing of certain stimuli. These selective processes are important to guide our attention to stimuli that are most important for surviving or dealing with everyday tasks. Stimuli that catch our attention most and also fastest are called salient stimuli. 

In my lab rotation, I used so called saliency maps that are created by a deep neural network (DeepGaze by Matthias Kümmerer et al.) to investigate whether or not saliency is processed in V4. In short, DeepGaze is trained with eye-tracker data and can predict saliency maps for new images where in the picture salient density points most probably are. 

In order to investigate if saliency might play a role in V4 neurons, I tried the following different approaches:



<u>Part 1:</u>

* Feeding in the saliency maps as an additional channel to a CNN network (trying different readouts and variations here)
* Writing a new readout (called **SaliencyShifter)** that takes the images through the core and the saliency maps to a remapper network that calculates possible shifts of each neuron individually. I will go into more detail below and also direct to the permalinks of the code.

<u>Part 2:</u>

* The last approach was more about visualizing to what features in the input image neurons in V4 reacted most (see therefore the notebooks  RF_PCA.ipynb und Exploration_of_PCA_of_gradientRFs.ipynb) The idea behind this was to see if RFs of the neurons where shifting  towards salient objects. I could not observe shifts, but some results indicate that the RFs might be larger than expected and  neurons seem to react to shapes and textures. 

# Part 1: Saliency maps as inputs to CNN network

The background for this approach is the prediction of activation patterns of neurons in V4 neurons during visual processing. CNNs are so far the best architectures for doing that. The idea was now to use an existing CNN network, but adding the possibility to the data loader to take into account saliency maps and feed them to the network as well as a separate input channel.  The V4 area is already an intermediate processing layer in the hierarchy of visual processing and Roe, Chelazzi, Connor & collegues (2012) propose that V4 is involved in feature extraction and attentional processes. 

### How to calculate saliency maps?

In order to derive the information which parts in an image are probably the most salient ones, we could make use of the package Deep Gaze, written by Matthias Kümmerer, Lucas Theis and Matthias Bethge (2014). Deep Gaze uses a pretrained network to generate a high-dimensional feature space and then combines this network with the data on fixations of the MIT-1003 (a dataset for training and testing saliency in humans). 

### Difficulties and possible error sources with saliency maps

* Input size of the network seems to matter. It is trained for 35 pixel per degree of visual angle
* It can make a large difference if we zoom in the picture and then calculate the saliency maps or the other way round. So this should be always be kept in mind when for example cropping is applied to the images
* A center bias is added to the network in order to account for the tendency to focus more on the center of an image.  If a center bias is appropriate is probably a question of the context. The algorithm of DeepGaze is constructed such that it can predict with a high accuracy the areas of highest gaze density. In our case tough, it is not so obvious if a center bias is contributing to the task to give the network information about salient points in an image.
* The DeepGaze algorithm blurs the salient density points - it's a trade-off of loosing information or missing neighborhood parts of interest and to make the area of attention a bit 'softer'. 



### Further extension with spatial gradients

In order to give the network also the information in which direction the salient point are, I computed the spatial gradients with regard to the x and y axis. For this, I first applied a Gaussian filter and then the Sobel gradients with respect to x and y separately. 

The data loader, called **saliency loader**, that I adapted from the monkey static loader, takes several arguments as inputs as should the gradients be included, should the saliency maps be included or only one of them. For more information on how I implemented this, you can refer to this permalink:

https://github.com/andererka/nnsaliency/blob/589fc29ec421522f1901c0ade4fbdbf7a345fb29/nnsaliency/datasets/saliency_loaders.py#L25

Here is an example of how saliency maps and gradients could look like:

![img](/home/kathi/Dokumente/Sinz_praktikum/Blog_post/media/x8lRYGgsnU0DrO5hhWkp9tLhISMp01d_E9dZhxLkWlvQf2Hk14_cGZq35UHJmG0JHAHUjgRgOjcgEdqALocQgaMaUaCj7lx939ddNHeQXsr1HlQbBgrtgKA-jR_ucPrS1YrJ)

### Testing different CNN architectures and readouts

I tested this approach with the following architectures/ readouts:

* CNN with a Gaussian readout (nnvision.models.se_core_full_gauss_readout)

* CNN with a multiple attention 2D readout: https://github.com/fabiansinz/neuralpredictors/blob/b510273fa3302db06549dfa943b3c03b46008884/neuralpredictors/layers/readouts.py#L1677

* untrained ResNet (ptrmodels.task_core_gauss_readout)

* **Saliency Shifter Model:** The images are put through the core and the saliency maps and the gradients are used to compute the neuron specific shift. At the end, the core and the neuron specific shift are used for the final feature map readout (Multi Gaussian). For more information, you can refer to these permalinks of the code:

  https://github.com/andererka/nnvision/blob/e3046e4227fc7207f44d1ff106d90a1f3f008067/nnvision/models/readouts.py#L215 (readouts.py), 

  https://github.com/andererka/nnvision/blob/e3046e4227fc7207f44d1ff106d90a1f3f008067/nnvision/models/readouts.py#L314 (readouts.py),

  https://github.com/andererka/nnvision/blob/e3046e4227fc7207f44d1ff106d90a1f3f008067/nnvision/models/models.py#L130 (models.py)

  Here is a schema figure of this:

<img src="/home/kathi/Dokumente/Sinz_praktikum/Blog_post/media/image-20210202211023337.png" alt="image-20210202211023337" style="zoom:50%;" />

### Results of first approach

#### Training curves

The plot to the left is showing training curves (validation correlation is plotted on the y-axis) for CNN and ResNet architecturs that get only the images as input. The plot to the right show the training curves where additionally the saliency maps are fed into the networks. From eye-balling, one can not see differences between only images & images + saliency. There is only the difference that the ResNet architecture seems to learn a lot faster and stops earlier (after around 30 epochs compared to ~70 epochs for the CNN with Gaussian readout). 

<img src="/home/kathi/Dokumente/Sinz_praktikum/Blog_post/media/training_curves" alt="img" style="zoom:67%;" />

A similar comparison was done for the Saliency Shifter model (orange curve shows the training curve where no information is used for the remapper) :

The plot below shows a comparison of three different inputs for the remapping part:

1. Orange: tensor of ones for the remapper 
2. Blue: Saliency and gradients for the remapper
3. Green: Images for the remapper (channel is copied, so that we always have three channels for the remapper input). 

The last model is performing the best (both with respect to validation error and testing error). It seems that the information of only the image is so far the most informative for calculating shifts

![image-20210202214239049](/home/kathi/.config/Typora/typora-user-images/image-20210202214239049.png)

#### Validation and test correlations

Using a larger input area of the image, and therefore a smaller cropping on each side of the image, was almost always leading the an increase in performance, indicating that the receptive fields of the neurons were quite large or what could also be true, that the receptive fields were shifted sometimes. 

When I compared the some cropping sizes, the validation correlations and test correlations of the three different scenarios - Only Images, Images + Saliency, All (Images + Saliency + Gradients) - were quite similar, so that we could not conclude that the saliency maps were contributing any information for the prediction of neural activation. 

<img src="/home/kathi/Dokumente/Sinz_praktikum/Blog_post/media/cnn_test_corr" alt="img" style="zoom:80%;" />

The same was the case for the other models:

<img src="/home/kathi/Dokumente/Sinz_praktikum/Blog_post/media/attention_test_corr" alt="img" style="zoom: 80%;" /><img src="/home/kathi/Dokumente/Sinz_praktikum/Blog_post/media/reampper_test_corr" alt="img" style="zoom:;" />



## Discussion

blabla



# Part 2: Visualization of what neurons find interesting





# References

Roe, A. W., Chelazzi, L., Connor, C. E., Conway, B. R., Fujita, I.,  Gallant, J. L., ... & Vanduffel, W. (2012). Toward a unified theory  of visual area V4. *Neuron*, *74*(1), 12-29.

Kümmerer, M., Theis, L., & Bethge, M. (2014). Deep gaze i: Boosting saliency prediction with feature maps trained on imagenet. arXiv preprint arXiv:1411.1045.

