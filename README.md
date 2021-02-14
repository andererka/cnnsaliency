# nnsaliency
This repositiory is adding a saliency loader to the data loaders from nnvision/datasets.
The saliency loader is similar to the monkey static loader, but allows the input of a second folder where the saliency maps should be stored. The loader can as well take an argument to compute the spatial gradients of the saliency maps.
The output of the saliency loader is, depending on the arguments, a 1-4 channels batch loader. 

Further, a new model, called se_core_saliency_shifted_readout was added that consists of a CNN core and a small shifter CNN network that is calculating shifts based on the input of the images (or saliency maps, depending on the argument) for each neuron individually. To be more specific, the idea here was to design a model that puts images through the CNN core and the saliency maps and the gradients are used to compute the neuron specific shift. At the end, the core and the neuron specific shift are used for the final feature map readout (Multi Gaussian).

Some notebooks in this repository are about several different approaches to visualize what neurons in V4 represent and how to calculate image-specific gradient receptive fields of the neurons. Refer to these notebooks for more details: analysis_of_RF_V4.ipynb, Exploration_of_PCA_of_gradientRFs.ipynb, PCA_RF_images.ipynb, kPCA_notebook.ipynb
