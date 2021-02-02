# nnsaliency
In this project, I tried several approaches to investigate if saliency plays a role in V4 vision area of macaque monkeys. 

The first approach was to adapt code that predicted the activations of neurons, dependent on inout images, in order to also feed the network with saliency maps in addition to the normal images.Therefore, the input of the training network was now containing an image channel, plus a saliency maps channel (and sometimes even channels for the spatial gradients of the saliency maps). 
I also used a remapper network that was calculating shifts of the RFs of the neurons. The results of this approach didn't reveal any hint that saliency was improving neural prediction.

The second approach was more about visualizing to what features of the input image neurons in V4 reacted most (see therefore the notebooks RFs_PCA und kPCA..)
The idea behind this was to see if RFs of the neurons where shifting towards salient objects. Shifts were not obvious, but some results indicate that the RFs might be quite large - larger than expected and neurons seem to react to shapes and textures. 
