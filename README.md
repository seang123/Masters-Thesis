# Masters Thesis
## Think and Tell - image captioning from brain data

---

Links

- [Im2txt](https://github.com/HughKu/Im2txt)
- [NeuralTalk2](https://github.com/karpathy/neuraltalk2)
- [Another implementation](https://github.com/jazzsaxmafia/show_and_tell.tensorflow)

- [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo#resnets-deep-residual-networks-from-msra-at-imagenet-and-coco-2015)
  - Provides ResNet pre-trained networks that "won the 1st places in: ImageNet classification, ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation", which could be good for the CNN layer. 


---


# Plan

1. Implement the CNN-RNN architecture from the "Show and Tell" paper, download the pre-trained weights so save time if possible.
 - test it on the mscoc dataset
 - convert it to work with the fMRI data
2. Add an attention mechanism
3. Use a transformer network
