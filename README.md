# Bottom-Up-and-Top-Down-Attention-for-Image-Captioning-pytorch



## getting started
This repo is a pytorch implementation of [Bottom-up and Top-down Attention](https://arxiv.org/pdf/1707.07998v3.pdf) for Image Captioning.
I tried many variations while following what the paper said.

        first, I use Efficientdet, not Faster RCNN, as a model for obtaining bottom-up-features.
        
        Second, I use GRU instead of LSTM as a caption_model. This resulted in faster convergence.
        


## bottom-features : efficientdet
I used efficientdet, which recently has shown high performance in the field of image detection.
Because my environment was colab, there was a limit(gpu), so it was best to use efficientdet version 1.



## visual_genome_dataset: data prepare
Dealing with visual genome data was very difficult. I think it is because the data itself is made with various purposes. I spent a lot of time organizing these data into a dataframe for the capture task (although the api provided by the paper's original author existed, it just produced a data form that fits me, which is organized in the 'data' directory).

download images here : [image1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [image2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)


download  caption annotation here : [annotation](http://visualgenome.org/static/data/dataset/attributes.json.zip)


Also, when you get a bottom-up-features, you will not cover all of the classes, but 1600 classes, which you refer to the original github of paperd.


