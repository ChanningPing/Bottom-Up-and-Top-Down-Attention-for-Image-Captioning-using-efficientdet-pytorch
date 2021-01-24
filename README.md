# Bottom-Up-and-Top-Down-Attention-for-Image-Captioning-pytorch



## getting started
This repo is a pytorch implementation of Bottom-up and Top-down Attention for Image Captioning.
I tried many variations while following what the paper said.

        first, I use Efficientdet, not Faster RCNN, as a model for obtaining bottom-up-features.
        
        Second, I use GRU instead of LSTM as a caption_model. This resulted in faster convergence.
        


## bottom-features : efficientdet
I used efficientdet, which recently has shown high performance in the field of image detection.
Because my environment was colab, there was a limit(gpu), so it was best to use efficientdet version 1.
