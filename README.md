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


Also, when you get a bottom-up-features, you will not cover all of the classes, but 1600 classes, which you refer to the original github of paper.











## encoder : get 10 features (originally 36 features in paper)
In the original paper, 36 features were extracted and attention was paid to. However, I had a limit of gpu, so I shortened it to 10, but there was no big difference between 36 and performance. Intuitively, 36 objects are not needed to create a single caption, so I think it would be okay to use 10 objects.






## evaluation
I tried to evaluate with visual genome dataset, but due to the nature of the data, we did not proceed with evaluation such as bleu because it had annotation that fits the dense capture task rather than the normal one. If I have time later, I will try using coco dataset.




## tip for user
1. Due to the nature of the capture task, evaluation is difficult, so it is recommended to write a code first and overfitting a small amount of data to see if the model works properly. After confirming that it is overfitting, train on a lot of data and try to evaluate it like bleu for coco data.
In my case, I had a hard time because I couldn't overfitting.;;
(Overfitting is more difficult when the batch size is large.)


2. I had a lot of stress from gpu because the working environment was colab. I introduce [automatic_mixed_precision](https://arxiv.org/abs/1710.03740) that can efficiently use gpu.
In my case, the usage of gpu fell in half and I could increase the batch size from 16 to 36.




## reference

I got a lot of help from (https://github.com/poojahira/image-captioning-bottom-up-top-down)

and (https://github.com/hengyuan-hu/bottom-up-attention-vqa)



