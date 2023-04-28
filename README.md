# 

### orientation/

By fine-tuning a pretrained DenseNet, I achieve 98.8% validation accuracy
on determining whether text is upright or upside down. 

Starting off with a pretrained DenseNet is inspired from the 
[TrOCR](https://arxiv.org/abs/2109.10282) and [DeiT](https://arxiv.org/abs/2012.12877) papers, which
also used an image classifier pretrained on ImageNet for text classification. 
Starting with an untrained model did better than random, but progress was much slower than
with the pretrained model and I had problems with overfitting. Strangely, with the pretrained model,
validation accuracy was higher than training accuracy. (Maybe dropout related?)

Data processing code is in orientation_nn.py

My previous attempts failed probably because of lack of parameters - the DenseNet is 100x more massive than my 
previous attempt. 
