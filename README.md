## :fire:VGG16
This is a VGG16 model with CIFAR10 dataset training.

## :collision:Illustration Training
my_vgg_train.py is a train code for vgg16.

my_vgg_model.py is the vgg16 model.

vgg_test_CIFAR10 is a process to test CIFAR10 with pre-trained vgg16 parameters.

If you run the my_vgg_train code with the epoch of 19, you will get the trained parameters of vgg16, and you can use the parameters to test new dataset.Of course, i use CIFAR10 for classification test.

The Training accuracy can reach 96%.

## :boom:Warning
However, high training accuracy just means the network design is good , which does not mean better robust for new data.
Because of too small dataset(CIFAR10 is a small dataset), the model just fits the training data, but cannot fits new data.

## :tada:Have fun!
