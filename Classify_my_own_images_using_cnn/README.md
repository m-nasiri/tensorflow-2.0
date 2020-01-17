In this code we use tf.data api to read images and labels from folders. 
At the first step we extract MNIST dataset images and store all images in train and test 
folders then we prepare a list of train and test images and also train and test labels by
running extract_mnist_images.py 

After running the script we must have a folder structured like following in the main 
folder beside the script. train_images_list.txt contain path to images in train folder and 
test_images_list.txt contain path to images in test folder.

       --- dataset
              |
              |-- train
              |    |__ image_0.png
              |    |__ image_1.png
              |    |__ ...
              |    |__ image_N.png
              |
              |-- test
              |    |__ image_0.png
              |    |__ image_1.png
              |    |__ ...
              |    |__ image_N.png
              |
              |-- test_images_list.txt
              |-- test_labels_list.txt
              |-- train_images_list.txt
              |-- train_labels_list.txt

You can prepare your own dataset as our structure. 
dataset_utils.py contain functions to read images and labels for train and test portion of
dataset. These function uses tf.data api to read, prefetch and batch data for model.

mnist_tfdata_cnn.py is the main code to read images and labels, building a CNN model and feed 
train and test data to model for training and evaluation. 

We used Keras an high level api form tensorflow 2.0 to build our sequentioal model.
    
`$ python extract_mnist_images.py`        to extract MNIST dataset images to train and test folders.

`$ python mnist_tfdata_cnn.py`            to read images and labels and push to CNN and train the model.


