# Behavioral Cloning Project

[//]: # (Image References)

[image1]: ./images/image1.jpg "Grayscaling"
[image2]: ./images/image2.jpg "Recovery Image"
[image3]: ./images/image3.jpg "Recovery Image"
[image5]: ./images/image4.jpg "Normal Image"
[image6]: ./images/image5.jpeg "Flipped Image"

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* video.mp4 containing the video of the trained network

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

The model it's based on the NVIDIA autonomous driving car architecture
I apply strides of 2 with 5x5 filter for the first 3 convolutional layers instead of a subsampling (code lines 183 - 189), for other two convolutional layers I apply a 3x3 kernerl filter (code lines 192 - 195).

The model includes RELU activation function for each layer and the data is normalized in the model using a Keras lambda layer (code line 181).
After 5 layers of convolutions

### 2. Attempts to reduce overfitting in the model

To not lose execution time and prevent overifitting I introduce the early stopping function using "min" mode. The model contains also dropouts layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and increased number of traing data about steering.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA for self driving car architecture I thought this model might be appropriate because it's already used into a self driving cars. 

I started by implement strides of 2 instead of use subsamplig on convolutional layers, also I added a RELU activation function for each layer. After that I run the simulator in order to collect more data.

To combat the overfitting, I modified the model by adding some dropouts layers after each layer, after convolutional layers I applied 0.3 of probabilities and after dense layers 0.2 of probabilities.

Cropping the environment around the track makes the model much better.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I added some corrections training data to increase number of steering data in specific cases.

At the end of the process, the vehicle was not completly accurated so I decided to accept YUV colospace as input of the model. This make the simulator able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 173-210) consisted of a convolution neural network with the following layers and layer sizes:

 | Layer         | Parameters                         |
| ------------- | ---------------------------------- |
| Cropping2D    | 70 top, 25 bottom, 5 left, 5 right |
| Normalization | range -0.5 <-> 0.5                 |
| Convolution2D | 24x5x5, strides=(2, 2)             |
| Activation    | RELU                               |
| Dropout       | 0.3                                |
| Convolution2D | 36x5x5, strides=(2, 2)             |
| Activation    | RELU                               |
| Dropout       | 0.3                                |
| Convolution2D | 48x5x5, strides=(2, 2)             |
| Activation    | RELU                               |
| Dropout       | 0.3                                |
| Convolution2D | 64x3x3                             |
| Activation    | RELU                               |
| Dropout       | 0.3                                |
| Convolution2D | 64x3x3                             |
| Activation    | RELU                               |
| Flatten       |                                    |
| Dropout       | 0.2                                |
| Dense         | 1100                               |
| Activation    | RELU                               |
| Dropout       | 0.2                                |
| Dense         | 100                                |
| Activation    | RELU                               |
| Dropout       | 0.2                                |
| Dense         | 50                                 |
| Activation    | RELU                               |
| Dropout       | 0.2                                |
| Dense         | 10                                 |
| Activation    | RELU                               |
| Dropout       | 0.2                                |
| Dense         | 1                                  |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to the center. These images show what a recovery looks like starting from right lane:

![alt text][image2]
![alt text][image3]


To augment the data sat, I also flipped images and angles thinking that this would be helpful beacuse the track has more left turns than right. For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]


I preprocessed this data by change the colorspace from RGB to YUV.
I also add some layers as preprocess into the pipeline: crop 70 pixels on top, 25 on bottom and 5 pixels on left and right sides. After that I applied a normalization function to change values into -0.5 - 0.5 range.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was near 10 as evidenced by the early stopping I used an adam optimizer so that manually training the learning rate wasn't necessary.


## Dependencies of the project
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)