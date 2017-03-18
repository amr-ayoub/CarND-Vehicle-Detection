

# Vehicle Detection Project (SDC ND Project 5)

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


### Project code is in the the IPython notebook [vehicle_detection.ipynb](vehicle_detection.ipynb)


## Data Exploration

I started by reading in all the vehicle and non-vehicle images dataset provided by Udacity which comes in two separate datasets: images containing vehicle and images not containing vehicles. The dataset contains 17,760 color RGB images 64×64 px each, with 8,792 samples labeled as containing vehicles and 8,968 samples labeled as non-vehicles.

Here is an example of one of each of the vehicle and non-vehicle classes:

#### Random sample labeled as containing vehicle:
![alt text](output_images/example_vehicle_data_images.jpg)


#### Random sample labeled as containing non-vehicle:
![alt text](output_images/example_non_vehicle_data_images.jpg)



## Feature extraction

I then explored different features: color spaces and different skimage.hog() parameters (orientations, pixels_per_cell and cells_per_block). I settled on a combination of HOG (Histogram of Oriented Gradients), spatial information and color channel histograms, all using YCbCr color space. Initially I used only the Y channel but I found that it was not enough so I used all 3 colour channels

#### As a feature vector I used a combination of:

### 1 - Spatial features, which are nothing else but a down sampled copy of the image patch to be checked itself (32x32 pixels).

#### Example of data image:
![alt text](output_images/image_example_2.jpg)

#### Example of data image applying Spatial binning
![alt text](output_images/image_example_2_spatial_binning.jpg)


### 2- Color histogram features using individual color channel histogram information (YCbCr color space), breaking it into 32 bins within (0, 256) range.

#### Example image
![alt text](output_images/image_example_2.jpg)

#### YCbCr color space histogram
![alt text](output_images/image_example_2_YCrCb_Histogram.jpg)


### 3- Histogram of oriented gradients (HOG) features, that capture the gradient structure of each image channel and work well under different lighting conditions.

I explored different parameters: `orientations`, `pixels_per_cell` and `cells_per_block`. eventually I settled on HOG with 10 orientations, 8 pixels per cell, 2 cells per block and 'YCrCb' color_space. The experiments went as training the classifier and checking the accurcy. 

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=10`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

#### Example of data image:
![alt text](output_images/image_example_2.jpg)

#### Y channel:
![alt text](output_images/image_example_2_Y_channel.jpg)

#### HOG on Y channel:
![alt text](output_images/image_example_2_Y_channel_HOG.jpg)

#### Cr channel:
![alt text](output_images/image_example_2_Cr_channel.jpg)

#### HOG on Cr channel:
![alt text](output_images/image_example_2_Cr_channel_HOG.jpg)


#### Cb channel:
![alt text](output_images/image_example_2_Cb_channel.jpg)

#### HOG on Cb channel:
![alt text](output_images/image_example_2_Cb_channel_HOG.jpg)


## Training a linear support vector machine classifier
A linear SVM offered the best compromise between speed and accuracy, outperforming nonlinear SVMs (rbf kernel). I trained a Linear SVC (sklearn implementation), used sklearn `train_test_split` to split the dataset into training and validation sets and used sklearn `StandardScaler` for feature scaling.


## Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

