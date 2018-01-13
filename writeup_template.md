

# **Vehicle Detection Project**
### Sara Collins

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[plain_car_notcar]: ./writeup_images/plain_car_and_notcar.png
[HOG_car_notcar]: ./writeup_images/HOG_car_and_notcar.png
[windows]: ./writeup_images/window_vis.png
[heatmap_windows]: ./writeup_images/heatmap_vis.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

The current file comprises the writeup.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 75 through 102 of `vehicle_detection.py`.  

This function is applied to all the `vehicle` and `non-vehicle` images by way of the `extract_features` function beginning in line 136.  
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][plain_car_notcar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=10`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][HOG_car_notcar]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and actually used scoring of the 
Support Vector Machine classifier to optimize scoring for this project, and 
the classifier will be described in the next section.  
I also saved out the visualizations to do a 'sanity' check on the classifier score. 
That way, I could look and see if a car was really visible or not in a discernable 
fashion from the HOG image, as it was above.  
    
The parameters for HOG that I landed on were `YCrCb` colorspace, 
`orientations=10`, `pixels_per_cell=(8, 8)`, and `cells_per_block=(2, 2)`. 
Although I only show the visualization above in one channel, using call 
three channels proved to be by far the best for the classifier during my
parameter trials. 

I did over 40 runs with various colorspaces and other parameters.
Some of the 'highlights' of those runs are listed here:  
  
| Color Space | Orientations | Pix Per Cell | Cell Per Block | Image Height/Width | Histogram Bins | Feature Vector Length | Score   |
|:-----------:|:------------:|:------------:|:--------------:|:------------------:|:--------------:|:---------------------:|:-------:|
| **YCrCb**       | **10**           | **8**            | **2**              | **32x32**              | **12**             | **8968**                  | **0.9927**  |
| RGB         | 10           | 12           | 4              | 32x32              | 32             | 5088                  | 0.9682  |
| RGB         | 10           | 12           | 2              | 30x30              | 30             | 4710                  | 0.9724  |
| RGB         | 8            | 14           | 2              | 32x32              | 32             | 4032                  | 0.9645  |
| YCrCb       | 8            | 14           | 2              | 30x30              | 16             | 3612                  | 0.9716  |
| YCrCb       | 9            | 8            | 2              | 32x32              | 16             | (Not recorded)        | 0.9885  |
            
  
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the parameters noted in bold on the first row in the table above. 
I initially took a sample of 1000, but for the final dataset, I used all of the data available 
in the training sets. 

I also used scikit-learn's GridSearchCV to find the optimal value of the C parameter, which ended up being 5. I added this 
as a default to my 'train_svc' function in lines 728 through 748 in `vehicle_detection.py`

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I initially incorporated a sliding window search function, which is still contained 
in `vehicle_detection.py` in lines 208 through 258. When I was initially using this particular 
function, I had initially had an issue with the extract_features function and needed to tune 
my classifier. However, this function specifies a shape and amount of overlap to use in the windows.  
  
Here is an image from relatively early in my development process. From the output of this particular image,
it's clear to see that the main part of the roadway being searched wasn't covering quite the right area,
so I adjusted it so that the y axis starts at pixel 320 to hit the roadway and stops at 1280 to get as low
as possible in the image but still avoid searching the hood of the car. 
  
![alt text][windows]  
  
 Here is the same image, applying a heatmap for 'hot' areas where cars were initially found (at this point, the classifier had an error, but it was fixed later).  
   


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][heatmap_windows]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
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

