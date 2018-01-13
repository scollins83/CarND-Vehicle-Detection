

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
[heatmap]: ./writeup_images/heatmap_vis_8.png
[heatmap_labels]: ./writeup_images/heatmap_vis_7.png
[video_example]: ./writeup_images/from_video.png
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
   
![alt text][heatmap_windows]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

After ultimately tuning the feature extraction, my best classifier got up to .9927 accuracy. 

In order to stabilize the images over frames in a video and to try to 
track multiple cars, I also implemented a Vehicle and VehicleTracker in `vehicle.py`. In that,
I track vehicles over a set history (which is set at 5 in `vehicle_detection.py` line 845).

To try to get the vehicles to split and not just have one box at a time, every time a box 
was added, if it varied from any of the other existing 'vehicle' records (which were truncated
to 4x the per vehicle history size). 

I also rewrote the sliding_windows function with slightly different functionality which worked
on a window, and it can be found in the 'find_cars' function in `vehicle_detection.py` lines 
425 - 524. 
  
Ultimately I searched on two scales using YCrCb with all 3-channel HOG features,
spatial features, and histograms of color in the feature vector for my best results, as specified above. 

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_run_7.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

On earlier versions of the classifier, I had to filter out a lot more false positives from my heatmap
because the accuracy of the classifier wasn't as good. Often lower confidence 'hits' had 
would show up as a lower heatmap 'hit' than higher confidence areas, so filtering them out 
using the `apply_threshold` function in line 527 and 535 in `vehicle_detection.py`. However,
once the classifier improved, I was able to set it much lower, but still worked well on a threshold
of 1 or 2 (4 was too high). From there, I created my bounding boxes for cars being detected 
using scipy image measurements labels (`vehicle_detection.py` line 602). 
### Here are six frames and their corresponding heatmaps:

![alt text][heatmap]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames, and note that carrying over the boxes over time impacted these as well:
![alt text][heatmap_labels]

### Here the resulting bounding boxes are drawn onto a frame from the video:
![alt text][video_example]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had a few problems with the pipeline recognizing cars in the opposite lanes being used. Perhaps something akin to lane
detection from previous projects would be even better than the current method. Other than that, the false positives
were still relatively low, occasionally triggering from signs instead of cars on other versions of the model tested.

I also had quite a bit of trouble getting the split with the vehicle detector in the VehicleTracker class to actually work right to track multiple cars at once. Allowing it to work on a single 'vehicle' worked very well, but I had trouble getting this to note multiple cars and draw multiple bounding boxes, except when I didn't limit the vehicle count, in which case my code was really inefficient. Perhaps my threshold is too low, but otherwise, moving forward, this would be the first problem I'd try to solve. 
