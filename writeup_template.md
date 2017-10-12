# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./writeupimg/origin.png "Origin"
[image3]: ./writeupimg/hsv.png "hsv filter"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 7 steps. 

#### Step 1: Color Filtering

Because most of the lane line in the world will only have two color: yellow and white. So, I defined a function called "hsv_filtering", which will filter out all the colors except yellow and white.

```python
def hsv_filter(img, sensitive=15):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0,0,255-sensitive])
    upper_white = np.array([180,sensitive,255])
    lower_yellow = np.array([30-sensitive,100,100])
    upper_yellow = np.array([30+sensitive,255,255])
    
    mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(mask_white,mask_yellow)
    res = cv2.bitwise_and(img,img, mask=mask)

    return res
```

detail explanation:
	First convert the image from RGB color space to HSV color space, this will make it easy to filter out colors. Then, set the value threshold(lower and upper threshold) for white and yellow color. Later on, I create white color mask and yellow color mask and combining them using "bitwise_or". Finally, apply the mask to the image and return the image.

image demo:
![alt text][image3]

#### Step 2: Greyscale Transform

In order to perform the Canny edge detection, we need to transform the image from RGB color space to greyscale, which only contain one color channel.

```python
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
```



#### Step 3: Gaussian Filtering
#### Step 4: Canny Edge Detection
#### Step 5: Adding Region of Interest
#### Step 6: Hough Transform
#### Step 7: Drawing Lane Boundary Line - draw_lines() function

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
