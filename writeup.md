# **Finding Lane Lines on the Road** 

## Arthor: Leon

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./writeupimg/origin.png "Origin"
[image3]: ./writeupimg/hsv.png "hsv filter"
[image4]: ./writeupimg/gray.png "grayscale"
[image5]: ./writeupimg/gaussian.png "gaussian"
[image6]: ./writeupimg/canny.png "canny"
[image7]: ./writeupimg/region.png "region"
[image8]: ./writeupimg/hough.png "hough"
[image9]: ./writeupimg/final.png "final"

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

![alt text][image3]

#### Step 2: Greyscale Transform

In order to perform the Canny edge detection, we need to transform the image from RGB color space to greyscale, which only contain one color channel.

```python
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
```

![alt text][image4]

#### Step 3: Gaussian Smoothing

Gaussian smoothing is a way of suppressing noise and spurious gradients. It is a type of image-blurring filter that uses a Gaussian function. Mathematic approach can be found [here](https://en.wikipedia.org/wiki/Gaussian_blur)

```python
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```

I choose kernel_size = 5 for the parameter.

![alt text][image5]

#### Step 4: Canny Edge Detection

Canny edge detection is a way to detect edges in image. Lane lines are edges in the image, using this method, it can be easily detect.

```python
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)
```

I choose low_threshold = 50, high_threshold = 150 for canny function.

![alt text][image6]

#### Step 5: Adding Region of Interest

There are some other objects detected around the periphery that aren't lane lines. We want to get rid of those noise. A good thing is lane lines are usually on the bottom area of the pictures/videos. So we can add a region of interest to focus on only "lane line" region.

```python
imshape = image.shape
#vertices = np.array([[(100,500),(450, 330), (510, 330), (900,500)]], dtype=np.int32)
vertices = np.array([[(imshape[1]*0.15,imshape[0]),(imshape[1]*0.47, imshape[0]*0.61), (imshape[1]*0.54, imshape[0]*0.61), (imshape[1]*0.95,imshape[0])]], dtype=np.int32)

def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```

detail explanation:
	The key point of this function is how to define vertices. First, we should understand that polygon is better than triangle, because it will fit the lane line region much better. Then we need to figure out the vertex of polygon. In the very beginning, I hard code the numbers on it(see number in the comment line). But it actually does not work for the challenge video. Why? Because frames will be in different size. So, we need to define percentage instead of pixel value. The four vertex which I define is (15% of x,100% of y),(47% of x, 61% of y), (54% of x, 61% of y), (95% of x,100% of y). All these percentage is calculated from my hard code pixel value.

![alt text][image7]

#### Step 6: Hough Transform

Hough transform is used to identify of lines in the image. By transform the points into hough space, we can found which points are belong to a line.

```python
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
```

The key part of this step is to tune the hough parameters(calibration). I use rho = 2, theta = np.pi/180, threshold = 15, min_line_len = 5, max_line_gap = 1. Here min_line_len and max_line_gap are more strick compared with the quiz. Why? Because I've applied hsv filter in the first step, it will filter out more noise. So, here, strict values will make result more accurate.

![alt text][image8]

#### Step 7: Drawing Lane Boundary Line - draw_lines() function

This is the last main step of my pipeline. We already have the edges from hough transform, now the task is to extrapolate the line segments to map out the full extent of the lane. Detailed exlanation see the section below code.

```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    
    threshold = 0.4
    threshold_max = 100
    mask_top = int(img.shape[0]*0.61)
    queue_size = 10
    x_left = []
    y_left = []
    x_right = []
    y_right = []
    y_min = img.shape[0]
    y_max = mask_top
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if(abs(slope) > threshold and abs(slope) < threshold_max):
                if(slope < 0):
                    x_left.extend((x1,x2))
                    y_left.extend((y1,y2))
                else:
                    x_right.extend((x1,x2))
                    y_right.extend((y1,y2))

    if x_left and x_right:
        p_left = np.polyfit(x_left, y_left, 1)
        p_right = np.polyfit(x_right, y_right, 1)
        f_left = np.poly1d(p_left)
        f_right = np.poly1d(p_right)
        
        x_min_left = int((-y_min+f_left).r[0])
        x_min_right = int((-y_min+f_right).r[0])
        
        x_max_left = int((-y_max+f_left).r[0])
        x_max_right = int((-y_max+f_right).r[0])
    
        # add to queue
        if x_min_l.size() >= queue_size:
            x_min_l.get()
            x_min_r.get()
            x_max_l.get()
            x_max_r.get()            
        x_min_l.put(x_min_left)
        x_min_r.put(x_min_right)
        x_max_l.put(x_max_left)
        x_max_r.put(x_max_right)
    
    x_min_left_a = x_min_l.avg()
    x_min_right_a = x_min_r.avg()
    x_max_left_a = x_max_l.avg()
    x_max_right_a = x_max_r.avg()
    # move these two line out of the if    
    cv2.line(img,(x_min_left_a, y_min),(x_max_left_a, y_max),color,thickness)
    cv2.line(img,(x_min_right_a, y_min),(x_max_right_a, y_max),color,thickness)
```

detail explanation:
	first of all, I define a slope threshold to filter out lines that cannot be lane lines, e.g. horizontal lines, vertical lines. Only the slope from 0.4 to 100 will be kept. Then I classify left lane and right line based on plus-or-minus value of slope. By storing the points in two different list, I make polyfit for each of them, which means a line will estimate the left lane points, and the other line will estimate the right lane points. However, if I stoped here, the line drawing on video will be jumpy. So, I create a smoothing method. See below.

Smoothing method (eliminate jumpy in video)

```python
class Queue:
    #define a queue for smooth lines in video
    def __init__(self):
        self.items = []
    def empty(self):
        while self.items:
            self.items.pop()
        return self.items
    def put(self, item):
        self.items.insert(0, item)
    def get(self):
        self.items.pop()
    def size(self):
        return len(self.items)
    def avg(self):
        return int(np.mean(self.items))
    def isEmpty(self):
        if not self.items:
            return True
        else:
            return False
```

I create a class to queue the line(two point needed for one line) I want to draw. So in draw_lines() function, whenever I have two point for the drawing line, I queue them. The queue is size of 10, meaning it will queue 10 lines for valid frame. Because the lines on video will not change suddenly, I use the average value of the queue to draw line. The result improves a lot in video.

![alt text][image9]

Video demos is here:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=5deeV4hq51M
" target="_blank"><img src="http://img.youtube.com/vi/5deeV4hq51M/0.jpg" 
alt="Challenge Video" width="240" height="180" border="10" /></a>




### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be the algorithem fell to detect line in frame 1 in video. If it fell, the queue is empty also, there is no line to draw. Here it will break the code. One solution is to use error handling mechanics to make the algorithm running without drawing lines on 1st frame. 

Another shortcoming after hough transform, there will be some noise points that far away from main line. These noise will affect the slope of drawing lines, making is unaccurate. One posible solution is to create another function to get rid of these noise.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use error handling mechanics to make the algorithm running withour breaking if the algorithm falls to detect lanes on 1st frame in video.

Another potential improvement could be to create functions to get rid of noise after using hough transform.

One last improvement would be to use more video to test this algorithm. It will help tuning parameters and improve accuracy under difference road condition, e.g. day v.s. night, sunny v.s. rainy.

