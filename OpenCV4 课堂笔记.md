# OpenCV4 课堂笔记

安装 `opencv-contrib-python`，模块更多

```python
import cv2 as cv
```

## Read

```python
img = cv.imread('/path/to/photo')
```

- read an image
- Absolute or relative path is right.

```python
cv.imshow('<name_of_window>', <matrix_of_pixels_to_display>)
```

- display an image in the specified window

```python
cv.waitKey(0)
```

- a key board binding function
- wait for a specific delay, or time in milliseconds for a key to be pressed
- `0` means it will wait for an infinite amount of time for a keyboard key to be pressed

> [!important]
>
> If you have large images, it's possibly going to go off screen.

```python
capture = cv.VideoCapture(<integer>)
capture = cv.VideoCapture('/path/to/video')
```

- integer like `0`, `1`, `2`: when using computer camera or webcam
- absolute or relative path is right

> [!important]
>
> Different from reading an image, the way to read a video is using a loop and read read the video frame by frame.

```python
while True:
    is_true, frame = capture.read()
    cv.imshow('video', frame)
    if cv.waitKey(20) and 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()
```

```python
capture.release()
```

- release capture device

```bash
error: (-215:Assertion failed) size.width>0 && size.height>0 in function 'cv::imshow'
```

- it means don't find the specific image or frame. like no such file or directory.

## Resize and Rescaling

```python
height = frame.shape[0]
width = frame.shape[1]
```

```python
frame_resized = cv.resize(frame, dimensions, interpolation=cv.INTER_AERA)
```

- resize specific frame to specific dimension

```python
def resize_frame(frame, scale=0.75):
    height = int(frame.shape[0] * scale)
    width  = int(frame.shape[1] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
```

```python
def change_resolution(width, height):
    capture.set(3, width)
    capture.set(4, height)
```

- `capture.set()` is specifically for videos, but work for images too.
- `3` and `4` are property identifier from `cv::VideoCaptuerPropertyies`

## Drawing Shapes & Putting Text

```python
import numpy as np
```

```python
blank = np.zeros((500, 500, 3), dtype='uint8')
```

- make a blank image

```python
blank[:] = 0, 0, 255
blank[200:300, 300:400] = 0,0,255
cv.imshow('Green', frame)
```

- paint the image a certain color

```python
cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=2)
cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=cv.FILLED/-1)
```

- draw a rectangle

```python
cv.circle(blank, (250, 250), 40, (0, 0, 255), thickness=3)
```

- draw a circle

```python
cv.line(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (255, 255, 255), thickness=3)
```

- draw a line

```python
cv.putText(blank, "Hello", (255, 255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
```

- write text

## 5 essential function in OpenCV

```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

- convert image to gray scale image

```python
blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
```

- blur
- kernel size should always be odd

```python
canny = cv.Canny(img, 125, 175)
canny = cv.Canny(blur, 125, 175)
```

- edge cascade
- reduce edges by blurring the image

```python
dilated = cv.dilate(canny, (3, 3), iterations=1)
```

- dilate the image

```python
eroded = cv.erode(dilated, (3, 3), iterations=1)
```

- erode the image

```python
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
```

- `cv.resize()` will ignore the aspect ratio.
- use `cv.INTER_AREA` when the resized imaged is smaller than the raw image
- use `cv.INTER_LINEAR` or `cv.INTER_CUBIC` when the resized image is bigger than the raw image
- `cv.INTER_CUBIC` is the slowest, but the result is best

```
cropped = img[50:200, 200:400]
```

- crop the image

## Image Transformations

```python
def translate(img, x, y):
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, translation_matrix, dimensions)

# -x --> left
# -y --> up
#  x --> right
#  y --> down
```

- translation

```python
def rotate(img, angle, rotation_point=None):
    (height, width) = img.shape[:2]
    
    if rotation_point is None:
        rotation_point = (width//2, height//2)
    
    rotation_matrix = cv.getRotationMatrix2D(rotation_point, angle, 1.0)
    dimensions = (width, height)
    
    return cv.warpAffine(img, rotation_matrix, dimensions)
```

- rotation
- rotation will introduce black triangles

```python
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
```

- resize

```python
flip = cv.flip(img, 0)
```

- `0`, `1`, `-1`

```python
crpped = img[200:400, 300:400]
```

- array slicing

## Contour Detection

Contour are basically the boundaries of objects, the line or curve that joins the continuous points along the boundary of an object.

From a mathematical point of view, they are not the same as edges.

For the most part, you can get away with thinking of contours as edges.

```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

- convert image into gray before contour detection

```python
canny = cv.Canny(img, 125, 175)
```

- canny edge

```python
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
```

- use blur to reduce contours

```python
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
```

- threshold method

```python
blank = np.zeros(img.shape[:2], dtype='uint8')

cv.drawContours(blank, contours, -1, (0, 0, 255), 2)
```

> [!important]
>
> what is recommended is to use canny instead of threshold.

## Color space

```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

- BGR to gray scale image, vice verse

```python
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
```

- BGR to HSV, vice verse

```python
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
```

- BGR to L\*a\*b, vice verse

> [!important]
>
> Only OpenCV use BGR, other libs or operating systems use RGB instead.

```python
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
```

- BGR to RGB, vice verse

```python
bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
```

- cannot convert gray scale image to HSV directly

## Color channels

```python
b, g, r = cv.split(img)
```

- when display `b`, `g`, `r` using `cv.imshow()`, they are displayed as grayscale image to show the distribution of pixel intensities(lighter <--> more concentration of pixel, vice verse)

```python
merged = cv.merged([b, g, r])
```

- merged b, g, r channel

```python
blank = np.zeros(img.shape[:2], dtype='uint8')

blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])
```

## Blurring Techniques

smooth out the image or reduce some of the noise by applying some blurring method

kernel, kernel size

blur is applied to the middle pixel as a result of the pixels around it (called surrounding pixels) 

```python
average = cv.blur(img, (3, 3))
```

- averaging blur:  just compute the average of surrounding pixels

```python
gauss = cv.GaussianBlur(img, (7, 7), 0)
```

- gaussian blur looks more natural compared to averaging blur

```python
median = cv.medianBlur(img, 3)
```

- median blur

```python
bilateral = cv.bilateralFilter(img, 5, 15, 15)
```

- bilateral blur (recommend)

> [!important]
>
> when you are trying to apply blurring the image, especially with the bilateral and median lowering, because higher values of this basic mouth or bilateral or the kernel size for medium glowing, you tend to end up with smudged version of this image.

## Bitwise operation

```python
bitwise_and = cv.bitwise_and(rectangle, circle)
```

- bitwise AND --> intersecting regions
- like *

```python
bitwise_or = cv.bitwise_or(rectangle, circle)
```

- bitwise OR --> non-intersecting and intersecting regions
- like +

```python
bitwise_xor = cv.bitwise_xor(rectangle, circle)
```

- bitwise XOR --> non-intersecting regions

## Masking

```python
import cv2 as cv
import numpy as np
```

```python
blank = np.zeros(img.shape[:2], dtype='uint8')
```

```python
mask = cv.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, 25)
```

```python
masked = cv.bitwise_and(img, img, mask=mask)
```

> [!important]
>
> The mask must has to be at the same dimensions as that of the  image, if not, error will be raised.

## Computing Histograms

Histograms allow you to visualize the distribution of pixel intensities in an image.

```python
import matplotlib,pyplot as plt
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])

plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()
```

- Grayscale Histogram

```python
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()
```

- Color Histogram

## Thresholding

Thresholding is a binary realisation of an image. Binary image: the pixel only equals to 0 (black) or 255 (white).

Choose a particular value that called thresholding, compare each pixel to it, less = 0, greater = 255.

Simple Thresholding and Adaptive Thresholding.

```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

```python
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
```

- Simple Thresholding.

```python
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 3)
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)
```

- Adaptive Thresholding: let computer find the best threshold value self.

## Edge Detection

Canney Detector before.

```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

```python
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
```

- Laplacian

```python
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)
```

- Sobel

Canny is a more advanced detection way.

## Face Detection

haarcascades

Face detection does not involve skin tone or the colors that are present in the image. Haarcascades essentially look at an object in an image and using the edges tries to determine whether it's a face or not.

First convert image to grayscale image.

```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

```python
haar_cascade = cv.CascadeClassifier('haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, miniNeightbors=3)
```

Draw the face rectangle.

``` python
for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
```

Haarcascades are really sensitive to noise in an image.

Haarcascades is not the most advanced method to detect faces.

If you wanted to extend this to videos, what you need to do is essentially detect haarcascades on each individual frame of a video.

## Face Recognition with OpenCV's built-in recognizer

```python
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)
face_recognizer.save("face_trained.yml")
```

```python
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")
```

## Deep Computer Vision

There are very few things that can actually beat a deep learning model.
