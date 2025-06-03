# project
## Aim
## Software Required:
Anaconda - Python 3.7
## Algorithm:
## I)Perform ROI from an image
### Step1:
Import necessary packages 
### Step2:
Read the image and convert the image into RGB
### Step3:
Display the image
### Step4:
Set the pixels to display the ROI 
### Step5:
Perform bit wise conjunction of the two arrays  using bitwise_and 
### Step6:
Display the segmented ROI from an image.
## II)Perform handwritting detection in an image
### Step1:
Import necessary packages 
### Step2:
Define a function to read the image,Convert the image to grayscale,Apply Gaussian blur to reduce noise and improve edge detection,Use Canny edge detector to find edges in the image,Find contours in the edged image,Filter contours based on area to keep only potential text regions,Draw bounding boxes around potential text regions.
### Step3:
Display the results.
## III)Perform object detection with label in an image
### Step1:
Import necessary packages 
### Step2:
Set and add the config_file,weights to ur folder.
### Step3:
Use a pretrained Dnn model (MobileNet-SSD v3)
### Step4:
Create a classLabel and print the same
### Step5:
Display the image using imshow()
### Step6:
Set the model and Threshold to 0.5
### Step7:
Flatten the index,confidence.
### Step8:
Display the result.




## i) Extract ROI from  an image

## PROGRAM


```py
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[42]:Read the image and convert the image into RGB
image_path = 'bird1.jpg'
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[44]:Display the image
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()


# Step 4: Set the pixels to display the ROI (Region of Interest)
# Define the coordinates for the Region of Interest (ROI)
# (startY:endY, startX:endX)
roi = img[100:420, 200:550]  # ROI coordinates (adjust as needed)

# Create a blank mask of the same size as the original image
mask = np.zeros_like(img)

# Place the ROI on the mask
mask[100:420, 200:550] = roi

# Step 5: Perform bitwise conjunction of the two arrays using bitwise_and
segmented_roi = cv2.bitwise_and(img, mask)


# Step 6: Display the segmented ROI from the image
segmented_roi_rgb = cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2RGB)
plt.imshow(segmented_roi_rgb)
plt.title("Segmented ROI")
plt.axis('off')
plt.show()
```

## OUTPUT

![image](https://github.com/user-attachments/assets/58914edc-64c0-4ea8-b9aa-c50cb8324200)



## ii) Perform handwritting detection in an image.

## PROGRAM

```py


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[3]:Define a function to read the image,Convert the image to grayscale,
#Apply Gaussian blur to reduce noise and improve edge detection,
#Use Canny edge detector to find edges in the image,
#Find contours in the edged image,
#Filter contours based on area to keep only potential text regions,
#Draw bounding boxes around potential text regions.


def detect_handwriting(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detector to find edges in the image
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to keep only potential text regions
    min_area = 100
    text_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Draw bounding boxes around potential text regions
    img_copy = img.copy()
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Handwriting Detection')
    plt.axis('off')
    plt.show()

# Path to the image containing handwriting
image_path ='handwriting.jpg'

# Perform handwriting detection
detect_handwriting(image_path)
# Display the image
```

## OUTPUT

![image](https://github.com/user-attachments/assets/a10f440e-fb30-45e5-9f6d-dee5c7931cfb)

## iii) Perform object detection with label in an image

## PROGRAM

```py

import cv2
import matplotlib.pyplot as plt

# In[4]:Set and add the config_file,weights to ur folder.

config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'

# In[5]:Use a pretrained Dnn model (MobileNet-SSD v3)

model=cv2.dnn_DetectionModel(frozen_model,config_file)

# In[6]:Create a classLabel and print the same
classLabels = []
file_name='Labels.txt'
with open(file_name,'rt')as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')

# In[7]:Print the classLabels
print(classLabels)

# In[8]:
print(len(classLabels))

# In[9]:Display the image using imshow()
img=cv2.imread('car.jpg')
plt.imshow(img)


# In[10]:
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[11]:Set the model and Threshold to 0.5
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)#255/2=127.5
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
print(ClassIndex)


# In[29]:Flatten the index,confidence.
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN
for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(0,0,255),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(255,0,0),thickness=1)


# In[30]:Display the result.
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
```
## OUTPUT
![image](https://github.com/user-attachments/assets/253398a8-c36d-4dde-a59c-b4af4f72f81b)

## Result:
Thus, a python program using OpenCV for following image manipulations is done successfully
