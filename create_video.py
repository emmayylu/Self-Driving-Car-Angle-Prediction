#Load packages
import pandas as pd
import numpy as np
import cv2 as cv
import os
import warnings
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

from keras.models import load_model
import matplotlib.image as mpimg

#Load data
warnings.filterwarnings("ignore")

model = load_model("model.h5")
data = pd.read_table("driving_dataset/data.txt",delimiter = ' ',names= ['files','angles'], dtype ={'results': np.float16})

def preprocess_img(path):
    img = mpimg.imread(path)
    part = img[150:,:,:]
    img = cv.cvtColor(part, cv.COLOR_RGB2YUV)
    img = cv.GaussianBlur(img,(3,3),0)
    img = cv.resize(img, (200,66))
    img = img/255
    return img

def new_generator(df, num):
    test_df = df.iloc[:num,:]
    while True:
        paths_series  =  "driving_dataset/" + test_df['files']

        imgs = np.asarray(list(map(preprocess_img, paths_series)))

        yield imgs

def write_angle(path, angle, count):
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    (x,y) = (50,50)

    message = 'degree = ' + str(angle)
    color = 'rgb(255, 255, 255)'
    draw.text((x, y), message, fill=color)
    name = 'a%d.png' % count
    image.save(name)
    
    return name

def make_video(video_name,file_names):
    frame = cv.imread(file_names[0])
    height, width = frame.shape[0], frame.shape[1]
    
    video = cv.VideoWriter(video_name + '.avi', 0, 1, (width,height))
    
    for file in file_names:
        video.write(cv.imread(file))

    video.release()

def whole(df, num_photos):
    model = load_model("model.h5")
        
    pred = model.predict_generator(new_generator(data,num_photos),1 )

    file_lst = data['files'].to_list()[:num_photos]
    image_folder = 'driving_dataset'
    count = 0
    new_names = []
    for file in file_lst:
        
        #1. Get the file
        path = os.path.join(image_folder, file)
        
        #2. Get prediction
        angle = pred[count][0]

        #3. Write the angle into image
        name = write_angle(path, angle, count)
        new_names.append(name)
        #Count
        count += 1
    make_video('result', new_names)

    return pred

pred = whole(data, 250)
