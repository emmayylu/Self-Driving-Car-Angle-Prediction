#Load modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv

from numpy import percentile

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, Flatten,Dense, Activation
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error, r2_score

#Load data
data = pd.read_table("driving_dataset/data.txt",delimiter = ' ',names= ['files','angles'], dtype ={'results': np.float16})

#Preprocess data
#Round dngles
data['angles'] = np.round(data['angles'])
#Drop data
#Drop occurrences < 200
lessthan200 = data.angles.value_counts() <200
selected_angles = data.angles.value_counts()[lessthan200].index.to_list()

for angle in selected_angles:
    data = data.loc[data.angles != angle]

#Keep at most 3000 for each angle
#shuffle Dataframe, reset indexes
data.sample(frac=1)
data = data.sample(frac=1).reset_index(drop=True)

morethan3000 = data.angles.value_counts() >3000
selected_angles = data.angles.value_counts()[morethan3000].index.to_list()

for angle in selected_angles:
    df = data.loc[data.angles == angle]
    num_drop = round(df.shape[0] - 3000)
    portion_drop = df.iloc[:num_drop,:]
    data = pd.concat([data, portion_drop]).drop_duplicates(keep = False)

#Remove outliers
q25, q75 = percentile(data.angles, 25), percentile(data.angles, 75)
iqr = q75 - q25
lower, upper = q25 - 1.5*iqr, q75 + 1.5*iqr
data = data.loc[data.angles > lower]
data = data.loc[data.angles < upper]

#Scale pixels to 0-1 range
def preprocess_img(path):
    img = mpimg.imread(path)
    part = img[150:,:,:]
    img = cv.cvtColor(part, cv.COLOR_RGB2YUV)
    img = cv.GaussianBlur(img,(3,3),0)
    img = cv.resize(img, (200,66))
    img = img/255
    return img

#Split data
#shuffle Dataframe, reset indexes
data.sample(frac=1)
data = data.sample(frac=1).reset_index(drop=True)

def split_df(df,train_per, val_per):
    df = df.sample(frac=1).reset_index(drop=True)
    total_samples = df.shape[0]

    num_train = total_samples * train_per
    num_val = total_samples* val_per
    
    train_df = df.loc[:num_train]
    val_df = df.loc[num_train:(num_train + num_val)]
    test_df = df.loc[(num_train + num_val):]

    return train_df, val_df, test_df

train_df, val_df, test_df = split_df(data,train_per = 0.75, val_per= 0.22)

#Create batch generator
def batch_generator(df,batch_size):
    while True:
        batch = df.sample(n = batch_size)
        batch_paths  =  "driving_dataset/" + batch['files']
        df_angles = batch['angles'] 

        batch_imgs = np.asarray(list(map(preprocess_img, batch_paths)))
        batch_angles = np.asarray(df_angles)

        yield (batch_imgs, batch_angles)

#Create model
def create_model():
    model = Sequential()
    model.add(Conv2D(24,(5,5),padding ="valid",strides = (2,2), input_shape = (66,200,3),activation = 'elu'))
    model.add(Conv2D(36,(5,5),padding = "valid",strides = (2,2), activation = "elu"))
    model.add(Conv2D(48,(5,5),padding = "valid",strides = (2,2), activation = "elu"))
    model.add(Conv2D(64,(3,3),padding = "valid", activation = "elu"))
    model.add(Conv2D(64,(3,3),padding = "valid", activation = "elu"))
    model.add(Flatten())
    model.add(Dense(100, activation = "elu",name = "2nd_dense"))
    model.add(Dense(50, activation = "elu",name= "3rd_dense"))
    model.add(Dense(10, activation = "elu",name = "4th_dense"))
    model.add(Dense(1))
    model.compile(optimizer = Adam(lr = 0.0005),loss="mse")#change lr
    return model

model = create_model()

#Train model
steps_train = train_df.shape[0]//64
steps_val = val_df.shape[0]//64

bs = 64
es = EarlyStopping(mode='min', verbose=1, patience=1) #change patience
ep = 30

history = model.fit_generator(
    batch_generator(train_df, batch_size = bs),
    steps_per_epoch = steps_train,
    validation_data = batch_generator(val_df, batch_size = bs),
    validation_steps = steps_val,
    callbacks=[es],
    verbose = 1,
    epochs = ep)

model.save('model.h5')

#Run test data
def new_generator(df):
    while True:
        paths_series  =  "driving_dataset/" + df['files']

        imgs = np.asarray(list(map(preprocess_img, paths_series)))

        yield imgs

test_pred = model.predict_generator(new_generator(test_df),1 )

mse = mean_squared_error(test_df.angles, test_pred)
r2 = r2_score(test_df.angles, test_pred)
print('Test set mean squared error = ', mse)
print('Test set R2 = ', r2)