import pandas as pd
import numpy as np
import cv2
import math
import psutil
from PIL import Image

class XtoYConverter():
   def __convert_rle_to_mask(rle,shape):
        res = np.zeros((shape[0],shape[1])).tolist()
    
        for i in range(len(rle)):
            encoded_pixels = rle.iloc[i]
            if (encoded_pixels != encoded_pixels):
                break
            arr = encoded_pixels.split()
            for j in range(0,len(arr),2):
                cnt = int(arr[j+1])
                value = int(arr[j])
                for k in range(cnt):
                    res[(value-1)//shape[0]][(value-1)%shape[0]] = 1
                    value+=1
        return np.array([res,res,res]) 
   
   def createYFromXdata(train,x):
         for indx, values in x.items():
            if values != None:
                image_rle = train[train['ImageId']==values]['EncodedPixels']
                yield (XtoYConverter.__convert_rle_to_mask(image_rle,(768,768,3)))
            else:
                yield (np.zeros((768,768,3)))


path_to_train_dataset = './data/help/train_ship_segmentations_v2.csv'
train = pd.read_csv(path_to_train_dataset)

train_with_ship = train[train['EncodedPixels'].notnull()]['ImageId'].unique()
train_without_ship = train[train['EncodedPixels'].isna()]['ImageId'].unique()
train_without_ship = train_without_ship[:len(train_with_ship)]
# 50% with ship, 50% without

specific_train = pd.concat([pd.Series(train_with_ship),pd.Series(train_without_ship)])
#shuffling dataframe 
specific_train = specific_train.sample(frac=1).reset_index(drop=True)
limit = 12000
specific_train = specific_train[:limit]
specific_train_train_x = specific_train[:math.floor(len(specific_train)*0.8)]
specific_train_val_x =  specific_train[math.floor(len(specific_train)*0.8):]

print(specific_train_train_x)
print(specific_train_val_x)

print(psutil.virtual_memory().available)

def read_image_train(file_name):
    image = cv2.imread('./data/train_v2/'+file_name)
    return image

def save_mask_as_image(mask, path):
    mask_image = Image.fromarray(mask[0]*255)
    mask_image = mask_image.convert('RGB')
    mask_image.save(path)

print(len(specific_train_train_x))

ind = 1
for i in XtoYConverter.createYFromXdata(train,specific_train_train_x):
    save_mask_as_image(i,'./processed_data_train/train/masks/'+'mask'+str(ind)+'.jpg')
    ind += 1

ind = 1
for index, values in specific_train_train_x.items():
    cv2.imwrite('./processed_data_train/train/images/image'+str(ind)+'.jpg',read_image_train(values))
    ind += 1

ind = 1
for i in XtoYConverter.createYFromXdata(train,specific_train_val_x):
    save_mask_as_image(i,'./processed_data_train/validation/masks/'+'mask'+str(ind)+'.jpg')
    ind+=1
ind = 1
for index, values in specific_train_val_x.items():
    cv2.imwrite('./processed_data_train/validation/images/image'+str(ind)+'.jpg',read_image_train(values))
    ind+=1


