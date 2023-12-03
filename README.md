# Airbus-ship-segmentation
Solution for Airbus Ship Detection Challenge

First I created an option to convert RLE to mask for each image. So, my idea was to convert all rles to masks then create processed_data_train folder with images and masks.
Also, to get predictions I created group of functions to convert masks to rle. You can look at this in exploratory analysis jupyter notebook.

So, that was my general idea.

To run my code you need install all packages. Download dataset for this competition. You must have this file structure.
![image](https://github.com/YakubT/Airbus-ship-segmentation/assets/73753564/be4e18bb-bf37-4d68-9e94-c292571a72c6)
Also you need create empty folders for processed_data_train
![image](https://github.com/YakubT/Airbus-ship-segmentation/assets/73753564/96208bd3-823b-41ac-8672-a1d08089136d)
That is needed to create convertion between rle and mask.

#Training
I used U-net model for training
I have limited resources for training so my model didn't train enough

You can look at modelCreationPreparation.py. This file is needed to convert rle to mask. ModelCreation.py - creates model. but you need to run at first modelCreationPreparation.py

I used dice loss for training

#Result
to generate submission csv file you need to run inferences.py.
In this part I convert masks to rle using treshold. Treshold = 0.5. If value > 0.5 then it is part of ship.