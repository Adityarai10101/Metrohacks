import os

import segmentation_models as sm
from PIL import Image

x_train = []
y_train = []
x_val = []
y_val = []
for item in os.listdir("D:/Programming/metrohacks/Foot Ulcer Segmentation Challenge/train/images"):
    x_train.append(list(Image.open(os.path.join("D:/Programming/metrohacks/Foot Ulcer Segmentation Challenge/train/images", item)).getdata()))
for item in os.listdir("D:/Programming/metrohacks/Foot Ulcer Segmentation Challenge/train/labels"):
    y_train.append(list(Image.open(os.path.join("D:/Programming/metrohacks/Foot Ulcer Segmentation Challenge/train/labels", item)).getdata()))
for item in os.listdir("D:/Programming/metrohacks/Foot Ulcer Segmentation Challenge/validation/images"):
    x_val.append(list(Image.open(os.path.join("D:/Programming/metrohacks/Foot Ulcer Segmentation Challenge/validation/images", item)).getdata()))
for item in os.listdir("D:/Programming/metrohacks/Foot Ulcer Segmentation Challenge/validation/labels"):
    y_val.append(list(Image.open(os.path.join("D:/Programming/metrohacks/Foot Ulcer Segmentation Challenge/validation/labels", item)).getdata()))

model = sm.Unet()
model.compile('Adam',loss=sm.losses.bce_jaccard_loss,metrics=[sm.metrics.iou_score])

model.fit(
   x=x_train,
   y=y_train,
   batch_size=16,
   epochs=200,
   validation_data=(x_val, y_val),
)