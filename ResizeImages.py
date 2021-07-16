from PIL import Image
import glob
import os

image_list = []
resized_image = []

for filename in glob.glob('E:/sklearn Datasets/ImagesTest/*.jpg'):

    img = Image.open(filename)
    image_list.append(img)
    
# Resized image
for image in image_list:
    image = image.resize((244,244))
    resized_image.append(image)
    
# Saved resized image
for (i,new) in enumerate(resized_image):
    new.save('{}{}{}'.format('E:/sklearn Datasets/NewImageTest',i+1,'.jpg'))
