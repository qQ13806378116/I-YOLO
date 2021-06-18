import os

images_dir = "E:/learning/dataset/FLIR_ADAS_1_3 (2)/FLIR_ADAS_1_3/train/hongwai2"
labels_dir = "E:/learning/dataset/FLIR_ADAS_1_3 (2)/FLIR_ADAS_1_3/train/sxml"

#删除多余的image,
labels = []
for label in os.listdir(labels_dir):
    #labels.append(label.split('.')[0])#不能用这一行，因为有些文件名字前面就有 . 这样得到的文件名字是不对的。 
    labels.append(os.path.splitext(label)[0]) 
print(labels)

for image_name in os.listdir(images_dir):
        #image_name = image_name.split('.')[0] #不能用这一行，因为有些文件名字前面就有 . 
        image_name = os.path.splitext(image_name)[0] 
        #print(image_name)
        if image_name not in labels:
            image_name = image_name + ".jpg"
            print(image_name)
            os.remove(os.path.join(images_dir, image_name))#删除图片，最开始先把这一行注释掉，运行下看看打印，以免误删导致数据还是重新做，