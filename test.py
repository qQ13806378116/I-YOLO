
from nets.yolo3 import yolo_body
from keras.layers import Input
inputs = Input([416,416,3])
model = yolo_body(inputs,3,80,phi=5)
model.summary()

# for i,layer in enumerate(model.layers):
#     print(i,layer.name)
