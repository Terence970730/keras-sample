from keras.preprocessing import image
from keras.applications import resnet50, inception_v3, vgg16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.optimizers import Adam
import numpy as np

batch_size = 50
num_classes = 2

#base_model = resnet50.ResNet50
base_model = inception_v3.InceptionV3
#base_model = vgg16.VGG16

base_model = base_model(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['acc'])

x_train = np.random.normal(loc=127, scale=127, size=(50, 224,224,3))
y_train = np.array([0,1]*25)
x_train = resnet50.preprocess_input(x_train)

print(model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0))
model.fit(x_train, y_train,
          epochs=1,
          batch_size=batch_size,
          shuffle=False,
          validation_data=(x_train, y_train))

model.save("model.h5")
print("saved")