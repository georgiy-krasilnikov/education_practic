from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications import InceptionV3
import matplotlib.pyplot as plt
import keras.utils as image
from os import listdir
#Полина Миронова - УИБ-213
def result(dir, true, false, model): # функция, выводящая результат распознавания каждого изображения в каждой папке контрольной выборки
    img_dir = listdir(dir) # применение функции listdir из пакет os для перебора файлов
    for file in img_dir: # цикл перебора файлов
        img = image.load_img(dir+"/"+file, target_size=(225, 225)) # загружаем изображение

        x = image.img_to_array(img)
        x = x.reshape(1,225, 225, 3)
        x /= 255
        prediction = model.predict(x) # распознавание изображения нейросетью

        if (prediction[0][0]<(0.5)): # оценка распознавания
            show(false, prediction[0][0], img)
        else:
            show(true, prediction[0][0], img)

def show(title, pred, img): # функция показа результата распознавания
    plt.imshow(img.convert('RGBA'))
    plt.title(title + str(pred))
    plt.show()

train_path = 'train' # путь для обучающей выборки
test_path = 'test' # путь для контрольной выбокри

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 5 # размерность подвыборки
img_width, img_height = 225, 225 # размерность тензора на основе изображения для входных данных в нейронную сеть

train_generator = train_datagen.flow_from_directory(train_path, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary') # генератор для обучения
test_generator = test_datagen.flow_from_directory(test_path, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary') # генератор для тестирования на контрольной выборке

inception = InceptionV3(include_top=False, input_shape=(img_width, img_height, 3)) # импорт модели InceptionV3

for layer in inception.layers:
    layer.trainable = False # игнорируем слой для классификации изображений на ILSVRC

model = Sequential() # немного модифицируем нашу модель, добавляя несколько дополнительных слоёв
model.add(inception)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 20 # устанавливаем кол-во эпох - 20
steps_per_epoch = train_generator.n // train_generator.batch_size # устанавливаем размеры шагом
validation_steps = test_generator.n // test_generator.batch_size


model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=test_generator, validation_steps=validation_steps)

accuracy = model.evaluate(test_generator)
print("Accuracy = ", accuracy[1])

result('test/zebra', "Зебра -> ", "Не зебра -> ", model)
result('test/nezebra', "Зебра -> ", "Не зебра -> ", model)
