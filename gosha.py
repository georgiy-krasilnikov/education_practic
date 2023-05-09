from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications import VGG16
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import keras.utils as image
from os import listdir

def result(dir, true, false, model): #функция, выводящая результат распознавания каждого изображения в каждой папке контрольной выборки
    img_dir = listdir(dir) 
    for file in img_dir: #перебор файлов в папке
        img = image.load_img(dir+"/"+file, target_size=(225, 225)) #загрузка изображения

        x = image.img_to_array(img)
        x = x.reshape(1,225, 225, 3)
        x /= 255
        prediction = model.predict(x) #обработка нейросетью

        if (prediction[0][0]<(0.5)): #вывод результата в зависимости от успешности распознавания
            show(false, prediction[0][0], img)
        else:
            show(true, prediction[0][0], img)

def show(title, pred, img): #функция показа результата распознавания с помощью библиотеки matplotlib
    plt.imshow(img.convert('RGBA'))
    plt.title(title + str(pred))
    plt.show()

train_dir = 'train' #установка путей к обучающей и контрольной выборкам
test_dir = 'test'
img_width, img_height = 225,225 #размеры для изображений
input_shape = (img_width, img_height, 3)
batch_size = 5 #размер мини-выборки
nb_train_samples=104 #количество изображений  для обучения
nb_test_samples = 56 #количество изображений для распознавания

vgg16_net = VGG16(weights='imagenet', include_top=False, input_shape=(225,225, 3)) #импорт модели VGG16

vgg16_net.trainable = False #игнорирование тренировки на ImageNet

vgg16_net.summary()

model = Sequential() #немного кастомизируем модель VGG16, добавляя некоторые слои
model.add(vgg16_net)
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu')) 
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-5), 
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory( #обучающий генератор
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = datagen.flow_from_directory( #генератор для тестов
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=30,
    validation_data=test_generator,
    validation_steps=nb_test_samples // batch_size)

accuracy = model.evaluate(test_generator)
print("Результат: ", accuracy[1])

result('test/zebra', "Зебра -> ", "Не зебра -> ", model) #выводим результат распознавания изображений из обоих папок тестовой выборки
result('test/nezebra', "Не зебра -> ", "Зебра -> ", model)

