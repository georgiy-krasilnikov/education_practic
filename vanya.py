from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import keras.utils as image
from os import listdir
#Иван Скорский - УИБ-213
def result(dir, true, false, model): #функция, выводящая результат распознавания каждого изображения в каждой папке контрольной выборки
    img_dir = listdir(dir) #применение функции listdir из пакет os для перебора файлов
    for file in img_dir: #цикл перебора файлов
        img = image.load_img(dir+"/"+file, target_size=(225, 225)) #загружаем изображение

        x = image.img_to_array(img)
        x = x.reshape(1,225, 225, 3)
        x /= 255
        prediction = model.predict(x) #распознавание изображения нейросетью

        if (prediction[0][0]<(0.5)):#оценка распознавания
            show(false, prediction[0][0], img) #вывод в неправильном случае
        else:
            show(true, prediction[0][0], img) #вывод в правильном случае

def show(title, pred, img): #функция показа результата распознавания
    plt.imshow(img.convert('RGBA'))
    plt.title(title + str(pred)) #установка названия
    plt.show() #показ результата

#каталог с данными для обучения
train_dir = 'train'
#каталог с данными для тестирования
test_dir = 'test'
#размеры изображения
img_width, img_height = 225, 225
#размерность тензора на основе изображения для входных данных в нейронную сеть
#backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
#количество эпох
epochs = 25
#размер мини-выборки
batch_size = 4
#количество изображений для обучения
nb_train_samples = 104
#количество изображений для тестирования
nb_test_samples = 56
#accuracy - обучающий набор данных
#val_accuray - проверочный набор данных
model = Sequential()
#начало каскада свертки
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#конец каскада свертки
#полносвязная часть необходимая для классификации
model.add(Flatten())#слой преобразует двух мерный вывод который мы получаем из слоя MaxPooling в одномерный вектор
model.add(Dense(64))#полносвязный слой в котором 64 нейрона
model.add(Activation('relu'))#функция активации полу-линейная
model.add(Dropout(0.5))#слой для регулиризации и предотвращения переобучения
model.add(Dense(1))#выходной слой который содержит 1 нейрон с сигмоидальной функцией активации(2 значения 0 и 1(зебра или не зебра)
model.add(Activation('sigmoid'))

#компиляция сети
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1. / 225)#обеспечение загрузки данных и их преобразования(делим каждый пексель на изображения размерностью 225)
#генераотр данных для обучения на основе изображений из каталога
train_generator = datagen.flow_from_directory( #загрузка изображения из каталога
    train_dir,#каталог
    target_size=(img_width, img_height),#размер изображения
    batch_size=batch_size,#размер выборки(кол-во изображений которые будут прочитаны за один раз)
    class_mode='binary')

#генератор данных для тестирования на основе изображений из каталога
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(#данные мы берем от генераторов
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,#общее колво изображений длеленное на количество изображений
    epochs=epochs,#кол - во эпох(16 - 25)
    validation_data=test_generator,
    validation_steps=nb_test_samples // batch_size)#сколько раз мы обращаемся к проверочному генератору

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)

print("Accuracy: ",scores[1])

result('test/zebra', "Зебра -> ", "Не зебра -> ", model)
result('test/nezebra', "Зебра -> ", "Не зебра -> ", model)
