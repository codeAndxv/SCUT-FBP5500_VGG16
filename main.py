from keras.applications import VGG16
from keras.models import Sequential
from keras import models
from keras.layers import Dense, Flatten, Dropout
import pandas as pd
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
def get_generator(df,batch_size):
    # 定义ImageDataGenerator
    datagen = ImageDataGenerator(
        rescale=1. / 255 # 进行归一化`
    )
    # 使用flow_from_dataframe方法读取数据
    generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='image_path',  # 图片路径列名
        y_col='score',  # 评分列名
        target_size=(224, 224),  # VGG16输入尺寸
        batch_size=batch_size,
        class_mode='raw'  # 以原始值作为输出
    )
    return generator


def get_model():
    # 加载VGG16模型，并去掉顶层
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # 冻结VGG16的权重
    for layer in vgg.layers:
        layer.trainable = False

    # 添加新的顶层
    model = Sequential()
    model.add(vgg)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    # 编译模型
    # model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def read_from_txt(file_path):
    # 读取txt文件内容
    with open(file_path, 'r') as f:
        content = f.readlines()

    # 分割得到图片路径和分数
    image_paths = []
    scores = []
    for line in content:
        line = line.strip()  # 去除行首行尾的空格和换行符
        filename, score = line.split()  # 以空格分割得到文件名和分数
        img_path = os.path.join('./SCUT-FBP5500_v2/Images', filename)  # 拼接出完整路径
        score = float(score)  # 将分数转换为浮点数
        image_paths.append(img_path)
        scores.append(score)
    return pd.DataFrame({'image_path':image_paths, 'score': scores})

model = None

def start():
    train_df = read_from_txt('./SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/train.txt')
    train_generator = get_generator(train_df, 32)
    test_df = read_from_txt('./SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/test.txt')
    valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    valid_generator = get_generator(valid_df, 32)
    test_generator = get_generator(test_df, 32)
    model = get_model()
    checkpoint, early_stopping = get_callbacks()
    model.fit(train_generator,
                steps_per_epoch=len(train_generator),
                epochs=50,
                callbacks=[checkpoint],
                validation_data=valid_generator,
                validation_steps=len(valid_generator))
    model.save('./model1.h5')

def get_callbacks():
    # 创建ModelCheckpoint的回调
    checkpoint = keras.callbacks.ModelCheckpoint(filepath='./my_model.h5',
                                 monitor='val_mae',
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='min',
                                 verbose=1)

    # 创建EarlyStopping的回调
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_mae',
                                                   patience=5,
                                                   mode='min',
                                                   verbose=1)
    return checkpoint, early_stopping

def evaluate():
    model = models.load_model('./my_model.h5')
    test_df = read_from_txt('./SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/test.txt')
    valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    test_generator = get_generator(test_df, 1)

    test_loss, test_mae = model.evaluate(test_generator)
    print('Test loss:', str(test_loss) + ", test_mae: " + str(test_mae))

def predict(file_path):
    model = models.load_model('./my_model.h5')
    # 加载图像并转换为numpy数组
    img = Image.open(file_path)
    img = img.resize((224,224))

    img_array = np.array(img)
    img_array = img_array/255.0
    img_array = np.expand_dims(img_array, axis=0)
    # 将数据进行标准化处理
    # scaler = StandardScaler()
    # img_array_scaled = scaler.fit_transform(img_array.reshape(-1, img_array.shape[-1])).reshape(img_array.shape)


    # 使用Keras的预处理函数进行进一步的转换
    predictions = model.predict(img_array)
    print(predictions)


if __name__ == "__main__":
    # predict('./test/img1.png')
    evaluate()