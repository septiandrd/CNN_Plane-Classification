import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
# from keras.applications import inception_v3
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.layers import Dense, Input, Conv2D, Flatten, BatchNormalization, ReLU, add, concatenate, GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D, MaxPooling2D
from keras.models import Sequential, Model
from datetime import datetime
import tensorflow as tf
import time
import pickle
import os

import telegram
bot = telegram.Bot(token="631334683:AAEKuP9g-WcJ_jJgIvFfaQ99uHs5C5S73nU")

def ReductionModule(inputs) :
  t1 = MaxPooling2D(strides=(2,2))(inputs)

  t2 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same')(inputs)
  t2 = BatchNormalization(axis=3)(t2)
  t2 = ReLU()(t2)
  t2 = Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding='same')(inputs)
  t2 = BatchNormalization(axis=3)(t2)
  t2 = ReLU()(t2)
  t2 = Conv2D(filters=96, kernel_size=(3,3), strides=(2,2), padding='same')(inputs)
  t2 = BatchNormalization(axis=3)(t2)
  t2 = ReLU()(t2)

  t3 = Conv2D(filters=384, kernel_size=(3,3), strides=(2,2), padding='same')(inputs)
  t3 = BatchNormalization(axis=3)(t3)
  t3 = ReLU()(t3)

  x = concatenate([t1,t2,t3],axis=3)
  return x

def InceptionModuleF5(inputs) :
  t1 = AveragePooling2D(strides=(1,1),padding='same')(inputs)
  t1 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same')(t1)
  t1 = BatchNormalization(axis=3)(t1)
  t1 = ReLU()(t1)

  t2 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same')(inputs)
  t2 = BatchNormalization(axis=3)(t2)
  t2 = ReLU()(t2)
  t2 = Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding='same')(inputs)
  t2 = BatchNormalization(axis=3)(t2)
  t2 = ReLU()(t2)
  t2 = Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding='same')(inputs)
  t2 = BatchNormalization(axis=3)(t2)
  t2 = ReLU()(t2)

  t3 = Conv2D(filters=48, kernel_size=(1,1), strides=(1,1), padding='same')(inputs)
  t3 = BatchNormalization(axis=3)(t3)
  t3 = ReLU()(t3)
  t3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inputs)
  t3 = BatchNormalization(axis=3)(t3)
  t3 = ReLU()(t3)

  t4 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same')(inputs)
  t4 = BatchNormalization(axis=3)(t4)
  t4 = ReLU()(t4)

  x = concatenate([t1,t2,t3,t4],axis=3)
  return x

def InceptionModuleF6(inputs) :
  t1 = AveragePooling2D(strides=(1,1),padding='same')(inputs)
  t1 = Conv2D(filters=192, kernel_size=(1,1), strides=(1,1), padding='same')(t1)
  t1 = BatchNormalization(axis=3)(t1)
  t1 = ReLU()(t1)

  t2 = Conv2D(filters=128, kernel_size=(1,1), strides=(1,1), padding='same')(inputs)
  t2 = BatchNormalization(axis=3)(t2)
  t2 = ReLU()(t2)
  t2 = Conv2D(filters=128, kernel_size=(7,1), strides=(1,1), padding='same')(t2)
  t2 = BatchNormalization(axis=3)(t2)
  t2 = ReLU()(t2)
  t2 = Conv2D(filters=128, kernel_size=(1,7), strides=(1,1), padding='same')(t2)
  t2 = BatchNormalization(axis=3)(t2)
  t2 = ReLU()(t2)
  t2 = Conv2D(filters=128, kernel_size=(7,1), strides=(1,1), padding='same')(t2)
  t2 = BatchNormalization(axis=3)(t2)
  t2 = ReLU()(t2)
  t2 = Conv2D(filters=192, kernel_size=(1,7), strides=(1,1), padding='same')(t2)
  t2 = BatchNormalization(axis=3)(t2)
  t2 = ReLU()(t2)

  t3 = Conv2D(filters=128, kernel_size=(1,1), strides=(1,1), padding='same')(inputs)
  t3 = BatchNormalization(axis=3)(t3)
  t3 = ReLU()(t3)
  t3 = Conv2D(filters=128, kernel_size=(7,1), strides=(1,1), padding='same')(t3)
  t3 = BatchNormalization(axis=3)(t3)
  t3 = ReLU()(t3)
  t3 = Conv2D(filters=192, kernel_size=(1,7), strides=(1,1), padding='same')(t3)
  t3 = BatchNormalization(axis=3)(t3)
  t3 = ReLU()(t3)

  t4 = Conv2D(filters=192, kernel_size=(1,1), strides=(1,1), padding='same')(inputs)
  t4 = BatchNormalization(axis=3)(t4)
  t4 = ReLU()(t4)

  x = concatenate([t1,t2,t3,t4],axis=3)
  return x

def InceptionModuleF7(inputs) :
  t1 = AveragePooling2D(strides=(1,1),padding='same')(inputs)
  t1 = Conv2D(filters=192, kernel_size=(1,1), strides=(1,1), padding='same')(t1)
  t1 = BatchNormalization(axis=3)(t1)
  t1 = ReLU()(t1)

  t2 = Conv2D(filters=448, kernel_size=(1,1), strides=(1,1), padding='same')(inputs)
  t2 = BatchNormalization(axis=3)(t2)
  t2 = ReLU()(t2)
  t2 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same')(t2)
  t2 = BatchNormalization(axis=3)(t2)
  t2 = ReLU()(t2)
  t21 = Conv2D(filters=384, kernel_size=(3,1), strides=(1,1), padding='same')(t2)
  t21 = BatchNormalization(axis=3)(t21)
  t21 = ReLU()(t21)
  t22 = Conv2D(filters=384, kernel_size=(1,3), strides=(1,1), padding='same')(t2)
  t22 = BatchNormalization(axis=3)(t22)
  t22 = ReLU()(t22)
  t2 = concatenate([t21,t22],axis=3)

  t3 = Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), padding='same')(inputs)
  t3 = BatchNormalization(axis=3)(t3)
  t3 = ReLU()(t3)
  t31 = Conv2D(filters=384, kernel_size=(3,1), strides=(1,1), padding='same')(t3)
  t31 = BatchNormalization(axis=3)(t31)
  t31 = ReLU()(t31)
  t32 = Conv2D(filters=384, kernel_size=(1,3), strides=(1,1), padding='same')(t3)
  t32 = BatchNormalization(axis=3)(t32)
  t32 = ReLU()(t32)
  t3 = concatenate([t31,t32],axis=3)

  t4 = Conv2D(filters=320, kernel_size=(1,1), strides=(1,1), padding='same')(inputs)
  t4 = BatchNormalization(axis=3)(t4)
  t4 = ReLU()(t4)

  x = concatenate([t1,t2,t3,t4],axis=3)
  return x

def InceptionV3(input_shape,num_class) :
  inputs = Input(input_shape)
  x = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(inputs)
  x = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same')(x)
  x = BatchNormalization(axis=3)(x)
  x = ReLU()(x)
  x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
  x = BatchNormalization(axis=3)(x)
  x = ReLU()(x)
  x = MaxPooling2D(padding='same')(x)
  x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
  x = BatchNormalization(axis=3)(x)
  x = ReLU()(x)
  x = Conv2D(filters=80, kernel_size=(3,3), strides=(1,1), padding='same')(x)
  x = BatchNormalization(axis=3)(x)
  x = ReLU()(x)
  x = Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same')(x)
  x = BatchNormalization(axis=3)(x)
  x = ReLU()(x)
  x = MaxPooling2D(padding='same')(x)
  x = InceptionModuleF5(x)
  x = InceptionModuleF5(x)
  x = InceptionModuleF5(x)
  x = ReductionModule(x)
  x = InceptionModuleF6(x)
  x = InceptionModuleF6(x)
  x = InceptionModuleF6(x)
  x = InceptionModuleF6(x)
  x = InceptionModuleF6(x)
  x = ReductionModule(x)
  x = InceptionModuleF7(x)
  x = InceptionModuleF7(x)
  x = GlobalAveragePooling2D()(x)
  predictions = Dense(num_class, activation='softmax')(x)

  model = Model(input=inputs, output=predictions)
  return model


if __name__ == '__main__':

    start = time.time()

    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        brightness_range=(0.0,1.0)
    )

    validation_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        'Dataset/Training',
        target_size=(200, 300),
        batch_size=25,
        class_mode='categorical',
        color_mode='grayscale'
    )

    validation_generator = validation_datagen.flow_from_directory(
        'Dataset/Validation',
        target_size=(200, 300),
        batch_size=25,
        class_mode='categorical',
        color_mode='grayscale'
    )

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    arch_name = "InceptionV3-m"
    model_name = 'Model_InceptionV3-m_' + datetime.now().strftime('%d%m%y') + '.hdf5'
    weight_name = 'Weight_InceptionV3-m_' + datetime.now().strftime('%d%m%y') + '.hdf5'
    history_name = 'History_InceptionV3-m_' + datetime.now().strftime('%d%m%y')
    model_path = os.path.join(save_dir, model_name)
    weight_path = os.path.join(save_dir, weight_name)
    EPOCH = 20

    model = InceptionV3(input_shape=(200,300,1), num_class=70)
    model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['acc'])
    model.summary()

    tensorboard = TensorBoard()
    earlystop = EarlyStopping(patience=5,monitor='val_acc')
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(
            save_dir, 'Checkpoint_InceptionV3-m_' +
            datetime.now().strftime('%d%m%y') +
            '_{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'),
        save_best_only=True,
        verbose=1)

    with tf.Session() as sess :
      sess.run(tf.global_variables_initializer())
      training = model.fit_generator(
          train_generator,
          steps_per_epoch=800,
          epochs=EPOCH,
          validation_data=validation_generator,
          validation_steps=55,
          callbacks=[checkpoint,tensorboard,earlystop],
      )

      score = model.evaluate_generator(
          validation_generator,
          steps=55
      )

    print('\nLoss \t\t:',score[0])
    print('Accuracy \t:',score[1]*100,'%')

    with open(os.path.join(save_dir,history_name), 'wb') as file:
        pickle.dump(training.history, file)

    end = time.time()

    menit = (end-start)/60

    print("\n"+model_name+
          "\n %i Epoch finished in %.2f minutes"%(EPOCH,menit))

    bot.send_message(chat_id='477030905', text="Training "+arch_name+" finished. "
                        "\nLoss : "+str(score[0])+" Accuracy : "+str(score[1]*100)+"%")
