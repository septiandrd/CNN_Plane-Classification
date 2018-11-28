import pandas
import keras

# import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import inception_v3
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential, Model
from datetime import datetime
import time
import pickle
import os

if __name__ == '__main__':

    start = time.time()

    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        brightness_range=(0.0,1.0),
        channel_shift_range=5.0,
    )

    validation_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        'Dataset/Training',
        target_size=(200, 300),
        batch_size=25,
        class_mode='categorical',
    )

    validation_generator = validation_datagen.flow_from_directory(
        'Dataset/Validation',
        target_size=(200, 300),
        batch_size=25,
        class_mode='categorical',
    )

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'Model_InceptionV3-trf_' + datetime.now().strftime('%d%m%y') + '.hdf5'
    weight_name = 'Weight_InceptionV3-trf_' + datetime.now().strftime('%d%m%y') + '.hdf5'
    history_name = 'History_InceptionV3-trf_' + datetime.now().strftime('%d%m%y')
    model_path = os.path.join(save_dir, model_name)
    weight_path = os.path.join(save_dir, weight_name)
    EPOCH = 100

    model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(200, 300, 3))

    # model.summary()
    # print(model.layers[:-12])

    for layer in model.layers[:-12] :
      layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(70, activation="softmax")(x)

    model_combined = Model(input = model.input, output = predictions)

    model_combined.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

    checkpoint = ModelCheckpoint(
      monitor='val_acc',
      filepath=os.path.join(
          save_dir, 'Checkpoint_InceptionV3-trf_' +
          datetime.now().strftime('%d%m%y') +
          '_{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'),
      save_best_only=True,
      verbose=1,
      mode='auto',
      period=1
    )

    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

    training = model_combined.fit_generator(
        train_generator,
        steps_per_epoch=1000,
        epochs=EPOCH,
        validation_data=validation_generator,
        callbacks=[checkpoint,early],
    )

    score = model_combined.evaluate_generator(
        validation_generator,
    )

    print('\nLoss \t\t:',score[0])
    print('Accuracy \t:',score[1]*100,'%')

    with open(os.path.join(save_dir,history_name), 'wb') as file:
        pickle.dump(training.history, file)

    end = time.time()

    menit = (end-start)/60

    print("\n"+model_name+
          "\n %i Epoch finished in %.2f minutes"%(EPOCH,menit))
