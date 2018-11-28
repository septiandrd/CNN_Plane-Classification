import pandas
import keras
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import inception_resnet_v2
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from datetime import datetime
import time
import pickle
import os

if __name__ == '__main__':

    start = time.time()

    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        zca_whitening=True,
        brightness_range=(0.0,1.0),
        channel_shift_range=5.0,
    )

    validation_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        'Dataset/Training',
        target_size=(200, 300),
        batch_size=25,
        class_mode='categorical',
        save_to_dir='Dataset/TrainingAugmented'
    )

    validation_generator = validation_datagen.flow_from_directory(
        'Dataset/Validation',
        target_size=(200, 300),
        batch_size=25,
        class_mode='categorical',
    )

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'Model_' + datetime.now().strftime('%d%m%y') + '.hdf5'
    weight_name = 'Weight_' + datetime.now().strftime('%d%m%y') + '.hdf5'
    history_name = 'History_' + datetime.now().strftime('%d%m%y')
    model_path = os.path.join(save_dir, model_name)
    weight_path = os.path.join(save_dir, weight_name)
    EPOCH = 100

    model = inception_resnet_v2.InceptionResNetV2(include_top=True, weights=None, classes=70,
                     pooling='avg', input_shape=(300, 200, 3))

    # model = load_model(os.getcwd()+'/saved_models/ResNet_checkpoint_050718_14-1.47-0.54.hdf5')

    # opt = Adam(lr=2e-5)
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['acc'])

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(
            save_dir, 'Checkpoint_' +
            datetime.now().strftime('%d%m%y') +
            '_{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'),
        save_best_only=True,
        verbose=1)

    training = model.fit_generator(
        train_generator,
        steps_per_epoch=1000,
        epochs=EPOCH,
        validation_data=validation_generator,
        callbacks=[checkpoint],
    )

    score = model.evaluate_generator(
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
