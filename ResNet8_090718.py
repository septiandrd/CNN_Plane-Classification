import pandas
import keras
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from tensorflow.python.keras._impl.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.keras._impl.keras.applications import inception_resnet_v2
from tensorflow.python.keras._impl.keras.models import load_model
from tensorflow.python.keras._impl.keras.optimizers import Adam
from tensorflow.python.keras._impl.keras.losses import categorical_crossentropy
import time
import pickle
import os

if __name__ == '__main__':

    start = time.time()

    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30
    )

    validation_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        'Dataset/Training',
        target_size=(200, 200),
        batch_size=5,
        class_mode='categorical',
    )

    validation_generator = validation_datagen.flow_from_directory(
        'Dataset/Validation',
        target_size=(200, 200),
        batch_size=5,
        class_mode='categorical',
    )

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'WoodID_ResNet8_model_' + datetime.now().strftime('%d%m%y') + '.hdf5'
    weight_name = 'WoodID_ResNet8_weight_' + datetime.now().strftime('%d%m%y') + '.hdf5'
    history_name = 'ResNet_history_' + datetime.now().strftime('%d%m%y')
    model_path = os.path.join(save_dir, model_name)
    weight_path = os.path.join(save_dir, weight_name)
    EPOCH = 20

    model = inception_resnet_v2.InceptionResNetV2(include_top=True, weights=None, classes=50,
                     pooling='max', input_shape=(183, 245, 3))

    # model = load_model(os.getcwd()+'/saved_models/ResNet_checkpoint_050718_14-1.47-0.54.hdf5')

    opt = Adam(lr=2e-5)
    model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['acc'])

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(
            save_dir, 'ResNet_checkpoint_' +
            datetime.now().strftime('%d%m%y') +
            '_{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'),
        save_best_only=True,
        verbose=1)

    training = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=EPOCH,
        validation_data=validation_generator,
        validation_steps=35,
        callbacks=[checkpoint],
    )

    score = model.evaluate_generator(
        validation_generator,
        steps=35
    )

    print('\nLoss \t\t:',score[0])
    print('Accuracy \t:',score[1]*100,'%')

    with open(os.path.join(save_dir,'saved_models',history_name), 'wb') as file:
        pickle.dump(training.history, file)

    end = time.time()

    menit = (end-start)/60

    print("\n"+model_name+
          "\n %i Epoch finished in %.2f minutes"%(EPOCH,menit))
