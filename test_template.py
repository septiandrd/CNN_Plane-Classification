
# coding: utf-8

# ![title](https://image.ibb.co/erDntK/logo2018.png)
# 
# ---
# 

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import csv


# In[ ]:


# dimensions of our images.
img_width, img_height = 200, 300


# all the images is in directory "/test/images"
test_data_dir = 'test'
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    shuffle=False,
    target_size=(img_width, img_height),
    batch_size=batch_size)

filenames = test_generator.filenames


# In[ ]:


# load your model here
model = load_model('saved_models/Checkpoint_InceptionResNetV2_281118_08-1.30-0.70.h5')

# or load your weights here
# model.load_weights("your_weights.hdf5")


# In[ ]:


# 'test_target.csv' is given
# class target for each file in test/images directory
test_target = np.genfromtxt('test_target.csv', delimiter=',').astype(int)

# 'ids_test.csv' is given
# classname (label) for 20 class plane dataset
id_target = np.load('ids_test.npy').item() 


# In[ ]:


score_predicted = model.predict_generator(test_generator)

class_predicted = score_predicted.argmax(axis=-1)

# print(class_predicted[:10])
# np.savetxt("predicted.csv", class_predicted, fmt='%i', delimiter=",")


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score, f1_score

accuracy = accuracy_score(test_target, class_predicted)
print("Data Test Accuracy = %.3f%%" % (accuracy*100))

f1_macro = f1_score(test_target, class_predicted, average='macro')
f1_micro = f1_score(test_target, class_predicted, average='micro')
print("F1 Score: Macro = %.3f, Micro = %.3f" % (f1_macro,f1_micro))


# In[ ]:


# target_names = list(map(str, id_target.values()))
target_names = list(map(str, id_target.keys()))

print(classification_report(test_target, class_predicted, target_names=target_names))


# ![footer](https://image.ibb.co/hAHDYK/footer2018.png)
