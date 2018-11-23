import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2

for subdirs,dirs,files in tqdm(os.walk('./train')) :
    if (dirs==[]) :
        dirname = subdirs.split('/')[-1]
        all = len(files)
        valid = round(all/5)
        train = all-valid
        if not os.path.exists('./Dataset/Training/'+dirname):
            os.makedirs('./Dataset/Training/'+dirname)
        if not os.path.exists('./Dataset/Validation/'+dirname):
            os.makedirs('./Dataset/Validation/'+dirname)

        i = 1
        for file in files:
            if (file.split('.')[1] == "jpg") :
                img = Image.open(os.path.join(subdirs, file)).convert('RGB')
                # imgArray = np.asarray(img.resize((1000,1000), Image.ANTIALIAS))
                # imgArray = np.rot90(imgArray)
                # imgArray = np.rot90(imgArray)
                # imgArray = np.rot90(imgArray)
                if(i<=train) :
                    newImg = img
                    newImg.save('./Dataset/Training/'+dirname+'/'+file.split('.')[0]+'.jpg')
                else:
                    newImg = img
                    newImg.save('./Dataset/Validation/' + dirname + '/' + file.split('.')[0]+ '.jpg')
                i+=1