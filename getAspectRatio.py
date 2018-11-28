from PIL import Image
from tqdm import tqdm
import os

for subdirs,dirs,files in os.walk('./train') :
  if (dirs==[]) :
    height = 0
    count = 0
    for file in tqdm(files):
      if (file.split('.')[1] == "jpg") :
        img = Image.open(os.path.join(subdirs, file)).convert('RGB')
        height+=img.size[0]
        count+=1

avg = height/count
print(avg)
