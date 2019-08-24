import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "E:\GithubRepos\PetImages"
CATEGORIES = ["Dog","Cat"]
IMG_SIZE = 500

# Note: 
# Remove non-differentiating factors. Filter out as much as these as possible

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to cats or dogs dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path)[:10]: # remove [:10] once done
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) # grayscale because colour isn't a differentiating factor
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            #plt.imshow(new_array, cmap ='gray')
            #plt.show()
            #break
        #break

create_training_data()
print(len(training_data))

random.shuffle(training_data)

X=[]
Y=[]

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE, 1)

# pickle is used as a serializer to save/retrieve objects so to reuse data on models directly
# We do this here so we don't have to create 
# training data from raw input every time
pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()
pickle_out = open("Y.pickle","wb")
pickle.dump(Y,pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
print(X[1])