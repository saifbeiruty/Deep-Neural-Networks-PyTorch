import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

# Don't want to rebuild the data every time
REBUILD_DATA = True

class DogsVSCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    onehotvector = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        #Label here is CATS and then in the second loop DOGS. CATS = "PetImages/Cat"
        for label in self.LABELS:
            print(label)
            #tqdm is just a progress bar
            #os.listdir(path) makes a list of all the files inside the path
            # This is the picture of cats and then dogs
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        # Getting the path where the images are located PetImages/Cat/123.jpg
                        path = os.path.join(label, f)
                        # Changing the images to grey scale
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        # resizing the images
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

                        self.training_data.append([np.array(img ,dtype = float), np.eye(2, dtype=float)[self.LABELS[label]]])

                        if label == self.CATS:
                            self.catcount += 1
                        elif label == self.DOGS:
                            self.dogcount += 1
                    except Exception as e:
                        pass
        
        #Shuffles the data in place
        np.random.shuffle(self.training_data)
        # save the data
        with open("training_data.pkl", "wb") as f:
            pickle.dump(self.training_data, f)


        
        #np.save("training_data.npy", self.training_data)

        print("Cats", self.catcount)
        print("Dogs", self.dogcount)

if REBUILD_DATA:
    dogsvscats = DogsVSCats()
    dogsvscats.make_training_data()

training_data = np.load("training_data.pkl", allow_pickle=True)
print(len(training_data))

plt.imshow(training_data[20][0])
plt.show()

#training_data = np.load("training_data.npy")

# for key in training_data:
#     print(training_data[key]) test
#     break
