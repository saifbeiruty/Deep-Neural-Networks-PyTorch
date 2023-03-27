import os
import cv2
import numpy as np
from tqdm import tqdm

# Don't want to rebuild the data every time
REBUILD_DATA = True

class DogsVSCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
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
                        # 
                        self.training_data.append([np.array(img, dtype=object), np.eye(2)[self.LABELS[label]]])

                        if label == self.CATS:
                            self.catcount += 1
                        elif label == self.DOGS:
                            self.dogcount += 1
                    except Exception as e:
                        pass
        
        #Shuffles the data in place
        np.random.shuffle(self.training_data)
        print(self.training_data)
        # save the data
        np.save("dddddd.npy", self.training_data)

        print("Cats", self.catcount)
        print("Dogs", self.dogcount)

if REBUILD_DATA:
    dogsvscats = DogsVSCats()
    dogsvscats.make_training_data()


#training_data = np.load("dddddd.npy")

# # np.load("training_data.npy", allow_pickle=True ,encoding='latin1')
#print(len(training_data))