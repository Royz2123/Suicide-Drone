import cv2
import os
import time
import numpy as np


class DroneRecognizer():
    DEFAUlT_XML_PATH = "./detection/cascades/drone_cascade_beta.xml"

    def __init__(self, xml_file=DEFAUlT_XML_PATH, trained=True):
        self._drone_cascade = None

        if trained:
            self._drone_cascade = cv2.CascadeClassifier(xml_file)
            print("CASCADE OBJ:\t", self._drone_cascade)

    # Currently done using outside program
    # https://amin-ahmadi.com/cascade-trainer-gui/
    def train(self, im_path="./train_images/drones"):
        pass

    def test(self, im_path="./test_images/drones/test_images/", viz=False):
        img_paths = [im_path + path for path in os.listdir(im_path)][:10]
        print("TESTING ON:\t\t", img_paths)

        imgs = [cv2.imread(path, 0) for path in img_paths]
        times = []

        for img in imgs:
            tick = time.time()
            drones = self._drone_cascade.detectMultiScale(
                img,
                scaleFactor=1.3,
                minNeighbors=4,
                # minSize=(10, 10),
                # maxSize=(500, 500)
            )
            tock = time.time()

            times.append(tock - tick)
            print(f"Detection time:\t\t{times[-1]}")

            for (x, y, w, h) in drones:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if viz:
                scale_percent = 30  # percent of original size
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

                cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        print(f"Average detection time:\t{np.mean(times)}")



