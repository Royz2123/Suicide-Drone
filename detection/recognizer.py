import cv2
import os
import time
import numpy as np


class DroneRecognizer:
    DEFAUlT_XML_PATH = "./detection/cascades/drone_cascade_beta.xml"
    SIZE_CONFIDENCE = 0.7

    def __init__(self, xml_file=DEFAUlT_XML_PATH, trained=True):
        self._drone_cascade = None

        if trained:
            self._drone_cascade = cv2.CascadeClassifier(xml_file)
            print("CASCADE OBJ:\t", self._drone_cascade)

    # Currently done using outside program
    # https://amin-ahmadi.com/cascade-trainer-gui/
    def train(self, im_path="./train_images/drones"):
        pass

    def test(self, im_path="./test_images/drones/images/input/", viz=True, edge=True):
        img_paths = [im_path + path for path in os.listdir(im_path)][:10]
        print("TESTING ON:\t\t", img_paths)

        imgs = [cv2.imread(path, 0) for path in img_paths]
        times = []
        last_size = None
        for img in imgs:
            if edge:
                v = np.median(img)
                sigma = 1.0

                lower = int(max(0, (1.0 - sigma) * v))
                upper = int(min(255, (1.0 + sigma) * v))
                img = cv2.Canny(img, lower, upper)

                kernel = np.ones((2, 2), np.uint8)
                img = cv2.dilate(img, kernel, iterations=1)




            tick = time.time()
            if last_size is None:
                drones = self._drone_cascade.detectMultiScale(
                    img,
                    scaleFactor=1.3,
                    minNeighbors=1,
                    flags=cv2.CASCADE_DO_CANNY_PRUNING
                    # minSize=(10, 10),
                    # maxSize=(500, 500)
                )
            else:
                drones = self._drone_cascade.detectMultiScale(
                    img,
                    minNeighbors=1,
                    flags=cv2.CASCADE_DO_CANNY_PRUNING,
                    minSize=(last_size[0] * DroneRecognizer.SIZE_CONFIDENCE,
                             last_size[1] * DroneRecognizer.SIZE_CONFIDENCE),
                    maxSize=(last_size[0] * (2 - DroneRecognizer.SIZE_CONFIDENCE),
                             last_size[1] * (2 - DroneRecognizer.SIZE_CONFIDENCE))
                )
            tock = time.time()

            times.append(tock - tick)
            print("Detection time:\t\t", str(times[-1]))
            if len(drones) > 0:
                best_drone = max(drones, key=lambda drone: drone[1])
                ((x, y, w, h), confidence) = best_drone
                last_size = (w, h)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                last_size = None
            if viz:
                scale_percent = 30  # percent of original size
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

                cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        print("Average detection time:\t", str(np.mean(times)))
