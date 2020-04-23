import cv2
import os


class DroneRecognizer(object):
    DEFAUlT_XML_PATH = "./detection/cascades/haarcascade_drones.xml"

    def __init__(self, xml_file=DEFAUlT_XML_PATH, trained=True):
        self._drone_cascade = None

        if trained:
            self._drone_cascade = cv2.CascadeClassifier(xml_file)
            print("CASCADE OBJ:\t", self._drone_cascade)

    # Currently done using outside program
    # https://amin-ahmadi.com/cascade-trainer-gui/
    def train(self, im_path="./train_images/drones"):
        pass

    def test(self, im_path="./test_images/drones"):
        img_paths = [im_path + path for path in os.listdir(im_path)]
        print("TESTING ON:\t\t", img_paths)

        imgs = [cv2.imread(path) for path in img_paths]
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

        print(imgs)

        for img in imgs:
            drones = self._drone_cascade.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in drones:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()





