import cv2
import os
import time
import numpy as np


class DroneRecognizer:
    DEFAUlT_XML_PATH = "./detection/cascades/drone_cascade_beta.xml"
    SIZE_CONFIDENCE = 0.7
    CHECK_ALL_FREQUENCY = 5
    CROP_SIDE_SIZE = 1

    def __init__(self, xml_file=DEFAUlT_XML_PATH, trained=True):
        self._drone_cascade = None
        self.vidcap = None
        self.last_position = None
        self.last_velocity = None
        self.last_size = None
        self.inner_count = 0
        self.crop_location = None
        self.x_right, self.x_left, self.y_bottom, self.y_up = 0, 0, 0, 0

        if trained:
            self._drone_cascade = cv2.CascadeClassifier(xml_file)
            print("CASCADE OBJ:\t", self._drone_cascade)

    # Currently done using outside program
    # https://amin-ahmadi.com/cascade-trainer-gui/
    def train(self, im_path="./train_images/drones"):
        pass

    def get_image_from_path(self, path):
        not_next_path = False
        img = None

        if path.endswith(".jpg") or path.endswith(".png") or path.endswith(".tiff"):
            img = cv2.imread(path, 0)
        elif path.endswith(".MOV") or path.endswith(".MP4") or path.endswith(".avi"):
            if self.vidcap is None:
                self.vidcap = cv2.VideoCapture(path)

            not_next_path, img = self.vidcap.read()

            if not not_next_path:
                self.vidcap = None
        else:
            print(
                "TOAR STOP BEING STUPID, YOU HAVE SOMETHING IN " + path +
                " THAT IS NOT PNG/TIFF/JPG/MP4/MOV")
        return not_next_path, img

    def get_min_size(self):
        minSize = (int(self.last_size[0] * DroneRecognizer.SIZE_CONFIDENCE),
                   int(self.last_size[1] * DroneRecognizer.SIZE_CONFIDENCE))
        return minSize

    def get_max_size(self):
        maxSize = (int(self.last_size[0] * (2 - DroneRecognizer.SIZE_CONFIDENCE)),
                   int(self.last_size[1] * (2 - DroneRecognizer.SIZE_CONFIDENCE)))
        return maxSize

    def crop_image(self, img):
        if self.last_position is None or self.last_size is None:
            self.x_right, self.x_left, self.y_bottom, self.y_up = img.shape[1], 0, img.shape[0], 0
            return img
        x, y, w, h = self.last_position[0], self.last_position[1], self.last_size[0], \
                     self.last_size[1]
        self.x_left = x - DroneRecognizer.CROP_SIDE_SIZE * w
        self.x_right = x + (DroneRecognizer.CROP_SIDE_SIZE + 1) * w
        self.y_up = y - DroneRecognizer.CROP_SIDE_SIZE * h
        self.y_bottom = y + (DroneRecognizer.CROP_SIDE_SIZE + 1) * h
        img = img[self.y_up:self.y_bottom, self.x_left:self.x_right, :]
        return img

    def get_rectangle_from_img(self, img):
        flag = cv2.CASCADE_SCALE_IMAGE
        if self.last_size is None or self.inner_count == DroneRecognizer.CHECK_ALL_FREQUENCY:
            if self.inner_count == DroneRecognizer.CHECK_ALL_FREQUENCY:
                self.inner_count = 0
            print("checked all")
            drones = self._drone_cascade.detectMultiScale3(
                img,
                flags=flag,
                scaleFactor=1.3,
                minNeighbors=1,
                outputRejectLevels=True
            )
        else:
            img = self.crop_image(img)
            self.inner_count += 1
            drones = self._drone_cascade.detectMultiScale3(
                img,
                flags=flag,
                minNeighbors=1,
                minSize=self.get_min_size(),
                maxSize=self.get_max_size(),
                outputRejectLevels=True
            )
            if len(drones[0]) > 0:
                drones[0][:, 0] += self.x_left
                drones[0][:, 1] += self.y_up
        return drones

    def update_drone_data(self, best_drone):
        (x, y, w, h) = best_drone
        if self.last_position is not None:
            last_x = self.last_position[0]
            last_y = self.last_position[1]
            self.last_velocity = (x - last_x, y - last_y)
        else:
            self.last_velocity = None
        self.last_position = (x, y)
        self.last_size = (w, h)
        return x, y, w, h

    def force_check_all(self):
        DroneRecognizer.inner_count = DroneRecognizer.CHECK_ALL_FREQUENCY

    def exterpulate_drone(self, img):
        if self.last_velocity is not None:
            current_x = self.last_position[0] + self.last_velocity[0]
            current_y = self.last_position[1] + self.last_velocity[1]
        else:
            current_x = self.last_position[0]
            current_y = self.last_position[1]
        img = cv2.rectangle(img, (current_x, current_y),
                            (current_x + self.last_size[0], current_y + self.last_size[1]),
                            (0, 0, 255), 2)
        self.force_check_all()
        print("Drone location is approximated")
        return img

    def find_best_drone(self, drones, img):
        if len(drones[2]) > 0:
            best_drone_index = np.argmax(drones[2])
            best_drone = drones[0][best_drone_index]
            x, y, w, h = self.update_drone_data(best_drone)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            img = self.exterpulate_drone(img)
        img = cv2.rectangle(img, (self.x_left, self.y_up), (self.x_right, self.y_bottom), (0, 0, 0),
                            2)
        return img

    def visalize_img(self, img):
        scale_percent = 30  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return img

    def show_img(self, img):
        cv2.imshow('img', img)
        cv2.waitKey(1)

    def create_video_from_images(self, images, file_name):
        height, width, layers = images[0].shape
        size = (width, height)
        out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        for i in range(len(images)):
            out.write(images[i])
        out.release()

    def test(self, im_path="./test_images/drones_easy/", viz=True):
        times = []
        self.last_size = None
        images_visualized = []
        img_paths = [im_path + path for path in os.listdir(im_path)]
        print("TESTING ON:\t\t", img_paths)

        for path in img_paths:
            not_next_path = True

            while not_next_path:
                not_next_path, img = self.get_image_from_path(path)

                tick = time.time()
                drones = self.get_rectangle_from_img(img)
                tock = time.time()
                times.append(tock - tick)
                print("Detection time:\t\t", str(times[-1]))
                img = self.find_best_drone(drones, img)
                if viz:
                    img = self.visalize_img(img)
                    # images_visualized.append(img)
                    self.show_img(img)


        if viz:
            self.create_video_from_images(images_visualized, im_path + "output.avi")

        print("Average detection time:\t", str(np.mean(times)))
