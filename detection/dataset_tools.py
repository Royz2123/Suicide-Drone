import cv2
import matplotlib.pyplot as plt
import numpy as np

import os


def create_dir(dir_path):
    try:
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
            print("Successfully created the directory %s " % dir_path)
        else:
            print("Creation of the directory %s succeeded, because it's already exists Toar..." % dir_path)
    except OSError:
        print("Creation of the directory %s failed" % dir_path)


class DroneDataset:
    SAMPLE_FREQUENCY = 20

    def __init__(self):
        self.PARSE_POS = False
        self.PARSE_NEG = True

        self._base = ".\\train_images\\drones\\"
        self._directory = self._base + "raw_data\\"

        self._positive_directory = self._directory + "positives\\"
        self._negative_directory = self._directory + "negatives\\"

        self._final_dir = self._base + "final_images\\"
        self._cropped_dir_name = "p\\"
        self._cropped_dir = self._final_dir + self._cropped_dir_name
        self._seperated_dir_name = "n\\"
        self._seperated_dir = self._final_dir + self._seperated_dir_name
        self._seperated_dir = self._final_dir + self._seperated_dir_name

        self._augmented_dir = self._base + "augmented\\"

        create_dir(self._cropped_dir)
        create_dir(self._seperated_dir)
        create_dir(self._augmented_dir)

    def extract_rect(self, line):
        size_str = line.split(" ")
        sizes = [int(s) for s in size_str]
        return sizes[0], sizes[1], sizes[2], sizes[3]

    def edit_file(self, mov_file, txt_file, save_dir):
        data_name = mov_file.split(".")[-2].split("\\")[-1]
        vidcap = cv2.VideoCapture(mov_file)
        success, image = vidcap.read()
        count = 0
        with open(txt_file, 'r') as txt:
            lines = txt.readlines()

        while success:
            if 3 * count >= len(lines):
                print("The data of %s is not complete it has %d labeled images but %d images" % (
                    data_name, count, len(lines) / 3))
                break
            x, y, w, h = self.extract_rect(lines[count * 3 + 2][:-1])
            crop_img = image[y:y + h, x:x + w]
            gray_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(save_dir + "\\" + data_name + "_frame%d.jpg" % count,
                        gray_image)  # save frame as JPEG file
            success, image = vidcap.read()
            print("Read frame: %d from %s" % (count, data_name))
            count += 1

    def ask_remove_duplicates(self, directory):
        remove_duplicates = input(
            "duplicate images were found, delete all duplicates? (you can erase %s and run the script again)[y/n]" % directory)
        if remove_duplicates == "y":
            used_names = []
            for filename in os.listdir(directory):
                if filename in used_names:
                    os.remove(directory + filename)
                    print("Removed %s" % filename)
                else:
                    used_names.append(filename)
        else:
            print("Ok, Toar...")

    def parse_positive(self):
        # POSITIVE HANDLING
        if self.PARSE_POS:
            edited_files = [self._cropped_dir_name]
            for filename in os.listdir(self._positive_directory):
                data_name = filename.split(".")[0]
                if not (filename.endswith(".txt") or filename.endswith(".MOV") or filename == self._cropped_dir_name):
                    print("Toar you stupid boy, %s isn't .MOV or .txt file, fix your bugs motherfucker" % data_name)
                    break
                if data_name in edited_files:
                    continue
                edited_files.append(data_name)
                self.edit_file(self._positive_directory + data_name + ".MOV",
                               self._positive_directory + data_name + ".txt", self._cropped_dir)

    def seperate_frames(self, mov_file, save_dir):
        data_name = mov_file.split(".")[-2].split("\\")[-1]
        vidcap = cv2.VideoCapture(mov_file)
        success, image = vidcap.read()
        count = 0

        while success:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if count % DroneDataset.SAMPLE_FREQUENCY == 0:
                print("Read frame: %d from %s" % (count, data_name))
                cv2.imwrite(save_dir + "\\" + data_name + "_frame%d.jpg" % count,
                            gray_image)  # save frame as JPEG file

            success, image = vidcap.read()

            count += 1

    def dilute_directory(self, dir_path=None):
        if dir_path is None:
            dir_path = self._cropped_dir

        for index, filename in enumerate(os.listdir(dir_path)):
            if filename.endswith(".jpg"):
                if index % DroneDataset.SAMPLE_FREQUENCY:
                    # print(f"Removed index {index}, filename: {filename}")
                    os.remove(dir_path + filename)
                else:
                    print(f"Kept index {index}, filename: {filename}")

    def parse_negative(self):
        if self.PARSE_NEG:
            for filename in os.listdir(self._negative_directory):
                if filename.endswith(".mp4"):
                    self.seperate_frames(self._negative_directory + filename, self._seperated_dir)
                elif filename == self._seperated_dir_name:
                    continue
                else:
                    print("All files should be mp4 files %s is not, Toar stop being an idiot" % filename)

    # assumes train_images/drones/ exists, and create an identical edge dataset
    def create_edge_dataset(self):
        edges_directory = self._base + "edges\\"
        edges_p = edges_directory + "p\\"
        edges_n = edges_directory + "n\\"

        create_dir(edges_directory)
        create_dir(edges_p)
        create_dir(edges_n)

        self.edgify(self._cropped_dir, edges_p)
        self.edgify(self._seperated_dir, edges_n)

    def edgify(self, input_dir, output_dir):
        for path in os.listdir(input_dir):
            img = cv2.imread(input_dir + path, 0)

            # edge detection
            v = np.median(img)
            sigma = 1.0

            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            img = cv2.Canny(img, lower, upper)

            cv2.imshow("test", img)
            cv2.waitKey(0)

            kernel = np.ones((2, 2), np.uint8)
            img = cv2.dilate(img, kernel, iterations=1)

            # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            cv2.imshow("test", img)
            cv2.imwrite(output_dir + path, img)

    # Augmentation functions

    @staticmethod
    def get_edge(image):
        edges = cv2.Canny(image, 100, 200)
        closing = cv2.morphologyEx(edges, kernel=None, op=cv2.MORPH_CLOSE, iterations=2)
        return closing

    @staticmethod
    def get_contours(img):
        contours, _ = cv2.findContours(img, mode=cv2.RETR_CCOMP,
                                       method=cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def augmentation_method_1(image):
        edges = DroneDataset.get_edge(image)
        contours = DroneDataset.get_contours(edges)

        for contour in contours:
            cv2.fillPoly(edges, pts=np.int32([contour]), color=(255, 255, 255))

        black_image = image
        black_image[edges != 0] = 255 - black_image[edges != 0]

        return black_image

    @staticmethod
    def augmentation_method_2(image):
        return 255 - image

    @staticmethod
    def augmentation_method_3(image):
        return 255 - image

    def create_auged_dataset(self):
        METHODS = [
            (lambda x: x, "non_method"),
            # (DroneDataset.augmentation_method_1, "ron_method"),
            (DroneDataset.augmentation_method_2, "inv_method"),
            # (DroneDataset.augmentation_method_3, "test_method"),
        ]

        for filename in os.listdir(self._cropped_dir):
            if filename.endswith(".jpg"):
                print(filename)

                image = cv2.imread(self._cropped_dir + filename, 0)

                for method, name in METHODS:
                    auged_image = method(image.copy())
                    cv2.imwrite(self._augmented_dir + name + "_" + filename, auged_image)

                    # cv2.imshow("test", auged_image)
                    # cv2.waitKey(0)
