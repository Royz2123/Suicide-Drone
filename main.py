import time

import detection.recognizer as recognizer
import prediction.tester as tester

import detection.dataset_tools as dataset_tools


FACE_XML = "./detection/test_cascades/haarcascade_frontalface_alt.xml"
WATCH_XML = "./train_images/watches/classifier/cascade.xml"


def create_dataset():
    dataset = dataset_tools.DroneDataset()
    # dataset.create_edge_dataset()
    dataset.dilute_directory()

def detect():
    recog_obj = recognizer.DroneRecognizer()
    recog_obj.test()


def predict():
    tester_obj = tester.Tester(test_name="test_abel")
    tester_obj.test_all_predictors()

    # OLD FORMAT
    # tester_obj.generate_test_file()
    # time.sleep(15)
    # tester_obj.check_test_file()


if __name__ == "__main__":
    create_dataset()