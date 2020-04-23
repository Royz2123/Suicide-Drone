import time

import detection.recognizer as recognizer
import prediction.tester as tester



FACE_XML = "./detection/test_cascades/haarcascade_frontalface_alt.xml"
WATCH_XML = "./train_images/watches/classifier/cascade.xml"


def detect():
    recog_obj = recognizer.DroneRecognizer(xml_file=WATCH_XML)
    recog_obj.test(im_path="./test_images/watches/")


def predict():
    tester_obj = tester.Tester(test_name="test_abel")
    tester_obj.test_all_predictors()

    # OLD FORMAT
    # tester_obj.generate_test_file()
    # time.sleep(15)
    # tester_obj.check_test_file()


if __name__ == "__main__":
    predict()