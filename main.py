import detection.recognizer as recognizer
import prediction.tester as tester


FACE_XML = "./detection/test_cascades/haarcascade_frontalface_alt.xml"
WATCH_XML = "./train_images/watches/classifier/cascade.xml"

def detect():
    recog_obj = recognizer.DroneRecognizer(xml_file=WATCH_XML)
    recog_obj.test(im_path="./test_images/watches/")

def predict():
    tester_obj = tester.Tester()
    tester_obj.generate_test_file()


if __name__ == "__main__":
    predict()