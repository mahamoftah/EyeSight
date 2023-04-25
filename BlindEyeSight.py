import os
import time
import gtts
from playsound import playsound
import pyttsx3
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from ultralytics import YOLO
from flask import Flask

key = "078e943cda2145bf9866e5fe8668faa6"
endpoint = "https://other-apis.cognitiveservices.azure.com/"
computerVision = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
BlindEyeSight = Flask(__name__)


def ttsByPYTTSX3(text):
    speech = pyttsx3.init()
    speech.setProperty('rate', 115)
    speech.say(text)
    speech.runAndWait()


def ttsByGTTS(txt):
    speech = gtts.gTTS(txt, lang="ar")
    speech.save("audio.mp3")
    playsound("audio.mp3")
    os.remove("audio.mp3")


@BlindEyeSight.route("/describe-image")
def imageDescription():
    text = "It's "
    image = "https://www.jll.pt/images/people/people-photography/privacy-in-the-open-plan-office.jpg"
    desc = computerVision.describe_image(image)
    for caption in desc.captions:
        text = text + caption.text
    ttsByPYTTSX3(text)
    return text


@BlindEyeSight.route("/ocr")
def ocr():
    text = ""
    image_url = "https://selfpublishing.com/wp-content/uploads/2020/11/How-to-Start-Writing-a-Book-700x1024.jpg"
    ocr = computerVision.read(image_url, raw=True)
    operation_location = ocr.headers["Operation-Location"]

    url_list = operation_location.split("/")
    operation_id = url_list[-1]

    while True:
        ocr_result = computerVision.get_read_result(operation_id)
        if ocr_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)
    for results in ocr_result.analyze_result.read_results:
        for result in results.lines:
            text = text + result.text
    ttsByPYTTSX3(text)
    return text


@BlindEyeSight.route("/detect-object")
def objectDetection():
    image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQv66NLKmCn3T5DV6uT0_1Hm1F1OIao6mVGNA&usqp=CAU"
    text = ""
    objects = []
    numOfDuplicates = {}
    noDuplicatedList = set()

    object_detect = computerVision.detect_objects(image_url)

    for object in object_detect.objects:
        objects.append(object.object_property)
        noDuplicatedList.add(object.object_property)
    for i in noDuplicatedList:
        numOfDuplicates[i] = objects.count(i)
    for key in numOfDuplicates:
        text = text + " " + str(numOfDuplicates[key]) + " " + key + ", "
    text = text + "in front of you"
    ttsByPYTTSX3(text)
    return text


@BlindEyeSight.route("/detect-landmark")
def landmarksDetection():
    text = ""
    image_url = "https://assets.traveltriangle.com/blog/wp-content/uploads/2019/02/Great-Sphinx-Of-Giza-Trivia.jpg"
    landmark_detect = computerVision.analyze_image_by_domain("Landmarks", image_url)
    if len(landmark_detect.result['landmarks']) == 1:
        text = text + "There is a landmark of "
    else:
        text = text + "There are landmarks of "
    for landmark in range(len(landmark_detect.result['landmarks'])):
        text = text + landmark_detect.result['landmarks'][landmark]['name'] + ", "
    ttsByPYTTSX3(text)
    return text


@BlindEyeSight.route("/detect-currency")
def currencyDetection():
    text = ""
    model = YOLO("best.pt")
    results = model('Currency-Detection-1/test/images/IMG_20230220_231743_jpg.rf.0d378477fbf3a09d479ef841ed8c8cf5.jpg')
    print(results)
    print("-----------------------------------------------------------------------------------------------")
    for result in results:
        print(result)
        print("-----------------------------------------------------------------------------------------------")
        print(result.boxes)
        print("-----------------------------------------------------------------------------------------------")
        print(result.boxes.cls)
        print("-----------------------------------------------------------------------------------------------")
        for label in result.boxes.cls:
            print(label)
            print("-----------------------------------------------------------------------------------------------")
            if model.names[int(label)] == 1:
                text = text + model.names[int(label)] + " pound, "
            else:
                text = text + model.names[int(label)] + " pounds, "
    ttsByPYTTSX3(text)
    return text


if __name__ == "__main__":
    BlindEyeSight.run(debug=True)

