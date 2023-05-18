import json
import os
import time
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from ultralytics import YOLO
from flask import Flask, request
from google.cloud import vision

key = "078e943cda2145bf9866e5fe8668faa6"
endpoint = "https://other-apis.cognitiveservices.azure.com/"
computerVision = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'graduation-project-379520-244a3ffc507c.json'
client = vision.ImageAnnotatorClient()
image = vision.Image()
BlindEyeVision = Flask(__name__)
jsonResponse = {}


@BlindEyeVision.route("/image-description", methods=['POST'])
def imageDescription():
    text = "It's "
    url = request.get_json()['url']
    desc = computerVision.describe_image(url)
    for caption in desc.captions:
        text = text + caption.text
    jsonResponse['response'] = text
    return json.dumps(jsonResponse)


@BlindEyeVision.route("/ocr", methods=['POST'])
def ocr():
    text = ""
    url = request.get_json()['url']
    ocr = computerVision.read(url, raw=True)
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
            text = text + result.text + "\n"
    jsonResponse['response'] = text
    return json.dumps(jsonResponse)


@BlindEyeVision.route("/object-detection", methods=['POST'])
def objectDetection():
    url = request.get_json()['url']
    text = ""
    objects = []
    numOfDuplicates = {}
    noDuplicatedList = set()

    object_detect = computerVision.detect_objects(url)

    for object in object_detect.objects:
        objects.append(object.object_property)
        noDuplicatedList.add(object.object_property)
    for i in noDuplicatedList:
        numOfDuplicates[i] = objects.count(i)
    for key in numOfDuplicates:
        text = text + " " + str(numOfDuplicates[key]) + " " + key + ", "
    text = text + "in front of you"
    jsonResponse['response'] = text
    return json.dumps(jsonResponse)


@BlindEyeVision.route("/landmark-detection", methods=['POST'])
def landmarkDetection():
    text = ""
    image.source.image_uri = request.get_json()['url']
    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations
    if len(landmarks) == 1:
        text = "Landmark is "
    elif len(landmarks) == 0:
        text = "Try again later when landmarks are available."
        jsonResponse['response'] = text
        return json.dumps(jsonResponse)
    else:
        text = "Landmarks are "
    for landmark in landmarks:
        text += landmark.description + ", "
    jsonResponse['response'] = text
    return json.dumps(jsonResponse)


@BlindEyeVision.route("/currency-detection", methods=['POST'])
def currencyDetection():
    text = ""
    model = YOLO("best.pt")
    results = model(request.get_json()['url']);
    for result in results:
       for label in result.boxes.cls:
           if model.names[int(label)] == 1:
                text = text + model.names[int(label)] + " pound, "
           else:
                text = text + model.names[int(label)] + " pounds, "
    jsonResponse['response'] = text
    return json.dumps(jsonResponse)


@BlindEyeVision.route('/face-detection', methods=['POST'])
def faceDetection():
    text = ""
    anger = 0
    joy = 0
    superise = 0
    sorrow = 0
    image.source.image_uri = request.get_json()['url']
    response = client.face_detection(image=image)
    faces = response.face_annotations
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')

    if len(faces) == 1:
        text += "There is one person, "
    else:
        text = text + "There are " + str(len(faces)) + " persons. "

    for face in faces:
        if likelihood_name[face.anger_likelihood] == 'VERY_LIKELY' or likelihood_name[
            face.anger_likelihood] == 'LIKELY':
            anger += 1
        if likelihood_name[face.joy_likelihood] == 'VERY_LIKELY' or likelihood_name[face.joy_likelihood] == 'LIKELY':
            joy += 1
        if likelihood_name[face.surprise_likelihood] == 'VERY_LIKELY' or likelihood_name[face.surprise_likelihood] == 'LIKELY':
            superise += 1
        if likelihood_name[face.sorrow_likelihood] == 'VERY_LIKELY' or likelihood_name[face.sorrow_likelihood] == 'LIKELY':
            sorrow += 1

    text += "" if anger == 0 else str(anger) + " are angery, "
    text += "" if joy == 0 else str(joy) + " are happy, "
    text += "" if superise == 0 else str(superise) + " are surprised, "
    text += "" if sorrow == 0 else str(sorrow) + " are sad, "

    jsonResponse['response'] = text
    return json.dumps(jsonResponse)


@BlindEyeVision.route('/logo-detection', methods=['POST'])
def logoDetection():
    text = ""
    image.source.image_uri = request.get_json()['url']
    response = client.logo_detection(image=image)
    logos = response.logo_annotations
    if len(logos) == 1:
        text = "Logo is "
    elif len(logos) > 1:
        text = "Logos are "
    for logo in logos:
        text += logo.description + ", "
    jsonResponse['response'] = text
    return json.dumps(jsonResponse)


if __name__ == "__main__":
    BlindEyeVision.run(debug=True)
