import os
import time
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from ultralytics import YOLO
from google.cloud import vision
from flask import Flask

key = "078e943cda2145bf9866e5fe8668faa6"
endpoint = "https://other-apis.cognitiveservices.azure.com/"
computerVision = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'graduation-project-379520-244a3ffc507c.json'
client = vision.ImageAnnotatorClient()
image = vision.Image()
BlindEyeVision = Flask(__name__)


@BlindEyeVision.route("/image-description")
def imageDescription():
    text = "It's "
    image = "https://www.jll.pt/images/people/people-photography/privacy-in-the-open-plan-office.jpg"
    desc = computerVision.describe_image(image)
    for caption in desc.captions:
        text = text + caption.text
    return text


@BlindEyeVision.route("/ocr")
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
            text = text + result.text + "\n"
    return text


@BlindEyeVision.route("/object-detection")
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
    return text


@BlindEyeVision.route("/landmark-detection")
def landmarkDetection():
    text = ""
    image.source.image_uri = 'https://dynamic-media-cdn.tripadvisor.com/media/photo-o/15/4f/38/f4/caption.jpg?w=1200&h=-1&s=1'
    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations
    if len(landmarks) == 1:
        text = "Landmark is "
    elif len(landmarks) == 1:
        text = "Try again."
        return text
    else:
        text = "Landmarks are "
    for landmark in landmarks:
        text += landmark.description + ", "
    return text


@BlindEyeVision.route("/currency-detection")
def currencyDetection():
    text = ""
    model = YOLO("best.pt")
    results = model('https://cloudfront-eu-central-1.images.arcpublishing.com/thenational/UNYSRZ52VJGC5BN2Y327UDVQVA.jpg')
    for result in results:
       for label in result.boxes.cls:
           if model.names[int(label)] == 1:
                text = text + model.names[int(label)] + " pound, "
           else:
                text = text + model.names[int(label)] + " pounds, "
    return text


@BlindEyeVision.route('/face-detection')
def faceDetection():
    text = ""
    anger = 0
    joy = 0
    superise = 0
    sorrow = 0
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
    return text

@BlindEyeVision.route('/logo-detection')
def logoDetection():
    text = ""
    image.source.image_uri = "https://i.ebayimg.com/images/g/nYYAAOSwfftijJsK/s-l1600.jpg"
    response = client.logo_detection(image=image)
    logos = response.logo_annotations
    if len(logos) == 1:
        text = "Logo is "
    elif len(logos) > 1:
        text = "Logos are "
    for logo in logos:
        text += logo.description + ", "
    return json.dumps(text)


if __name__ == "__main__":
    BlindEyeVision.run(debug=True)
