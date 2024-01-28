import cv2
import numpy as np
import requests
from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread('dworzec.jpg')
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))

        return {'count': len(boxes)}


class PeopleCounter_URL(Resource):
    def get(self):
        image_url = \
            ('https://rzeszow-news.pl/wp-content/uploads/'
             'sites/1/nggallery/'
             'prezentacja-dworzec-pks-pazdziernik-2021/'
             'pks-03-wiz-1920x1080.jpg')

        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image_url = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        boxes, weights = hog.detectMultiScale(image_url, winStride=(8, 8))

        return {'count': len(boxes)}


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


api.add_resource(PeopleCounter_URL, '/b')
api.add_resource(PeopleCounter, '/')
api.add_resource(HelloWorld, '/test')

if __name__ == '__main__':
    app.run(debug=True)
