import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import re
from time import time
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import normalize
base_path = "C:/Users/shahid/PycharmProjects/random stuff/"
assets = os.listdir(base_path + 'assets/')

detector = MTCNN()
model = load_model(base_path + 'facenet_keras.h5')
# model.summary()

def extract_face(filename, required_size=(160, 160), is_frame=False, return_coords=False):
	"""This method extracts face from the image:filename using the pretrained MTCNN model.
	is_frame: if its a frame from live feed then there is no file to read but if its not a frame file reading
	 abilities are availible.
	 return_coords: returns the coordinates of the face"""
    if is_frame:
        pixels = cv2.resize(filename, (160, 160))
    else:
        # load image from file
        image = Image.open(filename)
        # convert to RGB, if needed
        # image = image.convert('RGB')
        # convert to array
        pixels = np.asarray(image)
    
    # detect faces in the image
    results = detector.detect_faces(pixels)

    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    if return_coords:
        x1, y1 = abs(x1)/160.0, abs(y1)/160.0
        x2, y2 = x1 + width/160.0, y1 + height/160.0
        return face_array, ((x1, y1), (x2, y2))
    return face_array

def extract_faces(filename, required_size=(160, 160), is_frame=False, return_coords=False):
	"""This method extracts all the faces from the image:filename using the pretrained MTCNN model."""
    if is_frame:
        pixels = cv2.resize(filename, (160, 160))
    else:
        # load image from file
        image = Image.open(filename)
        # convert to RGB, if needed
        # image = image.convert('RGB')
        # convert to array
        pixels = np.asarray(image)
    
    # detect faces in the image
    results = detector.detect_faces(pixels)
    faces_array = []
    faces_array_coords = []

    # extract the bounding box from the first face
    for i in range(len(results)):
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)

        x1, y1 = abs(x1)/160.0, abs(y1)/160.0
        x2, y2 = x1 + width/160.0, y1 + height/160.0

        faces_array_coords.append([(x1, y1), (x2, y2)])
        faces_array.append(face_array)

    if return_coords:
        return faces_array, faces_array_coords
    return faces_array

def img_to_encoding(image_path):
	"""This face recognition system is built on the siamese network architecture,
	 this is the function that converts an image into its representation"""
    loaded = load_img(image_path)
    resized = loaded.resize([160, 160])
    resized = np.array(resized)
    resized = tf.expand_dims(resized, 0)

    encoding = tf.squeeze(model(resized))

    return encoding

database = {'shahid':img_to_encoding(base_path + "assets/shahid.jpg")}  # In this DB one person has only one pic

list_style_db = {'shahid':[img_to_encoding(base_path + 'assets/' + file) for file in assets if re.search(r"shahid_\d*", file)]}
                #  In this DB everyone can have multiple faces to compare to.



def verify(image_path, identity, is_frame=False, threshold=4.0):
	"""Checks the given face against the identity image from database"""
    if is_frame:
        resized = cv2.resize(image_path, (160, 160))
        resized = tf.expand_dims(resized, 0)
        encoding = model(resized)
    else:
        encoding = img_to_encoding(image_path)
    dist = np.linalg.norm(database[identity] - encoding)

    if dist <= threshold:
        # print("It's " + str(identity) + ", welcome in!")
        door_open = True
    else:
        # print("It's not " + str(identity) + ", please go away")
        door_open = False
    
    return dist, door_open

def ensembled_verify(image_path, identity, is_frame=False, threshold=4.0):
	"""Checks the given face against the identity images from list_style_db"""
    if is_frame:
        resized = cv2.resize(image_path, (160, 160))
        resized = tf.expand_dims(resized, 0)
        encoding = model(resized)
    else:
        encoding = img_to_encoding(image_path)

    distances = [np.linalg.norm(list_style_db[identity][i] - encoding) for i in range(len(list_style_db[identity]))]
    dist = np.mean(distances)

    if dist <= threshold:
        print("It's " + str(identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
    
    return dist, door_open

def face_recognition(face_frame, threshold=2.5):
	"""Checks the given face against all faces in the list_style_db to determine identity of the face"""
    frame_resized = cv2.resize(face_frame, (160, 160))
    frame_encoding = model(tf.expand_dims(frame_resized, 0))
    all_distances = {}
    for identity, encodings in list_style_db.items():
        
        distances = [np.linalg.norm(encodings[i] - frame_encoding) for i in range(len(encodings))]
        dist = np.mean(distances)
        all_distances[dist] = identity	

    dist, identity = np.min(list(all_distances.keys())), all_distances[np.min(list(all_distances.keys()))]
    print(all_distances)
    if dist <= threshold:
        # print("It's " + str(identity) + ", welcome in!")
        door_open = True
        face_id = identity
    else:
        # print("It's not " + str(identity) + ", please go away")
        face_id = ''
        door_open = False

    return face_id, door_open, dist

def main():
    camera = cv2.VideoCapture(1)  # 1 means its capturing device: 1, 0 means device: 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        while True:
            _, frame = camera.read()
            # print(frame.shape)
            # quit()  

            if cv2.waitKey(1) & 0xFF == ord('q'):
            	# Quits the app
                cv2.destroyAllWindows()
                camera.release()
                break
            elif cv2.waitKey(33) == ord('c'):
            	# Adds a new face to the list_style_db for the following person
                f_a = extract_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), is_frame=True)
                cv2.imwrite(base_path + f'assets/shahid_{int(time())}.jpg', cv2.cvtColor(f_a, cv2.COLOR_RGB2BGR))
            elif cv2.waitKey(33) == ord('d'):
            	# Performs proper face recognition on 160x160, then scales the coords returned to width x height (640, 480)
                try:
                    face, face_coords = extract_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), is_frame=True, return_coords=True)
                    coords1 = (int(face_coords[0][0] * 640.0), int(face_coords[0][1] * 480.0))
                    coords2 = (int(face_coords[1][0] * 640.0), int(face_coords[1][1] * 480.0))
                    cv2.rectangle(frame, coords1, coords2,(0,255,0),3)
                    face_id, db_check, dist = face_recognition(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), threshold=4.0)
                    print(face_id, dist, db_check)
                    if db_check:
                        cv2.putText(frame, face_id,(coords1[0] - 15, coords1[1] - 15), font, 1, (200,255,155), 2, cv2.LINE_AA)
           
                except IndexError:
                    print("IndexError Occured")
                        
            cv2.imshow('live', frame)

    finally:
        cv2.destroyAllWindows()
        camera.release()


main()

