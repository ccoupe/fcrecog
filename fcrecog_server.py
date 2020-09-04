# PyRpc Server for face regconigtion on Tcp port 4774
# This will be started by systemd.
import cv2
import numpy as np
import imutils
import sys
import json
import argparse
import warnings
from datetime import datetime
import time,threading, sched
import rpyc
#from lib.Algo import Algo
import logging
import face_recognition
import os
import os.path

debug = False;

KNOWN_FACES_DIR = '/home/ccoupe/fcrecog/known_faces'
UNKNOWN_FACES_DIR = '/home/ccoupe/fcrecog/unknown_faces'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model
known_faces = []
known_names = []

def init_models():
  # We oranize known faces as subfolders of KNOWN_FACES_DIR
  # Each subfolder's name becomes our label (name)
  global log, known_faces, known_names
  for name in os.listdir(KNOWN_FACES_DIR):
  
    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
      log.info('working on {}/{}'.format(name,filename))
      # Load an image
      image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        
      # Get 128-dimension face encoding
      # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
      try:
        encoding = face_recognition.face_encodings(image)[0]
        
        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)
      except:
        log.info(f"can't find a face in {name}/{filename}")
      
def update_models(name, image):
  global log, known_faces, known_names
  td = f'{KNOWN_FACES_DIR}/{name}' 
  if not os.path.exists(td):
    os.mkdir(td)
  fp = f'{td}/{name}.jpg'
  f = open(fp, 'wb')
  f.write(image)
  f.close()
  log.info(f'created {fp}')
 
  img = face_recognition.load_image_file(fp)
  try:
    encoding = face_recognition.face_encodings(img)[0]
    # Append encodings and name
    known_faces.append(encoding)
    known_names.append(name)
    log.info(f'updated running models')
  except:
    log.info(f'is there a face in {fp}')

  

class Settings:

  def __init__(self, logw):
    self.log = logw
    self.use_ml = None


class MyService(rpyc.Service):  
  
  def on_connect(self, conn):
    self.client_ip, _ = conn._config['endpoints'][1]
   
  def exposed_face_recog(self, imagestr):
    global log
    log.info('called')
    o = open("/tmp/face.jpg","wb")
    o.write(imagestr)
    o.close()
    # convert imagestr arg to cv2/numpy image. it's a jpeg
    #parr = np.fromstring(imagestr, np.uint8)
    #image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = face_recognition.load_image_file("/tmp/face.jpg")
    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(image, model=MODEL)
    
    # Now since we know loctions, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    encodings = face_recognition.face_encodings(image, locations)
    
    # We passed our image through face_locations and face_encodings, so we can modify it
    # First we need to convert it from RGB to BGR as we are going to work with cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
    log.info(f', found {len(encodings)} face(s)')
    match = None
    for face_encoding, face_location in zip(encodings, locations):
      
      # We use compare_faces (but might use face_distance as well)
      # Returns array of True/False values in order of passed known_faces
      results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
      
      # Since order is being preserved, we check if any face was found then grab index
      # then label (name) of first matching known face withing a tolerance
      if True in results:  # If at least one is true, get a name of first of found labels
          match = known_names[results.index(True)]
          log.info(f' - {match} from {results}')
          break
          
    return match
    
  def exposed_save_recog(self, name, imagestr):
    update_models(name, imagestr)
    
# process args - port number, 
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--port", action='store', type=int, default='4774',
  nargs='?', help="server port number, 4774 is default")
args = vars(ap.parse_args())

# logging setup
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')
log = logging.getLogger('ML_Shapes')

settings = Settings(log)

# load the pre-computed models...
init_models()

from rpyc.utils.server import ThreadedServer
t = ThreadedServer(MyService, port = args['port'])
t.start()
