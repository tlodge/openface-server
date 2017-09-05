#!/usr/bin/env python2
import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))
from os import listdir
from os.path import isfile, isdir, join
#import txaio
#txaio.use_twisted()

from autobahn.twisted.websocket import WebSocketServerProtocol, \
	WebSocketServerFactory
from twisted.internet import reactor,task, protocol, defer
from twisted.internet.ssl import DefaultOpenSSLContextFactory
from twisted.protocols.basic import NetstringReceiver
from twisted.python import log

import argparse
#import cv2
#import imagehash
import json
from PIL import Image
import numpy as np
import os
import StringIO
#import urllib
import base64

#from sklearn.decomposition import PCA
#from sklearn.grid_search import GridSearchCV
#from sklearn.manifold import TSNE
#from sklearn.svm import SVC
#from sklearn.mixture import GMM

#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

import openface
import pickle

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
# For TLS connections
tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
tls_key = os.path.join(fileDir, 'tls', 'server.key')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
					default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
					default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
					help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
					help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9000,
					help='WebSocket Port')

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)

PORT = 9001

class Face:

	def __init__(self, rep, identity):
		self.rep = rep
		self.identity = identity

	def __repr__(self):
		return "{{id: {}, rep[0:5]: {}}}".format(
			str(self.identity),
			self.rep[0:5]
		)

class MyServer(NetstringReceiver):


	def __init__(self):
		self.le = None
		self.clf = None

		classifierModel = join(fileDir, "features", "classifier.pkl")

		with open(classifierModel, 'rb') as f:
			if sys.version_info[0] < 3:
				(self.le, self.clf) = pickle.load(f)
			else:
				(self.le, self.clf) = pickle.load(f, encoding='latin1')

		
				
	def connectionMade(self):
		print >>sys.stderr, "connected!"
		try:
		   self.transport.setTcpKeepAlive(1)
		except AttributeError: pass

	def stringReceived(self, string):
		#raw = string.decode('utf8')
		msg = json.loads(string)
		#try:
		self.processFrame(json.loads(msg["payload"])["dataURL"],-1);
		#except:
		#	print "failed to process frame"
		#	pass

	def connectionLost(self, reason):
		print >>sys.stderr,  "closed connection"

	def infer(self, rep):
		predictions = self.clf.predict_proba(rep.reshape(1,-1)).ravel()
		maxI = np.argmax(predictions)
		person = self.le.inverse_transform(maxI)
		confidence = predictions[maxI]
		print >> sys.stderr, "Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence)
		return {"name": person.decode('utf-8'), "confidence":"{:.2f}".format(confidence)}
				
	def processFrame(self, dataURL, identity):
		
		head = "data:image/jpeg;base64,"
		height = 300
		width  = 400
		assert(dataURL.startswith(head))
		imgdata = base64.b64decode(dataURL[len(head):])
		imgF = StringIO.StringIO()
		imgF.write(imgdata)
		imgF.seek(0)
		img = Image.open(imgF)

		buf = np.fliplr(np.asarray(img))
		rgbFrame = np.zeros((height, width, 3), dtype=np.uint8)
		rgbFrame[:, :, 0] = buf[:, :, 2]
		rgbFrame[:, :, 1] = buf[:, :, 1]
		rgbFrame[:, :, 2] = buf[:, :, 0]


		identities = []

		bbs = align.getAllFaceBoundingBoxes(rgbFrame)
		#bb = align.getLargestFaceBoundingBox(rgbFrame)
		bbs = bbs if bbs is not None else []

		for bb in bbs:
			
			landmarks = align.findLandmarks(rgbFrame, bb)
			alignedFace = align.align(args.imgDim, rgbFrame, bb, landmarks=landmarks, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
			
			if alignedFace is None:
				continue

			rep = net.forward(alignedFace)
			person = self.infer(rep)
			identities.append({"name":person["name"], "confidence":person["confidence"], "point":{"cx": round((bb.left() + (bb.right()-bb.left())/float(2)) / float(width),2),  "cy": round((bb.top() + (bb.bottom()-bb.top())/float(2.0))/float(height),2), "r":round(((bb.bottom()-bb.top())/float(2))/float(height),2)}});
			
		if len(identities) > 0:
			#msg = identities
			print >>sys.stderr, identities
			self.sendString(json.dumps(identities))

class MyServerFactory(protocol.Factory):
	def buildProtocol(self, addr):
		return MyServer()

def main(reactor):
	#processImages()
	factory = MyServerFactory()
	reactor.listenTCP(PORT, factory)
	return defer.Deferred()

if __name__ == '__main__':
	task.react(main)