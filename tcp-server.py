#!/usr/bin/env python2
import sys
import os
import json
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))
from twisted.internet import reactor,task, protocol, defer
from twisted.protocols.basic import NetstringReceiver

PORT = 9001

class MyServer(NetstringReceiver):

	def __init__(self):
		self.lines = []

	def connectionMade(self):
    
		print >>sys.stderr, "connected!"
		try:
		   self.transport.setTcpKeepAlive(1)
		except AttributeError: pass

	def stringReceived(self, string):
		msg=json.loads(string)
		print >>sys.stderr, "Received message of length",len(string)
		self.sendString("thankyou thankyou");

	def connectionLost(self, reason):
		print >>sys.stderr,  "I have closed connection! %s" % reason


class MyServerFactory(protocol.Factory):
	def buildProtocol(self, addr):
		return MyServer()

def main(reactor):
	print >>sys.stderr, "ok here we go!"
	factory = MyServerFactory()
	reactor.listenTCP(PORT, factory)
	return defer.Deferred()

if __name__ == '__main__':
    task.react(main)