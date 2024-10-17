from datetime import datetime
import os
import csv
import io

class Logger(object):
    def __init__(self, logname):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
        if not os.path.exists(path) :
            os.makedirs(path)
        self.filename = os.path.join(path, logname + ".log")
        #f1 = open(self.filename, 'a')
        #print(self.filename)

    def write(self, *args, **kwargs):
        sio = io.StringIO()
        print(end='', file=sio, *args, **kwargs)
        #print(sio.getvalue())
        timestamp = datetime.utcnow().strftime("%y-%m-%d %H:%M:%S ")  # create unique directories
        f1 = open(self.filename, 'a')
        f1.write(timestamp + sio.getvalue() + "\n")
        f1.flush()
        f1.close()

    def printwrite(self, *args, **kwargs):
        sio = io.StringIO()
        print(end='', file=sio, *args, **kwargs)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S ")  # create unique directories
        print(timestamp + sio.getvalue())
        f1 = open(self.filename, 'a')
        f1.write(timestamp + sio.getvalue() + "\n")
        f1.flush()
        f1.close()
