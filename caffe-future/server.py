import socket
import threading
import time
import sys
sys.path.append('/home/ubuntu/pynb/caffe-future')
import inference
import solve
import os
import numpy as np
import shutil

rootpath = '/home/ubuntu/pynb/caffe-future/'
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('0.0.0.0', 8001))
sock.listen(5)
GPUAVAIL = np.array([0, 1], dtype = 'int')
condition = threading.Condition()

class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.robotid = '0000'
        self.robottrain = True
        self.gpuid = 0
        
    def run(self):
        global GPUAVAIL, condition
        if condition.acquire():
            if GPUAVAIL.size == 0:
                condition.wait()
            self.gpuid = GPUAVAIL[0]
            GPUAVAIL = np.delete(GPUAVAIL,0,0)
            condition.release()
            
        if self.robottrain:
            while not (os.path.exists(rootpath + self.robotid + '/' + self.robotid + '_' + 'flag_train.txt')):
                pass
            solve.solverobot(self.robotid, self.gpuid)
            os.remove(rootpath + self.robotid + '/'+ self.robotid + '_' + 'flag_train.txt')
        f = open(rootpath +  self.robotid + '/'+self.robotid + '_' + 'flag_continue.txt','w')
        f.close()
        inference.inferrobot(self.robotid, self.gpuid)
        os.remove(rootpath + self.robotid + '/'+ self.robotid + '_' + 'flag_end.txt')
        print "I am %s, my ROBOTID is %s" %(self.name, self.robotid)
        #os.remove(rootpath + ROBOTID + '_' + 'flag_continue.txt')
        
        if condition.acquire():
            GPUAVAIL = np.append(GPUAVAIL, self.gpuid)
            condition.notify()
        condition.release()
            
while True:
    connection, address = sock.accept()
    try:
        connection.settimeout(5)
        buf = connection.recv(1024)
        ROBOTID = buf[:-1]
        ROBOTTRAIN = buf[-1]
        if ROBOTTRAIN == '1':
            ROBOTTRAIN = True
        else:
            ROBOTTRAIN = False
        connection.send('ID Got, Please Go Ahead :)')
        my_thread = MyThread()
        my_thread.robotid = ROBOTID
        my_thread.robottrain = ROBOTTRAIN
        if not os.path.exists(ROBOTID):
            shutil.copytree('0000', ROBOTID)
            f = open(ROBOTID + '/solver.prototxt', 'r+')
            s = f.read()
            f.seek(0,0)
            f.write(s.replace("ROBOTID",ROBOTID))
            f.close()
            f = open(ROBOTID + '/solver.prototxt', 'r+')
            s = f.readlines()
            del s[-1]
            f.seek(0,0)
            f.truncate()
            f.writelines(s)
            f.close()
            
            f = open(ROBOTID + '/train.prototxt', 'r+')
            s = f.read()
            f.seek(0,0)
            f.write(s.replace("ROBOTID",ROBOTID))
            f.close() 
            f = open(ROBOTID + '/train.prototxt', 'r+')
            s = f.readlines()
            del s[-1]
            f.seek(0,0)
            f.truncate()
            f.writelines(s)
            f.close()
        my_thread.start()
        
    except socket.timeout:
        connection.send('Timeout :(')
    connection.close()
        

