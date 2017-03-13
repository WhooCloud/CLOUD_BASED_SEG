import socket
import threading
import time
import sys
sys.path.append('/home/ubuntu/pynb/caffe-future')
import inference
import solve
import os

rootpath = '/home/ubuntu/pynb/'
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('0.0.0.0', 8001))
sock.listen(5)
GPUAVAIL = [0, 1, 2, 3]
class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.robotid = '0000'
        self.robottrain = True
        self.gpuid = 0
    def run(self):
        if self.robottrain:
            while not (os.path.exists(rootpath + self.robotid + '_' + 'flag_train.txt')):
                pass
            solve.solverobot(self.robotid, self.gpuid)
            os.remove(rootpath + self.robotid + '_' + 'flag_train.txt')
        f = open(rootpath + self.robotid + '_' + 'flag_continue.txt','w')
        f.close()
        inference.inferrobot(self.robotid, self.gpuid)
        os.remove(rootpath + self.robotid + '_' + 'flag_end.txt')
        print "I am %s, my ROBOTID is %s" %(self.name, self.robotid)
        #os.remove(rootpath + ROBOTID + '_' + 'flag_continue.txt')
        
        
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
        my_thread.start()
        my_thread.join()
        
    except socket.timeout:
        connection.send('Timeout :(')
    connection.close()
        
