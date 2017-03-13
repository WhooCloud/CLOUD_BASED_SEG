import numpy as np
from PIL import Image
import scipy.io
import caffe
import os
import random
import time

def inferrobot(robotid, gpuid):
    rootpath = '/home/ubuntu/pynb/caffe-future/'+robotid+'/'
    deploypath = rootpath + 'deploy.prototxt'
    weightspath = rootpath +  robotid + '_iter_100.caffemodel'
    if os.path.exists(weightspath):
        pass
    else:
        weightspath = rootpath + '_iter_100.caffemodel'
    caffe.set_device(gpuid)
    caffe.set_mode_gpu()

    flagendpath = rootpath +robotid + '_'+ 'flag_end.txt'
    flaginferencepath = rootpath +robotid + '_'+'flag_inference.txt'
    imagepath = rootpath + robotid + '_'+'image.jpg'
    outimagepath = rootpath + robotid + '_'+'inference.jpg'
    net = caffe.Net(deploypath, weightspath, caffe.TEST)

    while not (os.path.exists(flagendpath) ):
        if os.path.exists(imagepath):
            im = Image.open(imagepath)
            in_ = np.array(im, dtype=np.float32)
            in_ = in_[:,:,::-1]
            in_ -= np.array((104.00698793,116.66876762,122.67891434))
            in_ = in_.transpose((2,0,1))

            net.blobs['data'].reshape(1, *in_.shape)
            net.blobs['data'].data[...] = in_

            net.forward()
            out = net.blobs['score'].data[0].argmax(axis=0)
            temp = np.array([out,out,out], dtype = 'uint8')
            temp = temp.transpose([1,2,0])
            outimage = Image.fromarray(temp)
            outimage.save(outimagepath, 'jpeg')
            f = open(flaginferencepath, 'wb')
            f.close()
            os.remove(imagepath)
        else:
            continue
if __name__ == '__main__':
    inferrobot('1003',0)