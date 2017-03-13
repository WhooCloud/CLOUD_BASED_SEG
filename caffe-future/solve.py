import caffe
import sys
sys.path.append('/home/ubuntu/pynb/caffe-future/')
import surgery, score
import numpy as np
import os
import setproctitle
import shutil
def solverobot(robotid, gpuid):
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
    rootpath = '/home/ubuntu/pynb/caffe-future/'+ robotid + '/'
    weights = rootpath +'_iter_100.caffemodel'
    weightssavepath = rootpath +robotid+'_iter_100.caffemodel'
    caffe.set_device(gpuid)
    caffe.set_mode_gpu()

    path = rootpath + 'data/'
    dirlist = os.listdir(path + 'img/')
    f = open(path +'train.txt', 'w')
    for name in dirlist:
        if name[:4] == robotid :
            f.write(name[:-4])
            f.write("\n")
    f.close()
    # init
    solver = caffe.SGDSolver(rootpath + 'solver.prototxt')
    solver.net.copy_from(weights)
 
    # surgeries
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    surgery.interp(solver.net, interp_layers)

    # scoring
    val = np.loadtxt(path + 'train.txt', dtype=str)

    for _ in range(10):
        solver.step(10)
        acc,iu = score.seg_tests(solver, False, val, layer='score')
#             print acc,iu
#             if (acc > 0.9) and (iu > 0.9):
#                 break
#             else:
#                 solver.step(10)

    solver.net.save(weightssavepath)  

if __name__ == "__main__":
    robottrain = True
    robotid = '1001'
    if robottrain:
        solverobot(robotid,0)
