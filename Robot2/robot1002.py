# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 20:06:35 2016

@author: Administrator
"""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import PIL 
import time
import os
import socket
from ftplib import FTP
from scipy import misc
from scipy import ndimage
from PIL import Image
from PIL import ImageFilter

ROBOTID = '1002'
ROBOTTRAIN = '0'
CARPET = 1
host = '52.53.235.53'
tcpport = 8001
user = 'ubuntu'
passwd = 'yun'
work_dir = '/home/ubuntu/pynb/caffe-future/'+ROBOTID + '/'
local_dir = 'e:/Code/Python/Robot2/'+ROBOTID+'/'
flagendpath = local_dir + ROBOTID + '_'+'flag_end.txt'
flagtrainpath = local_dir + ROBOTID + '_' + 'flag_train.txt'

def ftpUpload(name_server, path_upload):
    f = open(path_upload, 'rb')
    ftp.storbinary('STOR '+ name_server,f) # 上传FTP文件  
    f.close()

def ftpDownload(name_server, path_download):
    f = open(path_download, 'wb')
    ftp.retrbinary('RETR ' + name_server, f.write)
    f.close()

def tcpMail(host, tcp_port):
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.connect((host, tcp_port))
	print "Server Connected"
	time.sleep(3)
	sock.send(ROBOTID+ROBOTTRAIN)
	print sock.recv(1024)
	sock.close()
	
def waitFTP(ftp, waitfile):
	serverlist = ftp.nlst()      
	while (waitfile not in serverlist):
		serverlist = ftp.nlst()     

def imageProcess(inferfile):
	inference = Image.open(inferfile)
	inference = inference.filter(ImageFilter.MedianFilter(9))
	infer = np.array(inference)
	infer = infer[:,:,0]    
	
	label_im, nb_labels = ndimage.label(infer)
	sizes = ndimage.sum(infer, label_im, range(nb_labels + 1))
	maxsize = sizes.argsort()[-1]
	mask_size = np.ones(sizes.size, dtype = 'bool')
	mask_size[maxsize] = 0
	remove_pixel = mask_size[label_im]
	infer[remove_pixel] = 0
	return infer
def ftpUploadFlag(ftpflagname, ftpworkdir, localfile):
	f = open(localfile, 'wb')
	f.close()
	ftp.cwd(ftpworkdir)
	ftpUpload(ftpflagname, localfile) 
	os.remove(localfile)
	
if __name__ == '__main__':
	#连接服务器，上传ROBOTID+ROBOTTRAIN
	tcpMail(host, tcpport)
	#连接FTP
	ftp = FTP(host,user,passwd)
	ftp.cwd(work_dir)
	#初始化本地目录， trainpicdirlist为训练图片所在目录，
	#traingtdirlist为训练图片GroundTruth所在目录,
	#inferdirlist为待分割图片所在目录
	trainpicdirlist = os.listdir(local_dir + 'image_train/')
	traingtdirlist = os.listdir(local_dir + 'gt_train/')
	inferdirlist = os.listdir(local_dir + 'image_infer/')
	
	#上传训练图片及GT
	if ROBOTTRAIN == '1':
		ftp.cwd(work_dir+'data/img')
		for name in trainpicdirlist:
			ftpUpload(ROBOTID + '_' + name, local_dir + 'image_train/' + name)
			print "Uploading %s ,Done" %(name)

		ftp.cwd(work_dir+'data/cls')
		for name in traingtdirlist:
			ftpUpload(ROBOTID + '_' + name, local_dir + 'gt_train/' + name)
			print "Uploading %s ,Done" %(name)
		
		ftpUploadFlag(ROBOTID + '_' + 'flag_train.txt', work_dir, flagtrainpath)
		print 'Uploading Training Images & GT, Done'
		print 'ROBOT IS LEARNING THE PICS YOU UPLOADED :)'  
		
	#等待服务器训练结束
	ftp = FTP(host,user,passwd)
	ftp.cwd(work_dir)
	waitFTP(ftp, ROBOTID + '_' + 'flag_continue.txt')
	ftp.delete(ROBOTID + '_' + 'flag_continue.txt')
	print 'ROBOT LEARNING, DONE :)'
	
	#上传待分割图片
	ftp.cwd(work_dir)
	for name in inferdirlist:
		time1 = time.time()
		ftpUpload(ROBOTID + '_' + 'image_temp.jpg', local_dir + 'image_infer/' + name) 
		print 'time of uploading image', time.time()-time1
		ftp.rename(ROBOTID + '_' + 'image_temp.jpg',ROBOTID + '_' + 'image.jpg')
		print 'time of changing name', time.time() - time1
		#等待服务器分割结束
		waitFTP(ftp, ROBOTID + '_' + 'flag_inference.txt')
		ftp.delete(ROBOTID + '_' + 'flag_inference.txt')
		print 'time of infering', time.time() - time1
		#下载结果
		ftpDownload(ROBOTID + '_' + 'inference.jpg', local_dir + 'inference/' + name[:-4] + '.jpg')
		print 'time of downlowding', time.time() - time1
		#本地形态学处理
		infer = imageProcess(local_dir + 'inference/' + name[:-4] + '.jpg')
		image = misc.imread(local_dir + 'image_infer/' + name)
		plt.axis('off')		
		plt.subplot(1,2,1)
		plt.imshow(image)
		mask_carpet = infer == CARPET
		image[mask_carpet] = [255,0,0]
		

		plt.subplot(1,2,2)
		plt.imshow(image)
		plt.show()
		print 'time:', time.time() - time1, '\n'
	ftpUploadFlag(ROBOTID + '_' + 'flag_end.txt', work_dir, flagendpath)
	ftp.quit()