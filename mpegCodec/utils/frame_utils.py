# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:41:38 2015

@author: uluyac
"""

import cv2
import os
#from utils import detect_version as dtv

from Tkinter import Tk
from tkSimpleDialog import askstring

def sequence_iterator(sequence):
    
    nFrames = len(sequence)
    wName = 'Video'
    cv2.namedWindow(wName)
    
    print '\n### Sequence Iterator ###'
    print 'Number of frames: ' + str(nFrames)
    for i in range(nFrames):
        cv2.imshow('Video', sequence[i])
        cv2.waitKey(0)
        
    cv2.destroyAllWindows()
    
    
def write_sequence_frames(sequence, mode, hvsqm, video_name = ''):
    
    root = Tk()
    root.withdraw()
    
    nFrames = len(sequence)
    wName = 'Video'
    cv2.namedWindow(wName)
    
    dirName = ''
    if video_name == '':
        dirName = askstring("Directory Name", "Enter with the directory output name").__str__()
    else:
        dirName = video_name.split('/')
        dirName = dirName[-1]
        dirName = dirName.split('.')[0]
        hvs = dirName.split('_')[-1]
        
#    dirName = ''.join(e for e in dirName if e.isalnum())
    if not os.path.exists('./frames_output/'):
        os.makedirs('./frames_output/')
    if hvs == 'hvs':
        if mode == '444':
            directory = './frames_output/hvs/444/' + dirName + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)
        elif mode == '420':
            directory = './frames_output/hvs/420/' + dirName + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)
    else:
        if mode == '444':
            directory = './frames_output/normal/444/' + dirName + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)
        elif mode == '420':
            directory = './frames_output/normal/420/' + dirName + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)
    extension = '.png'
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    print '\n### Writing frames ###'
    print 'Number of frames: ' + str(nFrames)
    for i in range(nFrames):
        imName = directory + str(i) + extension
        print('Saving ' + imName)
        cv2.imwrite(imName, sequence[i])

#def readVideo(videoName):
#    '''
#    # MPEG Encoder:
#    Method: readVideo(self, videoName)-> sequence
#    About: This method store a video in a list.
#    '''
#    video = cv2.VideoCapture(videoName)
#    if(dtv.opencv_version() >= 3):
#        self.fps = video.get(cv2.CAP_PROP_FPS)
#    else:
#        self.fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
#    ret, fr = video.read()
#    sequence = []
#    sequence.append(self.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)))
#    self.oR, self.oC, self.oD = fr.shape
#
#    while ret:
#        ret, fr = video.read()
#        if ret != False:
#            sequence.append(self.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)))
#    video.release()
#    
#    self.nframes = len(sequence)
#    
#    if (len(sequence)-1)%6 != 0:
#        for i in range(6-(len(sequence)-1)%6):
#            sequence.append(sequence[-1])
#    
#    return sequence
#
#def resize (frame):
#    '''
#    # MPEG Encoder:
#    Method: resize(self, frame)-> auxImage
#    About: This method adjusts the shape of a given frame.
#    '''
#    rR, rC, rD = frame.shape
#    aR = aC = 0
#
#    if rR % self.mbr != 0:
#        aR = rR + (self.mbr - (rR%self.mbr))
#    else:
#        aR = rR
#
#    if rC % self.mbc != 0:
#        aC = rC + (self.mbc - (rC%self.mbc))
#    else:
#        aC = rC
#
#    for i in range (0,rR,2):
#        for j in range (0,rC,2):
#            frame[i+1,j,1] = frame[i,j+1,1] = frame[i+1,j+1,1] = frame[i,j,1]
#            frame[i+1,j,2] = frame[i,j+1,2] = frame[i+1,j+1,2] = frame[i,j,2]
#
#    auxImage = np.zeros((aR, aC, rD), np.float32)
#    auxImage[:rR,:rC] = frame
#
#    return auxImage

