# -*- coding: utf-8 -*-
"""
Created on Sat May 23 14:37:54 2015

@author: luan
"""
import cv2
import numpy as np
from frames import mpeg
from frames import MJPEGcodec as jpeg
from frames import MJPEGhvs as jpeghvs
from math import sqrt, log, exp
from utils import detect_version as dtv
#from utils.image_quality_assessment import metrics
import os

class Encoder:
    def __init__(self, videoName, quality = 50, sspace = 15, mode = '420', search = 0, hvsqm = 0, flat = 10.0, p = 2.0, dctkrnl='T'):
        '''
        # MPEG Encoder:
        Method: Constructor.
        About: This class runs the algorithm of the mpeg encoder.
        Parameters: 
            1) videoName: Video's name.
            2) quality: Quality of the image (default is 50).
            3) sspace: Search space (default is 16).
            4) mode: Down sample mode (default is 420).
            5) search: Search method. Use 0 self.outputr fullsearch and 1 self.outputr parallel hierarchical (default is 0).
        '''
        self.hvsqm = hvsqm
        self.hvstables = None
        self.p = p
        self.flat = flat
        self.oR, self.oC, self.oD = [0, 0, 0]
        self.mbr, self.mbc = [16, 16]
        self.mode = mode
        self.nframes = 0
        self.fps = 0
        self.video = self.readVideo(videoName)
        self.DCTKERNEL = dctkrnl
        self.avgBits = []    # Average amount of bits in a pixel per frame.
        self.CRate = []     # Compression rate per frame.
        self.redundancy = []    # Redundancy per frame.
        
        hvstype = ''
        if self.hvsqm==0:
            hvstype = 'normal'
        elif self.hvsqm==1:
            hvstype = 'hvs'
#        if self.DCTKERNEL=='T':
#            directory = './outputs/'+hvstype+'/'+self.mode+'/'
#            filename = videoName.split('/')[-1].split('.')[0]
#        else:
        directory = './outputs/'+hvstype+'/'+self.mode+'/'+\
        videoName.split('/')[-1].split('.')[0] + '/' + str(quality)+'/'
        filename = videoName.split('/')[-1].split('.')[0] +'-'+ self.DCTKERNEL
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        self.FILEDIR = directory
        self.FILENAME = filename
        self.output = open(directory + filename + '.txt', 'w')
        print 'Saving on: ' + directory + filename
        
        self.sspace = sspace
        self.search = search
        self.quality = quality
        self.hufftables = acdctables()
        self.Z = genQntb(self.quality)
        if self.hvsqm == 1:
            self.genHVStables()
    
    def readVideo(self, videoName):
        '''
        # MPEG Encoder:
        Method: readVideo(self, videoName)-> sequence
        About: This method store a video in a list.
        '''
        print 'Reading File....'
        video = cv2.VideoCapture(videoName)
        if(dtv.opencv_version() >= 3):
            self.fps = video.get(cv2.CAP_PROP_FPS)
        else:
            self.fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        ret, fr = video.read()
        sequence = []
        sequence.append(self.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)))
        self.oR, self.oC, self.oD = fr.shape
    
        while ret:
            ret, fr = video.read()
            if ret != False:
                sequence.append(self.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)))
        video.release()
        
        self.nframes = len(sequence)
        
        if (len(sequence)-1)%6 != 0:
            for i in range(6-(len(sequence)-1)%6):
                sequence.append(sequence[-1])
        print 'Reading Complete'
        return sequence
    
    def resize (self, frame):
        '''
        # MPEG Encoder:
        Method: resize(self, frame)-> auxImage
        About: This method adjusts the shape of a given frame.
        '''
        rR, rC, rD = frame.shape
        aR = aC = 0
    
        if rR % self.mbr != 0:
            aR = rR + (self.mbr - (rR%self.mbr))
        else:
            aR = rR
    
        if rC % self.mbc != 0:
            aC = rC + (self.mbc - (rC%self.mbc))
        else:
            aC = rC
    
#        for i in range (0,rR,2):
#            for j in range (0,rC,2):
#                frame[i+1,j,1] = frame[i,j+1,1] = frame[i+1,j+1,1] = frame[i,j,1]
#                frame[i+1,j,2] = frame[i,j+1,2] = frame[i+1,j+1,2] = frame[i,j,2]
    
        auxImage = np.zeros((aR, aC, rD), np.float32)
#        auxImage[:rR,:rC] = auxImage[aR-rR:aR,aC-rC:rC] =  frame
        auxImage[:rR,:rC] =  frame
    
        return auxImage
        
    def genHVStables (self):
        tables = [[0 for x in range (self.sspace+1)] for x in range (self.sspace+1)]
        g = [[0 for x in range (self.sspace+1)] for x in range (self.sspace+1)]
        qflat = self.flat*np.ones((8,8), float)
        for mh in range (self.sspace+1):
            for mt in range (self.sspace+1):
                vh = float(mh*self.fps)/float(self.oR)
                vt = float(mt*self.fps)/float(self.oC)
                v = sqrt(vh**2+vt**2)
                gaux = np.zeros((8,8), np.float32)
                const1 = float(mh*self.oR)/float(self.mbr*self.mbc)
                const2 = float(mt*self.oC)/float(self.mbr*self.mbc)
                if v != 0:
                    for i in range (8):
                        for j in range (8):
                            ai = const1*0.5*i
                            aj = const2*0.5*j
                            aij = ai + aj
                            gaux[i,j] = (6.1+7.3*abs(log(v/3.0))**3.0)*(v*(aij**2.0))*exp(-2.0*aij*(v+2.0)/45.9)
                    g[mh][mt] = gaux
                    
                else:
                    g[mh][mt] = gaux
                    
        g = np.array(g)
        gmax = np.max(g)
        for mh in range (self.sspace+1):
            for mt in range (self.sspace+1):
                qhvs = np.zeros((8,8), np.float32)
                for i in range (8):
                    for j in range (8):
                        qhvs[i,j] = ((mh+mt)/float(self.p))*(1.-(g[mh,mt,i,j]/gmax))
    
                tables[mh][mt] = qflat + qhvs
                
        self.hvstables = tables
        
    def run (self):
        '''
        # MPEG Encoder:
        Method: run(self)
        About: This method runs the algorithm of the MPEG encoder.
        '''
        self.output.write(str(self.oR)+','+str(self.oC)+','+str(self.oD)+' '+
        str(self.quality)+ ' '+ str(self.nframes)+' '+ self.mode + ' ' +
        str(self.sspace) + ' ' + str(self.hvsqm) + ' ' + self.DCTKERNEL + '\n')
        if self.hvsqm:
            self.output.write(str(self.flat)+' '+str(self.fps)+' '+str(self.p)+'\n')
        count = 0
        total = len(self.video)
        nBfr = 2
        gopsize = 6
        fP = 0       #        lP = nBfr+1
        print '\n### Starting MPEG Encoder ###\nTotal %d Progress..'%total
        #        if self.hvsqm: #            self.genHVStables#        else:#            self.Ztab = self.genQntb
        
        self.NBITS_I = np.zeros(total)
        self.NBITS_B = np.zeros(total)
        self.NBITS_P = np.zeros(total)
        self.ffNUMBITS = np.zeros(total)
        
        bitstream = ''  # to save in file
        
        for f in range(0,len(self.video)):
            # I B B P B B I B B I B B P B B I B B P
            # 0 1 2 3 4 5 6 7 8 9                    #            print 'Frame ', f
            ### I-FRAME ###
            if f%(gopsize) == 0:
                frame_type = '00'
                encoder = jpeg.Encoder( self.video[f], self.quality, self.hufftables, self.Z, self.mode, self.DCTKERNEL)  #                self.output.write(frame_type+'\n'+str(encoder.seqhuff[0])+'\n'+str(encoder.seqhuff[1])+'\n'+str(encoder.seqhuff[2])+'\n')
                bitstream += frame_type+'\n'+str(encoder.seqhuff[0])+'\n'+str(encoder.seqhuff[1])+'\n'+str(encoder.seqhuff[2])+'\n' 
                fP = f
                lP = f + (nBfr+1)
                count += 1   #                print 'I - Progress', count, total
                self.NBITS_I[f] = encoder.NumBits
            ### B-FRAME ###
            elif f%(nBfr+1) != 0:
                frame_type = '01'
                bframe = mpeg.Bframe(self.video[fP], self.video[f], self.video[lP], self.sspace, self.search)#                self.output.write(frame_type +' ')
                bitstream += frame_type +'\n'
                if self.hvsqm == 1:
                    vecsz = len(bframe.motionVec)
                    MV = list(np.zeros(vecsz)) #                print len(MV)
                    for j in range (vecsz):
                        if bframe.motionVec[j][1] == 'i':    #                        self.output.write(':'+str(bframe.motionVec[j][1])+','+str(bframe.motionVec[j][2])+','+str(bframe.motionVec[j][3])+','+str(bframe.motionVec[j][4])+','+str(bframe.motionVec[j][5]))  #                            bitstream += ':'+str(bframe.motionVec[j][1])+','+str(bframe.motionVec[j][2])+','+str(bframe.motionVec[j][3])+','+str(bframe.motionVec[j][4])+','+str(bframe.motionVec[j][5])
#                            if self.hvsqm == 1:
                            MV[j] = ( int(np.floor((abs(bframe.motionVec[j][2])+abs(bframe.motionVec[j][4]))/2.)) , int(np.floor((abs(bframe.motionVec[j][3])+abs(bframe.motionVec[j][5]))/2.) ))
                        else:       #                        self.output.write(':'+str(bframe.motionVec[j][1])+','+str(bframe.motionVec[j][2])+','+str(bframe.motionVec[j][3]))  #                            bitstream += ':'+str(bframe.motionVec[j][1])+','+str(bframe.motionVec[j][2])+','+str(bframe.motionVec[j][3])
#                            if self.hvsqm == 1:
                            MV[j] = ( abs(bframe.motionVec[j][2]), abs(bframe.motionVec[j][3]) )
    #                self.output.write('\n')  #                    bitstream += '\n'
                    ZtabVec = [self.hvstables, MV, self.sspace]
                    encoder = jpeghvs.Encoder(bframe.bframe, self.quality, self.hufftables, ZtabVec, self.hvsqm, self.mode)
                else:
                    encoder = jpeg.Encoder(bframe.bframe, self.quality, self.hufftables, self.Z, self.mode, self.DCTKERNEL)
#                self.output.write(str(encoder.seqhuff[0])+'\n'+str(encoder.seqhuff[1])+'\n'+str(encoder.seqhuff[2])+'\n')
                mv_seq = vec2bin(bframe.motionVec, frame_type, self.video[0].shape[1]/self.mbc)
                bitstream += mv_seq+'\n'+str(encoder.seqhuff[0])+'\n'+str(encoder.seqhuff[1])+'\n'+str(encoder.seqhuff[2])+'\n'
                count += 1  #                print 'B - Progress', count, total
                self.NBITS_B[f] = encoder.NumBits
            ### P-FRAME ###
            else:
                frame_type = '10'
                pframe = mpeg.Pframe(self.video[fP],self.video[f],self.sspace,self.search)    # P-frame #                self.output.write(frame_type +' ')
                bitstream += frame_type +'\n'
                if self.hvsqm==1:
                    vecsz = len(pframe.motionVec)
                    MV = list(np.zeros(vecsz))#[[] for x in range (vecsz)] list(np.zeros(vecsz))
                    for j in range(vecsz):
                        MV[j] = ( abs(pframe.motionVec[j][0]), abs(pframe.motionVec[j][1]) )     #                    self.output.write(':'+str(pframe.motionVec[j][0])+','+str(pframe.motionVec[j][1]))
                    ZtabVec = [self.hvstables, MV, self.sspace]
                    encoder = jpeghvs.Encoder(pframe.pframe, self.quality, self.hufftables, ZtabVec, self.hvsqm, self.mode)
#                self.output.write('\n') 
                else:
                    encoder = jpeg.Encoder(pframe.pframe, self.quality, self.hufftables, self.Z, self.mode, self.DCTKERNEL)
                
                mv_seq = vec2bin(pframe.motionVec, frame_type, self.video[0].shape[1]/self.mbc)  #                self.output.write(str(encoder.seqhuff[0])+'\n'+str(encoder.seqhuff[1])+'\n'+str(encoder.seqhuff[2])+'\n')
                bitstream += mv_seq+'\n'+str(encoder.seqhuff[0])+'\n'+str(encoder.seqhuff[1])+'\n'+str(encoder.seqhuff[2])+'\n'
                
                count += 1
                fP = f          #First P-frame
                lP = f+(nBfr+1) #Last P-frame
                self.NBITS_P[f] = encoder.NumBits
            
            self.ffNUMBITS[f] = encoder.NumBits
#            print('Progress: %s - %d/%d') % (frame_type, count, total)
            print('%d')%(count),
            self.avgBits.append(encoder.avgBits)
            self.CRate.append(encoder.CRate)
            self.redundancy.append(1.-(1./encoder.CRate))
        #END FOR
        self.output.write(bitstream)
        self.output.close()
        
#        ##### TEST #####
#        print "Saving bin file..."
#        bin_file=open(self.FILEDIR + self.FILENAME + '.bin', 'w')
#        b_fl= BitArray(bin=bitstream)
#        b_fl.tofile(bin_file)
#        bin_file.close()
            
class Decoder:
    def __init__(self, flName, videoName):
        '''
        # MPEG Decoder:
        Method: Constructor.
        About: This class runs the algorithm of the mpeg decoder.
        Parameters: 
            1) flname: File name.
        '''
        self.MBR, self.MBC = [16, 16]
        self.avgBits = []
        self.VIDEO = self.readVideo(videoName)
        self.psnrValues = []
        self.msimValues = []
        self.input = open(flName,'r')
        self.hvstables = None
        self.nframes = 0
        self.quality = 50
        self.mode = ''
        self.sspace = 0
        self.shape = []
        self.hufftables = acdctables()
        self.Z = []
        self.hvsqm = None
        self.mbr, self.mbc = [8, 8]
        self.DCTKERNEL = ''
#        self.FILEDIR = flName.strip(flName.split('/')[-1])
        
        self.VIDEOREC = []
        
    def readVideo(self, videoName):
        '''
        # MPEG Encoder:
        Method: readVideo(self, videoName)-> sequence
        About: This method store a video in a list.
        '''
        video = cv2.VideoCapture(videoName)
        if(dtv.opencv_version() >= 3):
            self.fps = video.get(cv2.CAP_PROP_FPS)
        else:
            self.fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        ret, fr = video.read()
        sequence = []
        sequence.append(self.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)))
        self.oR, self.oC, self.oD = fr.shape

        while ret:
            ret, fr = video.read()
            if ret != False:
                sequence.append(self.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)))
        video.release()
        
        self.nframes = len(sequence)
        
        if (len(sequence)-1)%6 != 0:
            for i in range(6-(len(sequence)-1)%6):
                sequence.append(sequence[-1])
        
        return sequence

    def resize (self, frame):
        '''
        # MPEG Encoder:
        Method: resize(self, frame)-> auxImage
        About: This method adjusts the shape of a given frame.
        '''
        rR, rC, rD = frame.shape
        aR = aC = 0

        if rR % self.MBR != 0:
            aR = rR + (self.MBR - (rR%self.MBR))
        else:
            aR = rR

        if rC % self.MBC != 0:
            aC = rC + (self.MBC - (rC%self.MBC))
        else:
            aC = rC

#        for i in range (0,rR,2):
#            for j in range (0,rC,2):
#                frame[i+1,j,1] = frame[i,j+1,1] = frame[i+1,j+1,1] = frame[i,j,1]
#                frame[i+1,j,2] = frame[i,j+1,2] = frame[i+1,j+1,2] = frame[i,j,2]
    
        auxImage = np.zeros((aR, aC, rD), np.float32)
#        auxImage[:rR,:rC] = auxImage[aR-rR:aR,aC-rC:rC] =  frame
        auxImage[:rR,:rC] =  frame
        return auxImage
        
    def genHVStables (self):
        tables = [[0 for x in range (int(self.sspace)+1)] for x in range (int(self.sspace)+1)]
        g = [[0 for x in range (int(self.sspace)+1)] for x in range (int(self.sspace)+1)]
        qflat = float(self.flat)*np.ones((8,8), np.float32)
#        print qflat
        for mh in range (int(self.sspace)+1):
            for mt in range (int(self.sspace)+1):
                
                vh = float(mh*float(self.fps))/float(self.shape[0])
                vt = float(mt*float(self.fps))/float(self.shape[1])
                v = sqrt(vh**2+vt**2)
                qhvs = np.zeros((8,8), float)
                gaux = np.zeros((8,8), np.float32)
                const1 = float(mh*int(self.shape[0]))/float(int(self.mbr)*int(self.mbc))
                const2 = float(mt*int(self.shape[1]))/float(int(self.mbr)*int(self.mbc))
                if v != 0:
                    for i in range (8):
                        for j in range (8):
                            ai = const1*0.5*i
                            aj = const2*0.5*j
                            aij = ai + aj
                            gaux[i,j] = (6.1+7.3*abs(log(v/3.0))**3.0)*(v*(aij**2.0))*exp(-2.0*aij*(v+2.0)/45.9)
                    g[mh][mt] = gaux
                    
                else:
                    g[mh][mt] = gaux
                    
        g = np.array(g)
        gmax = np.max(g)
        for mh in range (int(self.sspace)+1):
            for mt in range (int(self.sspace)+1):
                qhvs = np.zeros((8,8), np.float32)
                for i in range (8):
                    for j in range (8):
                        qhvs[i,j] = ((mh+mt)/float(self.p))*(1.-(g[mh,mt,i,j]/gmax))
                tables[mh][mt] = qflat + qhvs
                
        self.hvstables = tables
        
    def precover (self, pastfr, currentfr, motionVecs, sspace):
        '''
        # MPEG Decoder:
        Method: precover(self, pastfr, currentfr, motionVecs, sspace)-> result
        About: This method revocers a P frame.
        Parameters: 
            1) pastfr: Past frame.
            2) currentfr: Current frame.
            3) motionVecs: Motion vectors.
            4) sspace: Search space.
        '''
        result = np.zeros(pastfr.shape, np.float32)
        count = 0
        
        aR, aC, aD = currentfr.shape
        bckgndImgPast = np.zeros((aR+2*sspace, aC+2*sspace, aD), np.float32)
        bckgndImgPast[sspace:sspace+aR, sspace:sspace+aC] = pastfr
            
        for i in range(0,currentfr.shape[0],16):
            for j in range(0,currentfr.shape[1],16):
                a, b = motionVecs[count]
                result[i:i+16, j:j+16] = bckgndImgPast[i+a+sspace:i+a+sspace+16, j+b+sspace:j+b+sspace+16] + currentfr[i:i+16, j:j+16]
                count += 1
                
        return result
        
    def brecover (self, pastfr, currentfr, postfr, motionVecs, sspace):
        '''
        # MPEG Decoder:
        Method: precover(self, pastfr, currentfr, motionVecs, sspace)-> result
        About: This method revocers a B frame.
        Parameters: 
            1) pastfr: Past frame.
            2) currentfr: Current frame.
            3) postfr: Post frame.
            4) motionVecs: Motion vectors.
            5) sspace: Search space.
        '''
        result = np.zeros(pastfr.shape, np.float32)
        count = 0
        
        aR, aC, aD = currentfr.shape
        bckgImgPast = np.zeros((aR+2*sspace, aC+2*sspace, aD), np.float32)
        bckgImgPast[sspace:sspace+aR, sspace:sspace+aC] = pastfr
        bckgImgPost = np.zeros((aR+2*sspace, aC+2*sspace, aD), np.float32)
        bckgImgPost[sspace:sspace+aR, sspace:sspace+aC] = postfr
    
        for i in range(0,currentfr.shape[0],16):
            for j in range(0,currentfr.shape[1],16):
                if motionVecs[count][0] == 'i':
                    a, b, c, d = motionVecs[count][1:]
                    result[i:i+16, j:j+16] = (bckgImgPast[i+a+sspace:i+a+sspace+16, j+b+sspace:j+b+sspace+16] + 2.0*currentfr[i:i+16, j:j+16] + bckgImgPost[i+c+sspace:i+c+sspace+16, j+d+sspace:j+d+sspace+16])/2.0
                if motionVecs[count][0] == 'b':
                    a, b = motionVecs[count][1:]
                    result[i:i+16, j:j+16] = bckgImgPost[i+a+sspace:i+a+sspace+16, j+b+sspace:j+b+sspace+16] + currentfr[i:i+16, j:j+16]
                elif motionVecs[count][0] == 'f':
                    a, b = motionVecs[count][1:]
                    result[i:i+16, j:j+16] = bckgImgPast[i+a+sspace:i+a+sspace+16, j+b+sspace:j+b+sspace+16] + currentfr[i:i+16, j:j+16]
                count += 1
        return result

        
    def run (self):
        '''
        # MPEG Decoder:
        Method: run(self)
        About: This method runs the algorithm of the MPEG decoder.
        '''
        fo = self.input.read()
        fo = fo.split('\n')
        
        self.shape, self.quality, self.nframes, self.mode, self.sspace, self.hvsqm, self.DCTKERNEL = fo[0].split(' ')
        self.shape = np.array(self.shape.split(','), int)
        self.quality = int(self.quality)
        self.nframes = int(self.nframes)
        self.hvsqm   = int(self.hvsqm)
        self.Z = genQntb(self.quality)
        self.hvstables = None
        count = 1
        if self.hvsqm == 1:
            fo[count].split(' ')
            self.flat, self.fps, self.p = fo[count].split(' ')
            self.genHVStables()
            count += 1
        
        self.NBITS_I = np.zeros(self.nframes)
        self.NBITS_B = np.zeros(self.nframes)
        self.NBITS_P = np.zeros(self.nframes)
            
        sequence = []
        nauxfr = len(self.VIDEO) #self.nframes+(6-((self.nframes-1)%6))
        self.sspace = int(self.sspace)
        countfr = 0
        
        self.decNUMBITs = np.zeros(nauxfr)
        
        print '\n### Starting MPEG Decoder ###'
        print '1) MPEG: Recovering Frames\nTotal %d progress...'%nauxfr
        mb_type = {('00'):('I'), ('01'):('B'), ('10'):('P')}
        nl = self.shape[0] if self.shape[0]%16 == 0 else self.shape[0]+(self.shape[0]%16 - 16)
        ml = self.shape[1] if self.shape[1]%16 == 0 else self.shape[1]+(self.shape[1]%16 - 16)
        while countfr < nauxfr:
            aux = fo[count]
            if mb_type[aux] == 'I':
                ch = []
                aux = 0
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                self.avgBits.append(float(aux)/float(nl*ml))
                sequence.append([countfr, self.resize(jpeg.Decoder(ch, self.hufftables, self.Z, [self.shape, self.quality, self.mode, self.DCTKERNEL])._run_()), None])
            elif mb_type[aux] == 'P':
                count += 1
                motionVec = []
                motionVec = bin2vec(fo[count], self.VIDEO[0].shape[1]/self.MBC, aux)
                ch = []
                aux = 0
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                self.avgBits.append(float(aux)/float(nl*ml))
                if self.hvsqm == 1:
                    sequence.append([countfr, jpeghvs.Decoder(ch, self.hufftables, self.hvstables, [self.shape, self.quality, self.mode, motionVec])._run_(), motionVec])
                else:
                    sequence.append([countfr, self.resize(jpeg.Decoder(ch, self.hufftables, self.Z, [self.shape, self.quality, self.mode, self.DCTKERNEL])._run_()), motionVec])
            elif mb_type[aux] == 'B':
                count += 1
                motionVec = []
                motionVec = bin2vec(fo[count], self.VIDEO[0].shape[1]/self.MBC, aux)
                ch = []
                aux = 0
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                count += 1
                ch.append(fo[count])
                aux += len(fo[count])
                self.avgBits.append(float(aux)/float(nl*ml))
                if self.hvsqm == 1:
                    sequence.append([countfr, jpeghvs.Decoder(ch, self.hufftables, self.hvstables, [self.shape, self.quality, self.mode, motionVec])._run_(), motionVec])
                else:
                    sequence.append([countfr, self.resize(jpeg.Decoder(ch, self.hufftables, self.Z, [self.shape, self.quality, self.mode, self.DCTKERNEL])._run_()), motionVec])
                
            if countfr < self.nframes:
                self.decNUMBITs[countfr] = len(ch[0]) + len(ch[1]) + len(ch[2])
            count += 1
            countfr += 1
#            print 'Progress: %d/%d - nbits: %d' % (countfr,nauxfr,self.decNUMBITs[countfr-1])
            print '%d' % (countfr),
        #END WHILE
        count = 0
        sequence.sort(key=lambda tup: tup[0])
        self.input.close()
        del ch, motionVec
        print '\n2) Motion Compensation'
        nBfr = 2
        gopsize = 6
        fP = 0
        self.typeseq = []
        # I B B P B B I B B P  B  B  I  B  B P
        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13
        for f in range(len(sequence)):       #            print 'Frame ', f
            if f%(gopsize) == 0:  # I-FRAME
                fP = f
                lP = f + (nBfr+1)
                if lP<self.nframes:
                    sequence[lP][1] = self.precover(sequence[fP][1],sequence[lP][1], sequence[lP][2], self.sspace)
#                    
                self.typeseq.append('I')
                if f < self.nframes:
                    self.NBITS_I[f] = self.decNUMBITs[f]
#                print 'I -', f, ';',
            elif f%(nBfr+1) != 0:  # B-FRAME
                sequence[f][1] = self.brecover(sequence[fP][1], sequence[f][1], sequence[lP][1], sequence[f][2], self.sspace)
                self.typeseq.append('B')
                if f < self.nframes:
                    self.NBITS_B[f] = self.decNUMBITs[f]
#                print 'B -',f, ';',
            else:  # P-FRAME
                self.typeseq.append('P')
                if f < self.nframes:
                    self.NBITS_P[f] = self.decNUMBITs[f]
                    
                fP = f          #First P-frame
                lP = f+(nBfr+1) #Last P-frame
#        print '\n'
        self.VIDEOREC = sequence
        #EDITING

#        Controle de saturação
#        output = []
#        print "%s Computing visual metrics. %s\nPlease wait..." % ("#"*4,"#"*4)
#        for i in range (self.nframes):
##            sequence[i][1][sequence[i][1]>255.0] = 255.0
##            sequence[i][1][sequence[i][1]<0.0] = 0.0
##            output.append(cv2.cvtColor(np.uint8(sequence[i][1]), cv2.COLOR_YCR_CB2BGR))
#            output.append(sequence[i][1])
#            self.msimValues.append(metrics.mssim(self.originalVideo[i][:,:,0], output[i][:,:,0]))
#            self.psnrValues.append(metrics.psnr1c(self.originalVideo[i][:,:,0], output[i][:,:,0]))
##            self.msimValues.append(metrics.msim(self.originalVideo[i], sequence[i][1]))
##            self.psnrValues.append(metrics.psnr(self.originalVideo[i], sequence[i][1]))
#        
#        print "Thanks!"
#        
#        return [output, self.psnrValues, self.msimValues, typeseq]
        
        
def acdctables():
    '''
    Huffman code Tables for AC and DC coefficient differences.
        output: (dcLumaTB, dcChroTB, acLumaTB, acChrmTB)
    '''
    dcLumaTB = { 0:(2,'00'),     1:(3,'010'),      2:(3,'011'),       3:(3,'100'),
            4:(3,'101'),    5:(3,'110'),      6:(4,'1110'),      7:(5,'11110'),
            8:(6,'111110'), 9:(7,'1111110'), 10:(8,'11111110'), 11:(9,'111111110')}

    dcChroTB = { 0:(2,'00'),       1:(2,'01'),         2:( 2,'10'),          3:( 3,'110'),
            4:(4,'1110'),     5:(5,'11110'),      6:( 6,'111110'),      7:( 7,'1111110'),
            8:(8,'11111110'), 9:(9,'111111110'), 10:(10,'1111111110'), 11:(11,'11111111110')}
                 
    #Table for luminance DC coefficient differences
    #       [(run,category) : (size, 'codeword')]
    acLumaTB = {( 0, 0):( 4,'1010'), #EOB
            ( 0, 1):( 2,'00'),               ( 0, 2):( 2,'01'),
            ( 0, 3):( 3,'100'),              ( 0, 4):( 4,'1011'),
            ( 0, 5):( 5,'11010'),            ( 0, 6):( 7,'1111000'),
            ( 0, 7):( 8,'11111000'),         ( 0, 8):(10,'1111110110'),
            ( 0, 9):(16,'1111111110000010'), ( 0,10):(16,'1111111110000011'),
            ( 1, 1):( 4,'1100'),             ( 1, 2):( 5,'11011'),
            ( 1, 3):( 7,'1111001'),          ( 1, 4):( 9,'111110110'),
            ( 1, 5):(11,'11111110110'),      ( 1, 6):(16,'1111111110000100'),
            ( 1, 7):(16,'1111111110000101'), ( 1, 8):(16,'1111111110000110'),
            ( 1, 9):(16,'1111111110000111'), ( 1,10):(16,'1111111110001000'),
            ( 2, 1):( 5,'11100'),            ( 2, 2):( 8,'11111001'),
            ( 2, 3):(10,'1111110111'),       ( 2, 4):(12,'111111110100'),
            ( 2, 5):(16,'1111111110001001'), ( 2, 6):(16,'1111111110001010'),
            ( 2, 7):(16,'1111111110001011'), ( 2, 8):(16,'1111111110001100'),
            ( 2, 9):(16,'1111111110001101'), ( 2,10):(16,'1111111110001110'),
            ( 3, 1):( 6,'111010'),           ( 3, 2):( 9,'111110111'),
            ( 3, 3):(12,'111111110101'),     ( 3, 4):(16,'1111111110001111'),
            ( 3, 5):(16,'1111111110010000'), ( 3, 6):(16,'1111111110010001'),
            ( 3, 7):(16,'1111111110010010'), ( 3, 8):(16,'1111111110010011'),
            ( 3, 9):(16,'1111111110010100'), ( 3,10):(16,'1111111110010101'),
            ( 4, 1):( 6,'111011'),           ( 4, 2):(10,'1111111000'),
            ( 4, 3):(16,'1111111110010110'), ( 4, 4):(16,'1111111110010111'),
            ( 4, 5):(16,'1111111110011000'), ( 4, 6):(16,'1111111110011001'),
            ( 4, 7):(16,'1111111110011010'), ( 4, 8):(16,'1111111110011011'),
            ( 4, 9):(16,'1111111110011100'), ( 4,10):(16,'1111111110011101'),
            ( 5, 1):( 7,'1111010'),          ( 5, 2):(11,'11111110111'),
            ( 5, 3):(16,'1111111110011110'), ( 5, 4):(16,'1111111110011111'),
            ( 5, 5):(16,'1111111110100000'), ( 5, 6):(16,'1111111110100001'),
            ( 5, 7):(16,'1111111110100010'), ( 5, 8):(16,'1111111110100011'),
            ( 5, 9):(16,'1111111110100100'), ( 5,10):(16,'1111111110100101'),
            ( 6, 1):( 7,'1111011'),          ( 6, 2):(12,'111111110110'),
            ( 6, 3):(16,'1111111110100110'), ( 6, 4):(16,'1111111110100111'),
            ( 6, 5):(16,'1111111110101000'), ( 6, 6):(16,'1111111110101001'),
            ( 6, 7):(16,'1111111110101010'), ( 6, 8):(16,'1111111110101011'),
            ( 6, 9):(16,'1111111110101100'), ( 6,10):(16,'1111111110101101'),
            ( 7, 1):( 8,'11111010'),         ( 7, 2):(12,'111111110111'),
            ( 7, 3):(16,'1111111110101110'), ( 7, 4):(16,'1111111110101111'),
            ( 7, 5):(16,'1111111110110000'), ( 7, 6):(16,'1111111110110001'),
            ( 7, 7):(16,'1111111110110010'), ( 7, 8):(16,'1111111110110011'),
            ( 7, 9):(16,'1111111110110100'), ( 7,10):(16,'1111111110110101'),
            ( 8, 1):( 9,'111111000'),        ( 8, 2):(15,'111111111000000'),
            ( 8, 3):(16,'1111111110110110'), ( 8, 4):(16,'1111111110110111'),
            ( 8, 5):(16,'1111111110111000'), ( 8, 6):(16,'1111111110111001'),
            ( 8, 7):(16,'1111111110111010'), ( 8, 8):(16,'1111111110111011'),
            ( 8, 9):(16,'1111111110111100'), ( 8,10):(16,'1111111110111101'),
            ( 9, 1):( 9,'111111001'),        ( 9, 2):(16,'1111111110111110'),
            ( 9, 3):(16,'1111111110111111'), ( 9, 4):(16,'1111111111000000'),
            ( 9, 5):(16,'1111111111000001'), ( 9, 6):(16,'1111111111000010'),
            ( 9, 7):(16,'1111111111000011'), ( 9, 8):(16,'1111111111000100'),
            ( 9, 9):(16,'1111111111000101'), ( 9,10):(16,'1111111111000110'),
            (10, 1):( 9,'111111010'),        (10, 2):(16,'1111111111000111'),
            (10, 3):(16,'1111111111001000'), (10, 4):(16,'1111111111001001'),
            (10, 5):(16,'1111111111001010'), (10, 6):(16,'1111111111001011'),
            (10, 7):(16,'1111111111001100'), (10, 8):(16,'1111111111001101'),
            (10, 9):(16,'1111111111001110'), (10,10):(16,'1111111111001111'),
            (11, 1):(10,'1111111001'),       (11, 2):(16,'1111111111010000'),
            (11, 3):(16,'1111111111010001'), (11, 4):(16,'1111111111010010'),
            (11, 5):(16,'1111111111010011'), (11, 6):(16,'1111111111010100'),
            (11, 7):(16,'1111111111010101'), (11, 8):(16,'1111111111010110'),
            (11, 9):(16,'1111111111010111'), (11,10):(16,'1111111111011000'),
            (12, 1):(10,'1111111010'),       (12, 2):(16,'1111111111011001'),
            (12, 3):(16,'1111111111011010'), (12, 4):(16,'1111111111011011'),
            (12, 5):(16,'1111111111011100'), (12, 6):(16,'1111111111011101'),
            (12, 7):(16,'1111111111011110'), (12, 8):(16,'1111111111011111'),
            (12, 9):(16,'1111111111100000'), (12,10):(16,'1111111111100001'),
            (13, 1):(11,'11111111000'),      (13, 2):(16,'1111111111100010'),
            (13, 3):(16,'1111111111100011'), (13, 4):(16,'1111111111100100'),
            (13, 5):(16,'1111111111100101'), (13, 6):(16,'1111111111100110'),
            (13, 7):(16,'1111111111100111'), (13, 8):(16,'1111111111101000'),
            (13, 9):(16,'1111111111101001'), (13,10):(16,'1111111111101010'),
            (14, 1):(16,'1111111111101011'), (14, 2):(16,'1111111111101100'),
            (14, 3):(16,'1111111111101101'), (14, 4):(16,'1111111111101110'),
            (14, 5):(16,'1111111111101111'), (14, 6):(16,'1111111111110000'),
            (14, 7):(16,'1111111111110001'), (14, 8):(16,'1111111111110010'),
            (14, 9):(16,'1111111111110011'), (14,10):(16,'1111111111110100'),
            (15, 0):(11,'11111111001'),     #(ZRL)
            (15, 1):(16,'1111111111110101'), (15, 2):(16,'1111111111110110'),
            (15, 3):(16,'1111111111110111'), (15, 4):(16,'1111111111111000'),
            (15, 5):(16,'1111111111111001'), (15, 6):(16,'1111111111111010'),
            (15, 7):(16,'1111111111111011'), (15, 8):(16,'1111111111111100'),
            (15, 9):(16,'1111111111111101'), (15,10):(16,'1111111111111110')}
            
    #Table for chrominance AC coefficients
    acChrmTB = {( 0, 0):( 2,'00'), #EOB
            ( 0, 1):( 2,'01'),               ( 0, 2):( 3,'100'),
            ( 0, 3):( 4,'1010'),             ( 0, 4):( 5,'11000'),
            ( 0, 5):( 5,'11001'),            ( 0, 6):( 6,'111000'),
            ( 0, 7):( 7,'1111000'),          ( 0, 8):( 9,'111110100'),
            ( 0, 9):(10,'1111110110'),       ( 0,10):(12,'111111110100'),
            ( 1, 1):( 4,'1011'),             ( 1, 2):( 6,'111001'),
            ( 1, 3):( 8,'11110110'),         ( 1, 4):( 9,'111110101'),
            ( 1, 5):(11,'11111110110'),      ( 1, 6):(12,'111111110101'),
            ( 1, 7):(16,'1111111110001000'), ( 1, 8):(16,'1111111110001001'),
            ( 1, 9):(16,'1111111110001010'), ( 1,10):(16,'1111111110001011'),
            ( 2, 1):( 5,'11010'),            ( 2, 2):( 8,'11110111'),
            ( 2, 3):(10,'1111110111'),       ( 2, 4):(12,'111111110110'),
            ( 2, 5):(15,'111111111000010'),  ( 2, 6):(16,'1111111110001100'),
            ( 2, 7):(16,'1111111110001101'), ( 2, 8):(16,'1111111110001110'),
            ( 2, 9):(16,'1111111110001111'), ( 2,10):(16,'1111111110010000'),
            ( 3, 1):( 5,'11011'),            ( 3, 2):( 8,'11111000'),
            ( 3, 3):(10,'1111111000'),       ( 3, 4):(12,'111111110111'),
            ( 3, 5):(16,'1111111110010001'), ( 3, 6):(16,'1111111110010010'),
            ( 3, 7):(16,'1111111110010011'), ( 3, 8):(16,'1111111110010100'),
            ( 3, 9):(16,'1111111110010101'), ( 3,10):(16,'1111111110010110'),
            ( 4, 1):( 6,'111010'),           ( 4, 2):( 9,'111110110'),
            ( 4, 3):(16,'1111111110010111'), ( 4, 4):(16,'1111111110011000'),
            ( 4, 5):(16,'1111111110011001'), ( 4, 6):(16,'1111111110011010'),
            ( 4, 7):(16,'1111111110011011'), ( 4, 8):(16,'1111111110011100'),
            ( 4, 9):(16,'1111111110011101'), ( 4,10):(16,'1111111110011110'),
            ( 5, 1):( 6,'111011'),           ( 5, 2):(10,'1111111001'),
            ( 5, 3):(16,'1111111110011111'), ( 5, 4):(16,'1111111110100000'),
            ( 5, 5):(16,'1111111110100001'), ( 5, 6):(16,'1111111110100010'),
            ( 5, 7):(16,'1111111110100011'), ( 5, 8):(16,'1111111110100100'),
            ( 5, 9):(16,'1111111110100101'), ( 5,10):(16,'1111111110100110'),
            ( 6, 1):( 7,'1111001'),          ( 6, 2):(11,'11111110111'),
            ( 6, 3):(16,'1111111110100111'), ( 6, 4):(16,'1111111110101000'),
            ( 6, 5):(16,'1111111110101001'), ( 6, 6):(16,'1111111110101010'),
            ( 6, 7):(16,'1111111110101011'), ( 6, 8):(16,'1111111110101100'),
            ( 6, 9):(16,'1111111110101101'), ( 6,10):(16,'1111111110101110'),
            ( 7, 1):( 7,'1111010'),          ( 7, 2):(11,'11111111000'),
            ( 7, 3):(16,'1111111110101111'), ( 7, 4):(16,'1111111110110000'),
            ( 7, 5):(16,'1111111110110001'), ( 7, 6):(16,'1111111110110010'),
            ( 7, 7):(16,'1111111110110011'), ( 7, 8):(16,'1111111110110100'),
            ( 7, 9):(16,'1111111110110101'), ( 7,10):(16,'1111111110110110'),
            ( 8, 1):( 8,'11111001'),         ( 8, 2):(16,'1111111110110111'),
            ( 8, 3):(16,'1111111110111000'), ( 8, 4):(16,'1111111110111001'),
            ( 8, 5):(16,'1111111110111010'), ( 8, 6):(16,'1111111110111011'),
            ( 8, 7):(16,'1111111110111100'), ( 8, 8):(16,'1111111110111101'),
            ( 8, 9):(16,'1111111110111110'), ( 8,10):(16,'1111111110111111'),
            ( 9, 1):( 9,'111110111'),        ( 9, 2):(16,'1111111111000000'),
            ( 9, 3):(16,'1111111111000001'), ( 9, 4):(16,'1111111111000010'),
            ( 9, 5):(16,'1111111111000011'), ( 9, 6):(16,'1111111111000100'),
            ( 9, 7):(16,'1111111111000101'), ( 9, 8):(16,'1111111111000110'),
            ( 9, 9):(16,'1111111111000111'), ( 9,10):(16,'1111111111001000'),
            (10, 1):( 9,'111111000'),        (10, 2):(16,'1111111111001001'),
            (10, 3):(16,'1111111111001010'), (10, 4):(16,'1111111111001011'),
            (10, 5):(16,'1111111111001100'), (10, 6):(16,'1111111111001101'),
            (10, 7):(16,'1111111111001110'), (10, 8):(16,'1111111111001111'),
            (10, 9):(16,'1111111111010000'), (10,10):(16,'1111111111010001'),
            (11, 1):( 9,'111111001'),        (11, 2):(16,'1111111111010010'),
            (11, 3):(16,'1111111111010011'), (11, 4):(16,'1111111111010100'),
            (11, 5):(16,'1111111111010101'), (11, 6):(16,'1111111111010110'),
            (11, 7):(16,'1111111111010111'), (11, 8):(16,'1111111111011000'),
            (11, 9):(16,'1111111111011001'), (11,10):(16,'1111111111011010'),
            (12, 1):( 9,'111111010'),        (12, 2):(16,'1111111111011011'),
            (12, 3):(16,'1111111111011100'), (12, 4):(16,'1111111111011101'),
            (12, 5):(16,'1111111111011110'), (12, 6):(16,'1111111111011111'),
            (12, 7):(16,'1111111111100000'), (12, 8):(16,'1111111111100001'),
            (12, 9):(16,'1111111111100010'), (12,10):(16,'1111111111100011'),
            (13, 1):(11,'11111111001'),      (13, 2):(16,'1111111111100100'),
            (13, 3):(16,'1111111111100101'), (13, 4):(16,'1111111111100110'),
            (13, 5):(16,'1111111111100111'), (13, 6):(16,'1111111111101000'),
            (13, 7):(16,'1111111111101001'), (13, 8):(16,'1111111111101010'),
            (13, 9):(16,'1111111111101011'), (13,10):(16,'1111111111101100'),
            (14, 1):(14,'11111111100000'),   (14, 2):(16,'1111111111101101'),
            (14, 3):(16,'1111111111101110'), (14, 4):(16,'1111111111101111'),
            (14, 5):(16,'1111111111110000'), (14, 6):(16,'1111111111110001'),
            (14, 7):(16,'1111111111110010'), (14, 8):(16,'1111111111110011'),
            (14, 9):(16,'1111111111110100'), (14,10):(16,'1111111111110101'),
            (15, 0):(10,'1111111010'),       #(ZRL)
            (15, 1):(15,'111111111000011'),  (15, 2):(16,'1111111111110110'),
            (15, 3):(16,'1111111111110111'), (15, 4):(16,'1111111111111000'),
            (15, 5):(16,'1111111111111001'), (15, 6):(16,'1111111111111010'),
            (15, 7):(16,'1111111111111011'), (15, 8):(16,'1111111111111100'),
            (15, 9):(16,'1111111111111101'), (15,10):(16,'1111111111111110')}
                
    return (dcLumaTB, dcChroTB, acLumaTB, acChrmTB)
    
def genQntb(qualy):
    '''
    Gera tabela para quantização.    
    input: qualy -> int [1-99]
    '''

    fact = qualy
    Z = np.array([[[16., 17., 17.], [11., 18., 18.], [10., 24., 24.], [16., 47., 47.], [124., 99., 99.], [140., 99., 99.], [151., 99., 99.], [161., 99., 99.]],
              [[12., 18., 18.], [12., 21., 21.], [14., 26., 26.], [19., 66., 66.], [ 26., 99., 99.], [158., 99., 99.], [160., 99., 99.], [155., 99., 99.]],
              [[14., 24., 24.], [13., 26., 26.], [16., 56., 56.], [24., 99., 99.], [ 40., 99., 99.], [157., 99., 99.], [169., 99., 99.], [156., 99., 99.]],
              [[14., 47., 47.], [17., 66., 66.], [22., 99., 99.], [29., 99., 99.], [ 51., 99., 99.], [187., 99., 99.], [180., 99., 99.], [162., 99., 99.]],
              [[18., 99., 99.], [22., 99., 99.], [37., 99., 99.], [56., 99., 99.], [ 68., 99., 99.], [109., 99., 99.], [103., 99., 99.], [177., 99., 99.]],
              [[24., 99., 99.], [35., 99., 99.], [55., 99., 99.], [64., 99., 99.], [ 81., 99., 99.], [104., 99., 99.], [113., 99., 99.], [192., 99., 99.]],
              [[49., 99., 99.], [64., 99., 99.], [78., 99., 99.], [87., 99., 99.], [103., 99., 99.], [121., 99., 99.], [120., 99., 99.], [101., 99., 99.]],
              [[72., 99., 99.], [92., 99., 99.], [95., 99., 99.], [98., 99., 99.], [112., 99., 99.], [100., 99., 99.], [103., 99., 99.], [199., 99., 99.]]])
              
    if qualy < 1 : fact = 1
    if qualy > 99: fact = 99
    if qualy < 50:
        qualy = 5000 / fact
    else:
        qualy = 200 - 2*fact
    
    qZ = ((Z*qualy) + 50)/100
    qZ[qZ<1] = 1
    qZ[qZ>255] = 255

    return qZ
    
# Huffman doces for the motion vectors.
vlc_mv_enc = {  (-30):('0010'),       (-29):('00010'),      (-28):('0000110'),    (-27):('00001010'),
                (-26):('00001000'),   (-25):('00000110'),   (-24):('0000010110'), (-23):('0000010100'),
                (-22):('0000010010'), (-21):('00000100010'),(-20):('00000100000'),(-19):('00000011110'),
                (-18):('00000011100'),(-17):('00000011010'),(-16):('00000011001'),(-15):('00000011011'),
                (-14):('00000011101'),(-13):('00000011111'),(-12):('00000100001'),(-11):('00000100011'),
                (-10):('0000010011'), (-9) :('0000010101'),  (-8):('0000010111'),  (-7):('00000111'),
                (-6) :('00001001'),   (-5) :('00001011'),    (-4):('0000111'),     (-3):('00011'),
                (-2) :('0011'),       (-1) :('011'),          (0):('1'),            (1):('010'),  (2):('0010'),
                (3)  :('00010'),       (4) :('0000110'),      (5):('00001010'),     (6):('00001000'),
                (7)  :('00000110'),     (8):('0000010110'),   (9):('0000010100'),  (10):('0000010010'),
                (11) :('00000100010'), (12):('00000100000'), (13):('00000011110'), (14):('00000011100'),
                (15) :('00000011010'), (16):('00000011001'), (17):('00000011011'), (18):('00000011101'),
                (19) :('00000011111'), (20):('00000100001'), (21):('00000100011'), (22):('0000010011'),
                (23) :('0000010101'),  (24):('0000010111'),  (25):('00000111'),    (26):('00001001'),
                (27) :('00001011'),    (28):('0000111'),     (29):('00011'),       (30):('0011')}
            
vlc_mv_dec = {('00000011001'):(-16,16), ('00000011011'):(-15,17),
              ('00000011101'):(-14,18), ('00000011111'):(-13,19),
              ('00000100001'):(-12,20), ('00000100011'):(-11,21),
              ('0000010011'):(-10,22), ('0000010101'):(-9,23),
              ('0000010111'):(-8,24), ('00000111'):(-7,25),
              ('00001001'):(-6,26), ('00001011'):(-5,27),
              ('0000111'):(-4,28), ('00011'):(-3,29),
              ('0011'):(-2,30), ('011'):(-1,-1),
              ('1'):(0,0), ('010'):(1,1),
              ('0010'):(2,-30), ('00010'):(3,-29),
              ('0000110'):(4,-28), ('00001010'):(5,-27),
              ('00001000'):(6,-26), ('00000110'):(7,-25),
              ('0000010110'):(8,-24), ('0000010100'):(9,-23),
              ('0000010010'):(10,-22), ('00000100010'):(11,-21),
              ('00000100000'):(12,-20), ('00000011110'):(13,-19),
              ('00000011100'):(14,-18), ('00000011010'):(15,-17),
              ('00000011001'):(16,-16), ('00000011011'):(17,-15)}

            
def vec2bin (MV, frame_type, nCol):
    """
    # MPEG Encoder:  \n
    Method: vec2bin(self, MV, frame_type, nCol)-> vec_seq \n
    About: Encodes the motion vectors. \n
    """
    c = nCol
    vec_seq = ''
    MV_dif = []
    
    if frame_type == '01':    #B frame
        for i in range(len(MV)):
            if MV[i][1] == 'f':
                if i%c == 0:
                    MV_dif.append((MV[i][1],MV[i][2],MV[i][3]))
                else:
                    if MV[i-1][1] == 'i':
                        MV_dif.append((MV[i][1], MV[i][2]-MV[i-1][2], MV[i][3]-MV[i-1][3]))
                    else:
                        MV_dif.append((MV[i][1], MV[i][2]-MV[i-1][2], MV[i][3]-MV[i-1][3]))
            elif MV[i][1] == 'b':
                if i%c == 0:
                    MV_dif.append((MV[i][1],MV[i][2],MV[i][3]))
                else:
                    if MV[i-1][1] == 'i':
                        MV_dif.append((MV[i][1], MV[i][2]-MV[i-1][2], MV[i][3]-MV[i-1][3]))
                    else:
                        MV_dif.append((MV[i][1], MV[i][2]-MV[i-1][2], MV[i][3]-MV[i-1][3]))
            elif MV[i][1] == 'i':
                if i%c == 0:
                    MV_dif.append((MV[i][1],MV[i][2],MV[i][3],MV[i][4],MV[i][5]))
                else:
                    if MV[i-1][1] == 'i':
                        MV_dif.append((MV[i][1], MV[i][2]-MV[i-1][2], MV[i][3]-MV[i-1][3], MV[i][4]-MV[i-1][4], MV[i][5]-MV[i-1][5]))
                    else:
                        MV_dif.append((MV[i][1], MV[i][2]-MV[i-1][2], MV[i][3]-MV[i-1][3], MV[i][4]-MV[i-1][2], MV[i][5]-MV[i-1][3]))

    else:                       #P frame
        for i in range(len(MV)):
            if i%c == 0:
                MV_dif.append((MV[i][0],MV[i][1]))
            else:
                MV_dif.append((MV[i][0]-MV[i-1][0], MV[i][1]-MV[i-1][1]))
                
    mb_type = {('f'):('00'), ('b'):('01'), ('i'):('10')}
    
    for i in range (len(MV_dif)):
        if frame_type == '01':
            vec_seq += mb_type[MV_dif[i][0]]
            for j in range (1,len(MV_dif[i])):
                vec_seq += str(vlc_mv_dec[vlc_mv_enc[MV_dif[i][j]]].index(MV_dif[i][j]))+vlc_mv_enc[MV_dif[i][j]]
        else:
            for j in range (len(MV_dif[i])):
                vec_seq += str(vlc_mv_dec[vlc_mv_enc[MV_dif[i][j]]].index(MV_dif[i][j]))+vlc_mv_enc[MV_dif[i][j]]
            
    return vec_seq    

def bin2vec (vec_seq, nCol, p_type):
    """
    # MPEG Decoder: \n
    Method: bin2vec(self, vec_seq, nCol, ptype) -> resp \n
    About: Decodes the motions vectors. \n
    """
    c = nCol
    MV = []
    resp = []
    if p_type == '01':   # B frame
        string_aux = ''
        signal = ''
        MC_type = ''
        aux_vec = []
        control = 0
        mc_type = {('00'):('f'), ('01'):('b'), ('10'):('i')}
        for i in range(len(vec_seq)):
            if control == 1:
                control = 0
                continue
            
            elif len(MC_type) == 0:
                MC_type += vec_seq [i:i+2]
                aux_vec.append(mc_type[MC_type])
                control = 1
            else:
                if mc_type[MC_type] == 'i':
                    if len(signal) == 0:
                        signal = vec_seq[i]
                    elif len(signal) != 0:
                        string_aux += vec_seq[i]
                        if string_aux in vlc_mv_dec:
                            aux_vec.append(vlc_mv_dec[string_aux][int(signal)])
                            if len(aux_vec) == 5:
                                MV.append((aux_vec[0],aux_vec[1],aux_vec[2],aux_vec[3],aux_vec[4]))
                                string_aux = ''
                                signal = ''
                                MC_type = ''
                                aux_vec = []
                            else:
                                string_aux = ''
                                signal = ''
                        else:
                            pass
                else:
                    if len(signal) == 0:
                        signal = vec_seq[i]
                    elif len(signal) != 0:
                        string_aux += vec_seq[i]
                        if string_aux in vlc_mv_dec:
                            aux_vec.append(vlc_mv_dec[string_aux][int(signal)])
                            if len(aux_vec) == 3:
                                MV.append((aux_vec[0],aux_vec[1],aux_vec[2]))
                                string_aux = ''
                                signal = ''
                                MC_type = ''
                                aux_vec = []
                            else:
                                string_aux = ''
                                signal = ''
                        else:
                            pass
                                
        for i in range (len(MV)):
            if i%c == 0:
                if MV[i][0] != 'i':
                    resp.append((MV[i][0],MV[i][1],MV[i][2]))
                else:
                    resp.append((MV[i][0],MV[i][1],MV[i][2],MV[i][3],MV[i][4]))
                            
            else:
                if MV[i][0] != 'i':
                    if resp[-1][0] != 'i':
                        resp.append((MV[i][0],MV[i][1]+resp[i-1][1],MV[i][2]+resp[i-1][2]))
                    else:
                        resp.append((MV[i][0],MV[i][1]+resp[i-1][1],MV[i][2]+resp[i-1][2]))
                else:
                    if resp[-1][0] != 'i':
                        resp.append((MV[i][0],MV[i][1]+resp[i-1][1],MV[i][2]+resp[i-1][2],MV[i][3]+resp[i-1][1],MV[i][4]+resp[i-1][2]))
                    else:
                        resp.append((MV[i][0],MV[i][1]+resp[i-1][1],MV[i][2]+resp[i-1][2],MV[i][3]+resp[i-1][3],MV[i][4]+resp[i-1][4]))
    
    else:      # P frame
        string_aux = ''
        aux_vec = []
        signal = ''
        for i in range(len(vec_seq)):
            if len(signal) == 0:
                signal = vec_seq[i]
            else:
                string_aux += vec_seq[i]
                if string_aux in vlc_mv_dec:
                    aux_vec.append(vlc_mv_dec[string_aux][int(signal)])
                    if len(aux_vec) == 2:
                        MV.append((aux_vec[0],aux_vec[1]))
                        string_aux = ''
                        signal = ''
                        aux_vec = []
                    else:
                        string_aux = ''
                        signal = ''
                else:
                    pass
                
        for i in range(len(MV)):
            if i%c == 0:
                resp.append((MV[i][0],MV[i][1]))
            else:
                resp.append((MV[i][0]+resp[i-1][0], MV[i][1]+resp[i-1][1]))
    return resp

