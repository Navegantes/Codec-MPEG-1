# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:13:32 2016

@author: luan
@author: Navegantes
"""

from mpegCodec import codec
from mpegCodec.utils import frame_utils as futils
from mpegCodec.utils.image_quality_assessment import metrics
import matplotlib.pylab as plt
from Tkinter import Tk
from tkFileDialog import askopenfilename, askdirectory
import sys
import numpy as np
import cv2

root = Tk()
root.withdraw()

fileName = askopenfilename(parent=root, title="Enter with a file name.").__str__()
if fileName == '': 
    sys.exit('Filename empty!')
print("\nFile: " + fileName)
name = fileName.split('/')
#print name
name = name[-1]
print name
name = name.split('.')[-1]
#print name

# In order to run the encoder just enter with a video file.
# In order to run the decoder just enter with a output file (in the output directory).

files = 1       # 0 - Runs all encoder modes for a given video (normal - 444 and 420, hvs - 444 and 420)
                # 1 - Runs encoder for a given video with the following setup.

quality = 75    # Compression quality.
sspace = 15    # Search space.
search = 1        # 0 - Full search; 1 - Parallel hierarchical.
flat = 10.0    # Qflat value.
p = 2.0        # Parameter p.
mode = '420'       # 444 or 420
hvsqm = 0       # Normal or HVS based method

#dctkrnl = ['T3'] # cv2.dct()
dctkrnl = ['T','T0','T1','T2','T3','T4']


if name == 'mp4' or name == 'MP4' or name == 'mpg'or name == 'avi' or name == 'AVI' or name == 'mov':
    
    read=False
    i=0
    MI=[]
    MB=[]
    MP=[]
    NBITS=[]
    for dct in dctkrnl:
        mpeg = codec.Encoder(fileName, quality, sspace, mode, search, hvsqm, flat, p, dct)
        mpeg.run()
        
#        NBITS.append(sum(mpeg.ffNUMBITS[:mpeg.nframes]))
        MI.append(np.mean(mpeg.NBITS_I[mpeg.NBITS_I>0.0]))
        MB.append(np.mean(mpeg.NBITS_B[mpeg.NBITS_B>0.0]))
        MP.append(np.mean(mpeg.NBITS_P[mpeg.NBITS_P>0.0]))
        
        path = fileName.split('/')
        
#        means = np.mean(mpeg.NBITS_I[mpeg.NBITS_I>0.0]),np.mean(mpeg.NBITS_B[mpeg.NBITS_B>0.0]),np.mean(mpeg.NBITS_P[mpeg.NBITS_P>0.0])
        means = MI[-1],MB[-1],MP[-1]
        textstr = u'Médias\nQuadros-I:%.2f kbit\nQuadros-B:%.2f kbit\nQuadros-P:%.2f kbit'%(means[0]/1e3,means[1]/1e3,means[2]/1e3)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        xcoord = range(mpeg.nframes)
        ax.plot(xcoord, np.ones(len(xcoord))*MI[-1],'-',color="red")
        ax.plot(xcoord, np.ones(len(xcoord))*MB[-1],'-',color="blue")
        ax.plot(xcoord, np.ones(len(xcoord))*MP[-1],'-',color="green")
        ax.bar(xcoord, mpeg.NBITS_I[:mpeg.nframes], width=0.2, edgecolor="red", color="red")
        ax.bar(xcoord, mpeg.NBITS_B[:mpeg.nframes], width=0.2, edgecolor="blue", color="blue")
        ax.bar(xcoord, mpeg.NBITS_P[:mpeg.nframes], width=0.2, edgecolor="green", color="green")
        ax.set_xlabel("Quadros")
        ax.set_ylabel(u"Nùmeros de Bits (kbits)")
        locs, label = plt.yticks()
        plt.yticks(locs, map(lambda x: "%4.f" % x, locs/1e3))
        ax.set_title(path[-1].split('.')[0]+'-'+dct)
        ax.legend(["I-frame","B-frame","P-frame"], fancybox=True, framealpha=0.9, loc=4).get_frame().set_facecolor('wheat')
        ax.text(.04, .21, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
        ax.grid()
    
    #    name = './outputs/normal/' + mode + '/' + 'dctkernel' + '/' + path[-1].split('.')[0] + '/' + quality + '/'+ path[-1].split('.')[0] +'-'+dctkrnl+'-stats1enc' + '.png'
        name = mpeg.FILEDIR + path[-1].split('.')[0] +'-'+dct+'-stats1enc' + '.png'
        print "\nSavePicOn: ", name
        plt.savefig(name)
        if read==False:
            assesment = open(mpeg.FILEDIR + path[-1].split('.')[0]+'-ENCDATA.txt','w')
            read=True
        else:
            assesment = open(mpeg.FILEDIR + path[-1].split('.')[0]+'-ENCDATA.txt','a')
        data = "# "+path[-1].split('.')[0]+" - "+str(quality)+"\n"+dct+"\n"+\
        "-NUMBITS: %d"%sum(mpeg.ffNUMBITS[:mpeg.nframes])+"\n"+\
        "-NBITS_I: %d"%MI[-1]+"\n"+"-NBITS_B: %d"%MB[-1]+"\n"+"-NBITS_P: %d"%MP[-1]+"\n"+\
        "-MFRAMES: %.4f(BIT/FRAMES)"%np.mean(mpeg.ffNUMBITS[:mpeg.nframes])+"\n"+\
        "-BITRATE: %.4f"%(np.mean(mpeg.avgBits))+"\n"+\
        "-COMRATE: %.4f"%(np.mean(mpeg.CRate))+"\n"+\
        "-REDUND:  %.4f"%(np.mean(mpeg.redundancy))+"\n\n"
        assesment.write(data)
        assesment.close()
        print "\n%s Measures %s" % (5*'#', 5*'#')
        print"Número total de bits: ", sum(mpeg.ffNUMBITS[:mpeg.nframes])
        print "Media (bits/frame): %.2f" % np.mean(mpeg.ffNUMBITS[:mpeg.nframes])
        print "Bitrate = %.4f bits/pixel" % (np.mean(mpeg.avgBits))
        print "Compression rate = %.4f" % (np.mean(mpeg.CRate))
        print "Redundancy = %.4f" % (np.mean(mpeg.redundancy))
        filedir = mpeg.FILEDIR
#        nframes = mpeg.nframes
#        del mpeg
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    xcoord = range(len(dctkrnl))
    ax.plot(xcoord, MI,'-o',color="red")
    ax.plot(xcoord, MB,'-o',color="blue")
    ax.plot(xcoord, MP,'-o',color="green")
#    ax.plot(xcoord, NBITS, '-o')
    ax.set_xlabel(u"Matrizes de aproximação", fontsize=20)
    ax.set_ylabel(u"Número médio de bits (kbits)", fontsize=20)
    locs, label = plt.xticks()
    plt.xticks(locs, dctkrnl, fontsize=18)
    locs, label = plt.yticks()
    plt.yticks(locs, map(lambda x: "%4.f" % x, locs/1e3))
    ax.set_title(path[-1].split('.')[0])
    ax.legend(["Quadro-I","Quadro-B","Quadro-P"], fancybox=True, fontsize=18, framealpha=0.9, loc=4) #.get_frame().set_facecolor('wheat')
#    ax.text(.04, .21, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
    ax.grid()
    plt.savefig(filedir+path[-1].split('.')[0]+'-ESTATSKERNEL'+'.png')
                
elif name == 'txt' or name == 'TXT':
    
#    vdir = askdirectory().__str__()
    flnm = fileName.split('/')[-1].split('.')[0].split('-')[0]
    vdir = fileName.strip(fileName.split('/')[-1])
    
    videoName = askopenfilename(parent=root, title="Enter with the original video file.").__str__()
    path = fileName.split('/')
    
    print "File Name:"+vdir+flnm
    
    fo=open(vdir+flnm+'-ENCDATA.txt','a')
    
    PSNRmeans = []
    MSSIMmeans = []
    seq = []
    mssim = 0
    mpeg=[]
    
    for dct in dctkrnl:
        fileName = vdir+flnm+'-'+dct+'.txt'
        
        del mpeg
    
#    figurename = './outputs/normal/' + path[-3] + '/' + path[-2] + '/' + path[-1].split('.')[0] + '.png'
        figurename = fileName.split('.')[0]
        print "\nFigure name: " + figurename
    
        mpeg = codec.Decoder(fileName, videoName)    #    seq, psnrValues, msimValues, typeseq = mpeg.run()
        mpeg.run()
    
        print "3) Computing visual metrics. \nPlease wait..." #% ("#"*4,"#"*4)
        mssimValues = []
        psnrValues = []
    
        for i in range (mpeg.nframes):
#            seq.append(mpeg.VIDEOREC[i][1])
            mssimValues.append(metrics.mssim(mpeg.VIDEO[i][:,:,0], mpeg.VIDEOREC[i][1][:,:,0]))
            psnrValues.append(metrics.psnr1c(mpeg.VIDEO[i][:,:,0], mpeg.VIDEOREC[i][1][:,:,0]))
        
        MSSIMmeans.append(np.mean(np.array(mssimValues)))
        PSNRmeans.append(np.mean(np.array(psnrValues)))
        
        if dct != 'T' and mssim < MSSIMmeans[-1]:
            mssim = MSSIMmeans[-1]
            videoname=flnm+'-'+dct
            for i in range (mpeg.nframes):
                seq.append(mpeg.VIDEOREC[i][1])
                if i<6:
                    seq[i][seq[i]>255.0] = 255.0
                    seq[i][seq[i]<0.0] = 0.0
                    cv2.imwrite(vdir+"/"+"data"+"/"+flnm+"-"+str(i)+"-rec.png", cv2.cvtColor(np.uint8(seq[i]), cv2.COLOR_YCR_CB2BGR ))
                    cv2.imwrite(vdir+"/"+"data"+"/"+flnm+"-"+str(i)+"-org.png", cv2.cvtColor(np.uint8(mpeg.VIDEO[i]), cv2.COLOR_YCR_CB2BGR ))

        means = np.mean(mpeg.NBITS_I[mpeg.NBITS_I>0.0]),np.mean(mpeg.NBITS_B[mpeg.NBITS_B>0.0]),np.mean(mpeg.NBITS_P[mpeg.NBITS_P>0.0])
        textstr = 'Means\nI-frames:%.2f kbit\nB-frames:%.2f kbit\nP-frames:%.2f kbit'%(means[0]/1e3,means[1]/1e3,means[2]/1e3)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        xcoord = range(mpeg.nframes)
        ax.plot(xcoord, np.ones(len(xcoord))*np.mean(mpeg.NBITS_I[mpeg.NBITS_I>0.0]),'-',color="red")
        ax.plot(xcoord, np.ones(len(xcoord))*np.mean(mpeg.NBITS_B[mpeg.NBITS_B>0.0]),'-',color="blue")
        ax.plot(xcoord, np.ones(len(xcoord))*np.mean(mpeg.NBITS_P[mpeg.NBITS_P>0.0]),'-',color="green")
        ax.bar(xcoord, mpeg.NBITS_I[:mpeg.nframes], width=0.2, edgecolor="red", color="red")
        ax.bar(xcoord, mpeg.NBITS_B[:mpeg.nframes], width=0.2, edgecolor="blue", color="blue")
        ax.bar(xcoord, mpeg.NBITS_P[:mpeg.nframes], width=0.2, edgecolor="green", color="green")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Number of Bits (kbits)")
        locs, label = plt.yticks()
        plt.yticks(locs, map(lambda x: "%4.f" % x, locs/1e3))
        ax.set_title(flnm+'-'+dct)#path[-1].split('.')[0])
        ax.legend(["I-frame","B-frame","P-frame"], fancybox=True, framealpha=0.9, loc=4).get_frame().set_facecolor('wheat')
        ax.text(.04, .21, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
        ax.grid()
    #    name = './outputs/normal/' + mode + '/' + 'dctkernel' + '/' + path[-1].split('.')[0] + '-stats1dec' + '.png'
        name = figurename + '-stats1dec' + '.png'
        plt.savefig(name)
        print"Número total de bits: ", sum(mpeg.decNUMBITs[:mpeg.nframes]) 
        
        #############
        ### PLOTS ###
    #    futils.write_sequence_frames(seq, mpeg.mode, mpeg.hvsqm, fileName)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(211)
        psnrMean = np.mean(np.array(psnrValues)[:,0])
        ax1.plot(range(len(psnrValues)), np.ones(len(psnrValues))*psnrMean, '-',label=u"Média")
        ax1.plot(range(len(psnrValues)), np.array(psnrValues)[:,0], '-*',label="Valores",)
        legend1 = ax1.legend(loc=4, fancybox=True, framealpha=0.6)
        ax1.set_title(flnm+'-'+dct, fontsize=20)#path[-1].split('.')[0])
        ax1.set_xlabel("Quadros", fontsize=18)
        ax1.set_ylabel("PSNR(db)", fontsize=18)
        ax1.grid()
        
        ax2 = fig1.add_subplot(212)
        msimMean = np.mean(np.array(mssimValues)) #[:,0])
        ax2.plot(range(len(mssimValues)), np.ones(len(mssimValues))*msimMean, '-', label=u"Média")
        ax2.plot(range(len(mssimValues)),np.array(mssimValues), '-*', label="Valores")
        legend2 = ax2.legend(loc=4, fancybox=True, framealpha=0.6)
        ax2.set_xlabel("Quadros", fontsize=18)
        ax2.set_ylabel("MSSIM", fontsize=18)
        ax2.grid()
        plt.subplots_adjust(hspace=0.4)
        plt.savefig(figurename+'.png')
        
    tofile = '\n'+dct+'\nmssimeans: '+str(MSSIMmeans)+'\npsnrmeans: '+str(PSNRmeans) 
    fo.write(tofile)
    fo.close()
    
    ## MEDIAS DE TODOS OS NUCLEOS Ti MSSIM E PSNR ##
    f = plt.figure()
    ax = f.add_subplot(111)
    xcoord = range(len(dctkrnl))
    p1=ax.plot(xcoord, MSSIMmeans,'-o',color="blue", lw=2.)
    ay = ax.twinx()
    ay.plot(xcoord, PSNRmeans, '-o', color="green", lw=2.)
    ax.set_xlabel(u"aproximações DCT")
    ax.set_ylabel("MSSIM")
    ay.set_ylabel("PSNR")
    locs, label = plt.xticks()
    plt.xticks(locs, dctkrnl)
    ax.set_title(flnm)#path[-1].split('.')[0])
    ax.legend(["MSSIM"],fancybox=True, framealpha=0.9, loc=2).get_frame().set_facecolor('wheat')
    ay.legend(["PSNR"],fancybox=True, framealpha=0.9, loc=1).get_frame().set_facecolor('wheat')
#    ax.text(.04, .21, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
    ax.grid()
    plt.savefig(vdir+flnm+'-ASSESKERNEL'+'.png')

    ### PLAYER ###
    print "\n### [PLAYER] > ###\n Press\n - \'p\' to play\n - \'f\' to step-by-step\n - \'b\' to stepback\n - \'q\' to quit"
    play=True
    f = -1
    fator = 0
    press=False
    N,M,D = mpeg.shape
#    videoname = path[-1].split('.')[0]
    while play:
        if f == -1 or f >= mpeg.nframes:
            cv2.imshow(videoname,np.zeros((mpeg.shape)))
        else:
            seq[f][seq[f]>255.0] = 255.0
            seq[f][seq[f]<0.0] = 0.0
            cv2.imshow(videoname, cv2.cvtColor(np.uint8(seq[f][:N,:M]), cv2.COLOR_YCR_CB2BGR))
        k=cv2.waitKey(fator)
            
        if k==-1:
            f += 1
        elif k==ord('q'):
            play=False
            cv2.destroyAllWindows()
        elif k==ord('p') or k==ord(' '):
            if press == False:
                fator = int((1./mpeg.fps)*1000)
                press=True
            elif press==True:
                fator = 0
                press = False   #            f += 1
        elif k==ord('f'):
            f += 1
            fator = 0
        elif k==ord('b'):
            f -= 1
            fator = 0
        elif k==ord('r'):
            fator = int((1./mpeg.fps)*1000)
            press = False
            f = -1
        #Checa os limites da sequencia
        if f < -1:
            f = -1
            fator = 0
        elif f >= mpeg.nframes:
            f = mpeg.nframes
            fator = 0
### END PLAYER ###

else:
    print('Invalid filename!!!!')

