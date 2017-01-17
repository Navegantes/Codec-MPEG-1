# -*- coding: utf-8 -*-
"""
Created on Tue May 03 19:48:09 2016

@author: Navegantes
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


#### PLOT TEST ####
#x = range(90)
#values = 200000*np.random.rand(90)

dctkrnl = ['T','T0','T1','T2','T3','T4']

# "akiyo","ice","stefan","bus","garden","tennis","tempete","crew","football","husky", "foreman", "waterfall"
#                        T                    T0                   T1                     T2                  T3                  T4
mssim = np.array([[0.96870580286888919, 0.95201557782683344, 0.93390193338743344, 0.9508675613202352,  0.95883409506122463, 0.94854602116975328], #akiyo
                  [0.96568548883090943, 0.94640335544710186, 0.93117029594873035, 0.94513894402465037, 0.95501539719122885, 0.94486789976260188], #ice
                  [0.9636810576109236 , 0.93937955463190503, 0.9247274758135422 , 0.9367081011534405 , 0.95113670628214719, 0.93955668605420728], #stefan
                  [0.93778724612818409, 0.91049872699561585, 0.87986720643938043, 0.90675435725615594, 0.9266789949114711 , 0.91175252961949615], #bus
                  [0.93732225656031709, 0.91519453994526345, 0.90220308136913541, 0.91351271724968541, 0.9264489134148558 , 0.91632504446174012], #garden
                  [0.90678532825585612, 0.8893199249042798 , 0.8737478565343838 , 0.88878970229360954, 0.89891891185780237, 0.89016090803742642], #tennis
                  [0.95245099929884958, 0.93169977967249051, 0.90915114052530599, 0.93005911813053133, 0.94485915302932144, 0.9321826794961271 ], #tempete
                  [0.92542247161724966, 0.88506386351277588, 0.84931222245857252, 0.88279196344913013, 0.90506522727543703, 0.88516030455657047], #crew
                  [0.93428351452236824, 0.88864327684692423, 0.85705047641553156, 0.88578114589583801, 0.90953628337710624, 0.88897160696600952], #football
                  [0.92183722629676368, 0.88616828167757744, 0.86377038645695736, 0.88269669789583827, 0.9040308096944748 , 0.88796155647645225], #husky
			   [0.94812296333987467, 0.91858474721179051, 0.89208923412160512, 0.91762429971362436, 0.93343432025092477, 0.9177549521118491 ], #foreman
			   [0.92996431975250249, 0.89715452352969827, 0.84753171154041329, 0.89151942141652718, 0.91587311055432241, 0.90035049783540533]])#waterfall

psnr  = np.array([[38.246162639008311, 35.672538086449592, 34.717234024884597, 35.630950623351517, 36.663529143502494, 35.671726775612285], #akiyo
                  [36.729039437989996, 33.971781767416751, 32.922559564205805, 33.824734058336929, 35.074742733443188, 34.018634904381358], #ice
                  [31.619198721240732, 28.704186640217969, 27.814381618639104, 28.408016235589624, 29.90931721096818 , 28.833986557702051], #stefan
                  [30.798338924351423, 29.797253689487327, 28.742920214037756, 29.593883798832746, 30.687091802054614, 29.947304297117849], #bus
                  [27.633926116797916, 26.169610805103986, 25.785977274133042, 26.06622427625592 , 26.81258102364211 , 26.261909693456474], #garden
                  [31.59680608031292 , 30.450190792170723, 30.009368095815457, 30.443630827205318, 30.967353716881203, 30.470504507377832], #tennis
                  [31.243914746610038, 30.492514178897256, 29.454876455343431, 30.327160608144286, 31.488020808944793, 30.640049373285486], #tempete
                  [35.549814165958935, 33.28987009169105 , 31.967406770710426, 33.155672616245482, 34.282812115070087, 33.364130373707084], #crew
                  [34.771397023251112, 31.629096260383331, 30.560510525142583, 31.480555195403529, 32.82606956199114 , 31.722753587612473], #football
                  [27.329860610393268, 25.422953395013796, 24.804478846481445, 25.254904635988073, 26.222918646433577, 25.525585204229632], #husky
			   [34.992047934613353, 32.291090885477075, 31.049241232673786, 32.165760485019696, 33.434470564831109, 32.343345720179627], #foreman
			   [33.792273538807066, 31.707435029927446, 30.023473365139548, 31.468856626871023, 32.764971378508413, 31.891282882842102]])#waterfall

smean = np.mean(mssim,0)
pmean = np.mean(psnr,0)

#mframe = np.array([[31214.0433 , 31921.2633 , 30755.5700 , 31654.4200 , 31511.4700 , 32029.2900 ],[ 54056.2542,  54969.7833,  54113.1708,  54938.4292,  54575.0167,  55014.5208],
#                   [136878.9222, 141005.9778, 143522.8667, 141911.7000, 138775.4000, 140368.3333],[156733.5267, 160184.9267, 159392.0133, 160633.7533, 158522.8400, 159266.5067],
#                   [171503.2435, 173977.7130, 175941.4696, 174469.4609, 172287.3913, 173414.7391],[ 68600.2133,  69165.0733,  68333.6133,  69283.2000,  68867.6733,  69181.9133],
#                   [138046.7667, 143059.9500, 143714.7333, 143275.5167, 140065.3667, 141985.0333]])
          
#frameans = np.mean(mframe,0)

seqnames = ["akiyo","ice","stefan","bus","garden","tennis","tempete","crew","football", "husky","foreman","waterfall"]
ln = ['-o','--s','-.v',':p','-->','-D','-*', '--^','-h']
tm=[r'$T$',r'$T_{0}$',r'$T_{1}$',r'$T_{2}$',r'$T_{3}$',r'$T_{4}$']
######## PLOTING PSNR's ########
fig = plt.figure()
ax = plt.subplot(111)
#
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#
i=0
for fr in psnr[:9]:
    ax.plot(range(6), fr, ln[i], linewidth=1.5, markersize=10)
    i+=1
#
ax.set_xlabel(u'Matrizes de Transformação',fontsize=22)
ax.set_ylabel("PSNR",fontsize=20)
locs, label = plt.xticks()
plt.xticks(locs, tm, fontsize=20)
plt.grid()
ax.legend(seqnames, fancybox=True, framealpha=0.7, fontsize=18, loc=4).get_frame().set_facecolor('wheat')
#
##### media #####
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(range(6), pmean, '-o',color="blue",linewidth=2, markersize=10)
ax.set_xlabel(u'Matrizes de Transformação',fontsize=22)
ax.set_ylabel("PSNR",fontsize=20)
locs, label = plt.xticks()
plt.xticks(locs, tm, fontsize=18)
plt.grid()
#
######## PLOTING MSSIM's ########
fig = plt.figure()
ax = plt.subplot(111)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)

i=0
for fr in mssim[:9]:
    ax.plot(range(6), fr, ln[i], linewidth=1.2, markersize=10)
    i+=1

ax.set_xlabel(u'Matriz de Transformação',fontsize=22)
ax.set_ylabel("MSSIM",fontsize=20)
locs, label = plt.xticks()
plt.xticks(locs, tm, fontsize=20)
plt.grid()
ax.legend(seqnames, fancybox=True, framealpha=0.7, fontsize=18, loc=4).get_frame().set_facecolor('wheat')

#### media #####
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(range(6), smean, '-s',color="red", linewidth=2, markersize=10)
ax.set_xlabel(u'Matriz de Transformação',fontsize=20)
ax.set_ylabel("MSSIM",fontsize=20)
locs, label = plt.xticks()
plt.xticks(locs, tm, fontsize=18)
plt.grid()


#ax2 = fig.add_subplot(111)
#ax.text(.04, .21, textstr, transform=ax.transAxes, fontsize=13,verticalalignment='top', bbox=props)
#locs, label = plt.yticks()
#plt.yticks(locs, map(lambda x: "%4.f" % x, locs/1e3))

########################################################################

bpp = [[0.3096, 0.3168, 0.3052, 0.3141, 0.3126, 0.3179],  # akiyo
       [0.5348, 0.5440, 0.5355, 0.5437, 0.5400, 0.5445],  # ice
       [1.3553, 1.3967, 1.4219, 1.4057, 1.3745, 1.3905],  # stefan
       [1.3610, 1.3863, 1.3843, 1.3905, 1.3743, 1.3833],  # bus
       [1.7913, 1.8171, 1.8376, 1.8222, 1.7995, 1.8113],  # garden
       [0.8974, 0.9049, 0.8940, 0.9064, 0.9009, 0.9051],  # tennis
       [0.9811, 0.9874, 0.9674, 0.9894, 0.9877, 0.9896],  # tempete
       [0.9832, 0.9776, 0.9328, 0.9772, 0.9822, 0.9812],  # crew
       [1.2692, 1.3014, 1.2913, 1.3076, 1.2889, 1.3004],  # football
       [2.4328, 2.4957, 2.5359, 2.5076, 2.4566, 2.4843],  # husky
	  [0.8559, 0.8595, 0.8270, 0.8576, 0.8600, 0.8622],  #foreman
       [0.9531, 0.9345, 0.8847, 0.9355, 0.9498, 0.9447]]  #waterfall
      #   T       T0       T1     T2       T3      T4

ln = ['o','s','v','p','d','*']
clr = ['b','g','r', 'c','y','m','k','b','g']

row=3; col=3
a=0
sqnm = 0
#tm=[r'$T$',r'$T_{0}$',r'$T_{1}$',r'$T_{2}$',r'$T_{3}$',r'$T_{4}$']

data=[psnr, mssim]
label=["PSNR (db)","MSSIM"]

#for k in [0,1]:
gs = gridspec.GridSpec(row,col)
f = plt.figure()
for i in range(row):
    for j in range(col):
        ax = f.add_subplot(gs[i, j])
        ax2 = ax.twinx()
        for t in range(6):
            #ax.set_title(seqnames[a])
            print seqnames[a], a
            ax.plot(bpp[a][t], data[0][a][t], ln[t],color='k', markersize=10, markeredgecolor='w')
            ax2.plot(bpp[a][t], data[1][a][t], ln[t],color='b', markersize=10, markeredgecolor='w')
#            ax1.plot([min(bpp[a]), max(bpp[a])],[max(mssim[a]), min(mssim[a])],"-k", linewidth=0.5)
#            ax1.scatter(bpp[a][t], psnr[a][t], marker=ln[t],color='k', linewidths=2)
            ax.set_xlabel("bpp", fontsize=14)
            if j==0:
                ax.set_ylabel(label[0], fontsize=14)
            ax.set_xmargin(0.01)
            ax.set_ymargin(0.1)
            if j==2:
                ax2.set_ylabel(label[1], fontsize=14, color='b')
            ax2.set_xmargin(0.01)
            ax2.set_ymargin(0.1)
            for tl in ax2.get_yticklabels():
                tl.set_color('b')
        a += 1
        plt.grid()
        #gs.tight_layout()
    #a += 1
#    a=0
gs.update(hspace=0.28, wspace=0.22)
plt.legend(tm, loc=4, ncol=1, numpoints=1,bbox_to_anchor=(1.42, -0.09), fontsize=20, fancybox=True, framealpha=0.5,borderpad=.5,labelspacing=.5,handletextpad=.02).get_frame().set_facecolor('wheat')
plt.show()



####################################################################################################
####################################################################################################


#row=3; col=3
#gs1 = gridspec.GridSpec(row,col)
#gs2 = gridspec.GridSpec(row,col)
#f = plt.figure()
#a=0
#tm=[r'$T$',r'$T_{0}$',r'$T_{1}$',r'$T_{2}$',r'$T_{3}$',r'$T_{4}$']
##plt.suptitle("EXEMPLO")
#
#for i in range(row):
#    for j in range(col):
#        ax1 = f.add_subplot(gs1[i, j])
#        for t in range(6):
#            ax1.set_title(seqnames[a])
#            ax1.plot(bpp[a][t], psnr[a][t], ln[t],color='k', markersize=8, markeredgecolor='w')
##            ax1.plot([min(bpp[a]), max(bpp[a])],[max(mssim[a]), min(mssim[a])],"-k", linewidth=0.5)
##            ax1.scatter(bpp[a][t], psnr[a][t], marker=ln[t],color='k', linewidths=2)
#            ax1.set_xlabel("bpp")
#            ax1.set_ylabel("PSNR")
#            ax1.set_xmargin(0.2)
#            ax1.set_ymargin(0.2)
#        a+=1
#        plt.grid()
#gs1.update(hspace=0.3, wspace=0.2)
##plt.tight_layout(pad=0.2)
##plt.legend(tm, loc=4, ncol=1, numpoints=1,bbox_to_anchor=(-0.22, -0.16), fancybox=True, framealpha=0.5,borderpad=.5,labelspacing=.5,handletextpad=.02).get_frame().set_facecolor('wheat')
#plt.legend(tm, loc=4, ncol=1, numpoints=1,bbox_to_anchor=(1.25, 0.1), fancybox=True, framealpha=0.5,borderpad=.5,labelspacing=.5,handletextpad=.02).get_frame().set_facecolor('wheat')
#plt.show()
#
#f = plt.figure()
#a=0
#for i in range(row):
#    for j in range(col):
#        ax2 = f.add_subplot(gs2[i, j])
#        for t in range(6):
#            ax2.set_title(seqnames[a])
#            ax2.plot(bpp[a][t], mssim[a][t], ln[t],color='k', markersize=8, markeredgecolor='w')
##            ax2.scatter(bpp[a][t], mssim[a][t], marker=ln[t],color='k', linewidths=2)
#            ax2.set_xlabel("bpp")
#            ax2.set_ylabel("MSSIM")
#            ax2.set_xmargin(0.2)
#            ax2.set_ymargin(0.2)
#        a+=1
#        plt.grid()
#gs2.update(hspace=0.3, wspace=0.2)
##plt.tight_layout(pad=0.2)
##plt.legend(tm, loc=4, ncol=1, numpoints=1,bbox_to_anchor=(-0.22, -0.16), fancybox=True, framealpha=0.5,borderpad=.5,labelspacing=.5,handletextpad=.02).get_frame().set_facecolor('wheat')
#plt.legend(tm, loc=4, ncol=1, numpoints=1,bbox_to_anchor=(1.25, 0.1), fancybox=True, framealpha=0.5,borderpad=.5,labelspacing=.5,handletextpad=.02).get_frame().set_facecolor('wheat')
#plt.show()





#    a=0

#f1, ax1 = plt.subplots(2, 3, sharey=True)
#f2, ax2 = plt.subplots(1, 3, sharey=True)
#ax1.plot(x, y)
#ax1.set_title('Sharing Y axis')
#ax2.scatter(x, y)

        
#f.subplots_adjust(wspace=0.4)
#locs, label = plt.yticks()
#plt.yticks(locs, map(lambda p: "%.2f" % p, locs))

#ax.set_xlabel("bpp")
#ax.set_ylabel("MSSIM")
#ax.set_ylabel("PSNR")

#ax.legend(["T","T0","T1","T2","T3","T4"], fancybox=True, framealpha=0.5, loc=4).get_frame().set_facecolor('wheat')





