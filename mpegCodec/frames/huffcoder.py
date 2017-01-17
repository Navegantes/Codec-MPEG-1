# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 21:52:39 2015

@author: Gonçalves
@author: Navegantes
"""

import bitstring as bs
import numpy as np
#import cv2


class HuffCoDec:
    '''
    Classdoc
    '''
    
    def __init__(self, hufftables):
        
        self.DCLumaTB, self.DCChromTB, self.ACLumaTB, self.ACChromTB = hufftables
        self.DiffCat = [ (0), (1),(2,3), (4,7), (8,15), (16,31), (32,63), (64,127), (128,255), (256,511),
                         (512,1023), (1024,2047),(2048,4095), (4096,8191), (8192,16383), (16384,32767) ]
        
    def tobin(self, coef, cat):
        '''
        Gera a sequência de bits equivalênte de 'coef'.
        
        input: coef->int, cat->int
        output: sout -> string
        '''
        sout=[]                                              #Cadeia de bits de saída
        if coef == 0:
            return ''
        n = cat+1                                            #Numero de bits para gerar os binários de numb
        if coef<0:
    #       d = bs.BitStream(int=int(2**n/2)-1, length=n)    #Para bitwise de inversao dos bits
            d = bs.BitStream(int=-1, length=n)
            b = bs.BitStream(int=abs(coef), length=n)        #Gera bitstream de numb
            sout = b^d                                       #Inversão dos bits (XOR)'
        else:
            sout = bs.BitStream(int=coef, length=n)
            
        return sout.bin[1:]
    
    def fwdhuff(self, DCant, coefseq, ch):
        '''
        Gera o codigo Huffman dos coeficientes.
        
        input: DCant -> int
               coefseq -> lista.
               ch -> int
        output: (nbits, huffcode) -> tupla.
        '''
        sz, DC = 0, 0; cdwrd, ACs = 1, 1
        nbits=0; huffcode = ''
        #Gera codigo huffman do coeficiente DC
        DCcur = coefseq[DC]                   #Coeficiente DC corrente
        dif = DCcur - DCant                   #Diferença coeficiente DC atual e anterior
        cat = self.coefcat(dif)               #Categoria da diferença do coeficiente DC
        bitstr = self.tobin(dif, cat)         #Converte a diferença para representaçao binária
        if ch == 0:
            DCcode = self.DCLumaTB[cat]       #Consulta a tabela - out:(size, codeword)
        else:
            DCcode = self.DCChromTB[cat]

        DChfcode = DCcode[cdwrd] + bitstr     #Código Huffman para o coef DC - huffcode + repr binária
        nbits = cat + DCcode[sz]
        huffcode = DChfcode
        
        if len(coefseq)>1:                    #Gera codigo huffman dos coeficientes AC
            run=0
            for AC in coefseq[ACs:]:
                if AC==0:
                    run+=1                                #Conta o numero de zeros que precedem o coef
                else:
                    cat = self.coefcat(AC)                #Categoria do coef
                    bitstr = self.tobin(AC, cat)          #Representação binária
                    if ch==0:                             #Seleciona tabela luma-chroma
                        ACcode = self.ACLumaTB[(run,cat)] #Consulta tabela - out:(size, codeword)
                    else:
                        ACcode = self.ACChromTB[(run,cat)]
                    AChfcode = ACcode[cdwrd] + bitstr   #Código Huffman do coef AC
                    nbits += cat + ACcode[sz]         #Calcula numero de bits
                    huffcode += AChfcode
                    run=0
                if run==16:                           #15 zeros seguidos de um zero
                    if ch == 0:
                        huffcode += '11111111001'
                        nbits += 11
                        run=0
                    else:
                        huffcode += '1111111010'
                        nbits += 10
                        run=0
        # Se houver apenas um componente na seq(DC) adiciona o fim de bloco
        if ch == 0:
            huffcode += '1010'                            #Inclui EOB no final
            nbits += 4
        else:
            huffcode += '00'
            nbits += 2
            
        return (nbits, huffcode)
        
    def invhuff(self, seqhuff, ch):
        '''
        '''
        DCant = 0
        bscdmax = 16
        ini = 0; leng = 2;
        SZseq = len(seqhuff)
        nblocks = 0; cat = -1
        basecode = ''; coefsq = []; seqaux = []
        
        basecode = seqhuff[ini:leng]
        value=(leng, basecode)
#        print ini, leng, 'init', value

        if ch <= 0:
            tabDC = self.DCLumaTB
            tabAC = self.ACLumaTB
            cdwrdMax = 9
        else:
            tabDC = self.DCChromTB
            tabAC = self.ACChromTB
            cdwrdMax = 11
        
        while leng < SZseq:
#            print '** (RE)INICIATE DECODING ** - leng-sizeseq', leng, SZseq
#            print ini, leng, "Verify DC coef zero: ", value
        #TRATAMENTO COEFICIENTE DC
            cat = -1
            if tabDC[0]==value:
                #cat = 0
                magdif = 0      #MAGNITUDE DA DIFERENÇA DCi-DCi-1
                coefDC = magdif + DCant
                seqaux.append(coefDC)
                DCant = coefDC
                
                ini = leng
                if ch == 0: leng+=4
                else:       leng+=2
#                print ini, leng, 'DC zero OK', value, seqaux
            else: #ini=0  #confirma
                #VERIFICA O CODIGO BASE
                for sz in range(ini+2, ini+cdwrdMax+1):
                    leng=sz
                    basecode = seqhuff[ini:leng]
                    value=(leng-ini, basecode)
#                    print ini, leng, 'verifying basecode DC-', value
                    #Procura categoria dada a chave de valores (size, basecode)
                    for key in tabDC.iterkeys():
                        if tabDC[key]==value:
                            cat = key
#                            print 'category founded - ', cat, ':', value
                    if cat != -1: break
#                    print 'deny'
                ini = leng
                leng = ini + cat
                magBin = seqhuff[ini:leng]
                magInt = bs.BitStream(bin=magBin).int
#                print ini, leng, 'Verify Magnitude DC-r: ', magInt, magBin        #magInt = bs.BitStream(bin=seqhuff[ini:leng]).int
                
                if magInt == 0 and cat == 1:
                    magdif = -1
                elif magInt == -1 and cat == 1:
                    magdif = 1
                elif magInt < 0:
                    magdif = ( self.DiffCat[cat][1]+magInt )+1
                else: # magInt < -1:
                    magdif = (-1*self.DiffCat[cat][1])+magInt
                    
                ini = leng
                if ch == 0: leng = ini + 4
                else:       leng += 2
#                print ini, leng, 'DC complete-magdif', magdif, magdif + DCant
            #FIM DO ELSE          #coefDC = float(magdif + DCant)
                coefDC = magdif + DCant
                seqaux.append(coefDC)
#                print 'SEQUENCIAaux-DC: ', seqaux, magdif + DCant, basecode + magBin
                DCant = coefDC
            
            basecode = seqhuff[ini:leng]
            if (basecode == '1010') or (basecode == '00'):          #TESTA EOB 
                nblocks += 1                #SE SIM, CONTA MAIS BLOCO
                ini = leng;
                leng= ini + 2               #AJUSTA PARA PROXIMO BASECODE
                basecode = seqhuff[ini:leng]
#                print ini, leng, 'EOB founded-nblocks ', nblocks
                value = (leng-ini, basecode)
                coefsq.append(seqaux)
                seqaux = []             #Reinicia seqaux para nova sequencia
                #VOLTA PARA O INICIO - linha 122  while leng < SZseq:
            else:
            #TRATAMENTO COEFICIENTES ACs
                leng -= 2
                basecode = seqhuff[ini:leng]
                value = (leng-ini, basecode)
                run, cat = -1, -1
#                print '** AC INICIATED ** ', ini, leng, basecode
                while True: #basecode != '1010': #leng < len(seqhuff):
                #VERIFICA BASECODE
                    run, cat = -1, -1
                    for sz in range(ini+2, ini+bscdmax+1):    #ATE O MAXIMO TAMANHO DE BASECODE
                        leng = sz
                        basecode = seqhuff[ini:leng]
                        value=(leng-ini, basecode)#value=(leng-ini, basecode)
#                        print ini, leng, 'Verifying basecode AC', value
                        #Procura categoria dada a chave de valores (size, basecode)
                        for key in tabAC.iterkeys():
                            if tabAC[key]==value:
                                run, cat = key  #runcat = (run, category)
#                        print 'Basecode verified', (run, cat),':', value
                        if (run, cat) != (-1, -1):
                            break
#                        print 'Deny'
                    #SE BASECODE FOI (0,0) ENCONTROU 'EOB' -> '00' OU '1010'
                    if (run, cat) == (0, 0):
                        nblocks += 1
                        ini += value[0]
                        coefsq.append(seqaux)
                        DCant = coefDC
#                        print 'Fim de bloco - Adiciona seqaux: ', seqaux
                        seqaux = []
                    elif (run, cat) == (15, 0):
                    #ADICIONA RUN ZEROS NA SEQUENCIA
#                        print 'SEQUENCIA DE 16 ZEROS ENCONTRADA!!'
                        for zr in range(run+1):
                            seqaux.append(0)
                        ini += value[0]
                    else:
                        if run > 0:
#                            print 'Adicionando', run, 'zeros'
                            for zr in range(run):
                                seqaux.append(0)
                        
                        ini = leng
                        leng = ini + cat
#                        print ini, leng, 'Depois de verify basecode AC - magBin: ', seqhuff[ini:leng]
                        magBin = seqhuff[ini:leng]
                        magInt = bs.BitStream(bin=magBin).int
#                        print ini, leng, 'Magnitude AC magInt-magBin: ', magInt, magBin
                        if magInt == 0 and cat == 1:
                            magdif = -1
                        elif magInt == -1 and cat == 1:
                            magdif = 1
                        elif magInt < 0:
                            magdif = ( self.DiffCat[cat][1]+magInt )+1
                        else: # magInt < -1:
                            magdif = (-1*self.DiffCat[cat][1])+magInt
                            
                        seqaux.append(magdif)
#                        print 'SEQUENCIAux AC: ', seqaux, basecode + magBin #coefsq
                        ini = leng
                        if ch == 0: leng = ini + 4
                        else:       leng += 2
                        basecode = seqhuff[ini:leng]
                        value = (leng-ini, basecode)
                        
                    if (basecode == '1010') or (basecode == '00'):
                        break
                        
                nblocks += 1
                ini = leng
                leng = ini + 2
                basecode = seqhuff[ini:leng]
                value = (leng-ini, basecode)
#                print "Reinit Cont: ini-leng ", ini, leng, basecode
                coefsq.append(seqaux)
                seqaux = []
        
        return (nblocks, coefsq)
    
    def coefcat(self, mag):
        '''
        Encontra a categoria da magnitude do coeficiente.
        
        input: mag -> int
        output: cat -> int
        '''
        difcat = self.DiffCat  #               difcat = [ (0), (1),(2,3), (4,7), (8,15), (16,31), (32,63), (64,127), (128,255), (256,511), (512,1023), (1024,2047),(2048,4095), (4096,8191), (8192,16383), (16384,32767) ]
        
        if mag < 0: mag = abs(mag)
        if mag > 32767:
            mag=32767
            return 15
        if mag==0:
            return 0
        elif mag==-1 or mag==1:
            return 1
            
        for cat in range(2,len(difcat)):
            if difcat[cat][0]<=mag<=difcat[cat][1]:
                return cat

#FIM CLASS HUFFCODER

def adjImg(shape, blocksize=[8,8]):
    '''
    Ajusta as dimensões da imagem de modo a serem multiplas de 'blocksize'.
    blocksize=[m, n]
    '''
    
    Mo, No, Do = shape
    
    if int(Mo)%blocksize[0] != 0:
        M = Mo + (blocksize[0] - int(Mo)%blocksize[0])
    else:
        M = Mo
        
    if int(No)%blocksize[1] != 0:
        N = No + (blocksize[1] - int(No)%blocksize[1])
    else:
        N = No
        
    newImg = np.zeros((M,N,Do),np.float32)
    #newImg[:Mo,:No] = img
        
    return (M, N, Do), newImg
    
def zigzag(coefs, shp=[8,8]):
    '''
    Retorna a sequência de coeficiêntes ordenados de acordo com o padrão Zigue-Zague
    '''
    
    coefseq=[]
    
    indx = sorted(((x,y) for x in range(shp[0]) for y in range(shp[1])),
                        key = lambda (x,y): (x+y, -y if (x+y) % 2 else y))
                        
    for ind in range(len(indx)):
        coefseq.append( coefs[indx[ind]] )
                
    nelmnt = len(coefseq)   #Numero de elementos em 'coefseq' (ordenados em zig-zag)
    seq1D = []               
    i=-1; nz=0
    while abs(i)<=(nelmnt):#len(coefseq):
        if coefseq[i]==0.0:
            i-=1
            nz += 1
        else:
            seq1D = coefseq[0:(nelmnt-abs(i))+1]
            break
            
    if abs(i)>nelmnt:
        seq1D = [0]
    return seq1D     
                
def zagzig(seq, bshp=[8,8]): #imshape, 
    '''
    '''
    
    block = np.zeros(bshp)
        
    indx = sorted(((x,y) for x in range(bshp[0]) for y in range(bshp[1])),
                        key = lambda (x,y): (x+y, -y if (x+y) % 2 else y))
    if len(seq)>0:                    
        for s in range(len(seq)): #        for t in range(len(seq[s])):
            block[indx[s]] = seq[s]
        
    return np.float_(block)
