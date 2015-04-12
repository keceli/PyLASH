#!/usr/bin/python2.7
import matplotlib.pyplot as plt
import numpy as np
def makeDistanceHistograms():
    mydir='/Volumes/s/keceli/Dropbox/work/SEMO/xyz/'
    
    fig, axes = plt.subplots(nrows=1, ncols=1)
#    plt.setp(axes,xlim=[0,20],ylim=[0,500000],yscale='log')
  #  plt.setp(axes,xlim=[0,30],yscale='log')
  #  plt.setp(axes,xlim=[0.5,125],xscale='log',yscale='log')
    plt.setp(axes,xlim=[0.5,125])#,xscale='log',yscale='log')

    readFile(mydir+'diamond-2r_P2.xyz')
    plt.subplot(1, 1, 1)
    frame1 = plt.gca()
   # frame1.get_xaxis().set_ticklabels([])
    plt.hist(distances(),bins=1000,cumulative=True,color='red',edgecolor='red',alpha=0.7)

    readFile(mydir+'nanowire25-2r_P2.xyz')
    plt.subplot(1, 1, 1)
    plt.ylabel("Frequency")
    plt.hist(distances(),bins=1000,cumulative=True,color='green',edgecolor='green',alpha=0.7)
    frame1 = plt.gca()
  #  frame1.get_xaxis().set_ticklabels([])
    
    readFile(mydir+'nanotube2-r_P2.xyz')
    plt.subplot(1, 1, 1)
    plt.xlabel("Distance")
    plt.hist(distances(),bins=1000,cumulative=True,color='blue',edgecolor='blue',alpha=0.7)

    
    plt.savefig("disthist.png")                  
    return

def makeDistanceHistograms2():
    mydir='/Volumes/s/keceli/Dropbox/work/SEMO/xyz/'
    
    fig, axes = plt.subplots(nrows=3, ncols=1)
#    plt.setp(axes,xlim=[0,20],ylim=[0,500000],yscale='log')
  #  plt.setp(axes,xlim=[0,30],yscale='log')
  #  plt.setp(axes,xlim=[0.5,125],xscale='log',yscale='log')
    plt.setp(axes,xlim=[0.5,125])#,xscale='log',yscale='log')

    readFile(mydir+'diamond-2r_P4.xyz')
    plt.subplot(3, 1, 1)
    frame1 = plt.gca()
   # frame1.get_xaxis().set_ticklabels([])
    plt.hist(distances(),bins=1000,cumulative=True,color='red',edgecolor='red',alpha=0.7)

    readFile(mydir+'diamond-2r_P6.xyz')
    plt.subplot(3, 1, 2)
    plt.ylabel("Frequency")
    plt.hist(distances(),bins=1000,cumulative=True,color='green',edgecolor='green',alpha=0.7)
    frame1 = plt.gca()
  #  frame1.get_xaxis().set_ticklabels([])
    
    readFile(mydir+'diamond-2r_P8.xyz')
    plt.subplot(3, 1, 3)
    plt.xlabel("Distance")
    plt.hist(distances(),bins=1000,cumulative=True,color='blue',edgecolor='blue',alpha=0.7)

    
    plt.savefig("disthist2.png")                  
    return

def makePDFhist():
    mydir='/Volumes/s/keceli/Dropbox/work/SEMO/xyz/'
    
    fig, axes = plt.subplots(nrows=1, ncols=1)
    plt.setp(axes,xlim=[0,140],ylim=[0,0.10])
  #  plt.setp(axes,xlim=[0,30],yscale='log')
  #  plt.setp(axes,xlim=[0.5,125],xscale='log',yscale='log')
  #  plt.setp(axes,xlim=[0.5,125])#,xscale='log',yscale='log')

    readFile(mydir+'diamond-2r_P2.xyz')
    plt.subplot(1, 1, 1)
    frame1 = plt.gca()
   # frame1.get_xaxis().set_ticklabels([])
    plt.hist(distances(),bins=1000,cumulative=False,normed=True,histtype=u'step',color='red',edgecolor='red',alpha=0.7)

    readFile(mydir+'nanowire25-2r_P2.xyz')
    plt.subplot(1, 1, 1)
    plt.ylabel("Frequency")
    plt.hist(distances(),bins=1000,cumulative=False,normed=True,histtype=u'step',color='green',edgecolor='green',alpha=0.7)
 #   frame1 = plt.gca()
  #  frame1.get_xaxis().set_ticklabels([])
    
    readFile(mydir+'nanotube2-r_P2.xyz')
    plt.subplot(1, 1, 1)
    plt.xlabel("Distance")
    plt.hist(distances(),bins=1000,cumulative=False,normed=True,histtype=u'step',color='blue',edgecolor='blue',alpha=0.7)
    plt.ylim([0,0.1])
    plt.xlim([0,20])
    
    plt.savefig("pdfhist.png")                  
    return

def makePDFhist2():
    mydir='/Volumes/s/keceli/Dropbox/work/SEMO/xyz/'
    
    fig, axes = plt.subplots(nrows=3, ncols=1)
    plt.setp(axes,xlim=[0,20],ylim=[0,0.20])
  #  plt.setp(axes,xlim=[0,30],yscale='log')
  #  plt.setp(axes,xlim=[0.5,125],xscale='log',yscale='log')
  #  plt.setp(axes,xlim=[0.5,125])#,xscale='log',yscale='log')

    readFile(mydir+'diamond-2r_P2.xyz')
    distD=distances2()
    readFile(mydir+'nanowire25-2r_P2.xyz')
    distW=distances2()
    readFile(mydir+'nanotube2-r_P2.xyz')
    distT=distances2()

    plt.subplot(3, 1, 1)
    frame1 = plt.gca()
   # frame1.get_xaxis().set_ticklabels([])
    plt.hist(distD[0],bins=1000,cumulative=False,normed=True,histtype=u'step',color='red',edgecolor='red',alpha=0.7)
    plt.hist(distW[0],bins=1000,cumulative=False,normed=True,histtype=u'step',color='green',edgecolor='green',alpha=0.7)
    plt.hist(distT[0],bins=1000,cumulative=False,normed=True,histtype=u'step',color='blue',edgecolor='blue',alpha=0.7)
    plt.xlim([0,20])
    plt.ylim([0,1])
    
    plt.subplot(3, 1, 2)
    plt.ylabel("Frequency")
    plt.hist(distD[1],bins=1000,cumulative=False,normed=True,histtype=u'step',color='red',edgecolor='red',alpha=0.7)
    plt.hist(distW[1],bins=1000,cumulative=False,normed=True,histtype=u'step',color='green',edgecolor='green',alpha=0.7)
    plt.hist(distT[1],bins=1000,cumulative=False,normed=True,histtype=u'step',color='blue',edgecolor='blue',alpha=0.7) #   frame1 = plt.gca()
    plt.xlim([0,20])
    plt.ylim([0,1])
      #  frame1.get_xaxis().set_ticklabels([])
    
    plt.subplot(3, 1, 3)
    plt.xlabel("Distance")
    plt.hist(distD[2],bins=1000,cumulative=False,normed=True,histtype=u'step',color='red',edgecolor='red',alpha=0.7)
    plt.hist(distW[2],bins=1000,cumulative=False,normed=True,histtype=u'step',color='green',edgecolor='green',alpha=0.7)
    plt.hist(distT[2],bins=1000,cumulative=False,normed=True,histtype=u'step',color='blue',edgecolor='blue',alpha=0.7)
    plt.xlim([0,20])
    plt.ylim([0,1])

    plt.savefig("pdfhist2.png")                  
    return
def readFile(myfile):
    global atom,x,y,z
    print 'filename:',myfile        

    with open(myfile) as f:
        line = f.readline()
        a=int(line.split()[0])
        print 'natoms:', a
        line = f.readline()
        atom=['']*a
        x=[0.0]*a
        y=[0.0]*a
        z=[0.0]*a
        print line
        for i in range(a):         
            line = f.readline()
            if not line:
                print "corrupt file at line:", i 
                break
            tmp=line.split()
            atom[i]=tmp[0]
            x[i]=float(tmp[1])
            y[i]=float(tmp[2])
            z[i]=float(tmp[3])
    print "atomtypes:",set(atom)
    print 'xrange:',min(x),max(x)
    print 'yrange:',min(y),max(y)
    print 'zrange:',min(z),max(z)
    return

def getXYZ(myfile):
    print 'filename:',myfile        

    with open(myfile) as f:
        line = f.readline()
        a=int(line.split()[0])
        print 'natoms:', a
        line = f.readline()
        atom=['']*a
        x=[0.0]*a
        y=[0.0]*a
        z=[0.0]*a
        print line
        for i in range(a):         
            line = f.readline()
            if not line:
                print "corrupt file at line:", i 
                break
            tmp=line.split()
            atom[i]=tmp[0]
            x[i]=float(tmp[1])
            y[i]=float(tmp[2])
            z[i]=float(tmp[3])
    print "atomtypes:",set(atom)
    print 'xrange:',min(x),max(x)
    print 'yrange:',min(y),max(y)
    print 'zrange:',min(z),max(z)
    return [atom,x,y,z]

def getdistance(x1,y1,z1,x2,y2,z2):
    import math
    return math.sqrt((x2-x1)**2.0 +(y2-y1)**2.0+(z2-z1)**2.0)

def getdistance2(x1,x2):
    import math
    return math.sqrt((x2-x1)**2.0)
         
def distances():
    natoms=len(atom)
    tmp=[0.0]*((natoms*(natoms+1)/2) - natoms)
    k=0
    for i in range(natoms-1):
        for j in range(i+1,natoms):
            tmp[k]=getdistance(x[i],y[i],z[i],x[j],y[j],z[j])
            if tmp[k]<1.2:
                print i,j
                print x[i],y[i],z[i],x[j],y[j],z[j]
            k=k+1
    print k, len(tmp)        
    print 'min_distance:',min(tmp)#,np.argmin(np.asarray(tmp))
    print 'max_distance:',max(tmp)
    return tmp

def getDistanceMatrix():
    import scipy.sparse as spa
    mydir='/Volumes/s/keceli/Dropbox/work/SEMO/xyz/'
    readFile(mydir+'nanotube2-r_P2.xyz')
    natoms=len(atom)
    mat=np.zeros((natoms,natoms),dtype=np.int)
    for i in range(natoms-1):
        for j in range(i+1,natoms):
            tmp=getdistance(x[i],y[i],z[i],x[j],y[j],z[j])
            if tmp<5.6:
                mat[i][j]=tmp              
    return spa.coo_matrix(mat)


def distances2():
    natoms=len(atom)
    xtmp=[0.0]*((natoms*(natoms+1)/2) - natoms)
    ytmp=[0.0]*((natoms*(natoms+1)/2) - natoms)
    ztmp=[0.0]*((natoms*(natoms+1)/2) - natoms)
    k=0
    for i in range(natoms-1):
        for j in range(i+1,natoms):
            xtmp[k]=getdistance2(x[i],x[j])
            ytmp[k]=getdistance2(y[i],y[j])
            ztmp[k]=getdistance2(z[i],z[j])
            k=k+1

    return [xtmp,ytmp,ztmp]
            
def plotAtoms():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.plot(x,y,z,linestyle='None',marker='.') 
    plt.gca().set_aspect('equal', adjustable='box')  
    plt.show()
    return         
            
def initializeLog(debug):
    import sys
    import logging

    if debug: logLevel = logging.DEBUG
    else: logLevel = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(logLevel)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logLevel)

    logging.debug("Start in debug mode:") 
    return

def getArgs():
    import argparse
    parser = argparse.ArgumentParser(description=
     """
     Jan 19, 2015
     Murat Keceli
     Reads xyz files and prints some info and make some plots. 
     """
     )
    parser.add_argument('input', metavar='FILE', type=str, nargs='?',
        help='xyz file to be parsed. ')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')
    return parser.parse_args()  
          
def main():
    args=getArgs()
    initializeLog(args.debug)
    if args.input is not None:
        xyzFile=args.input
        readFile(xyzFile)
        plotAtoms()
     #   makeDistanceHistograms()
       # plt.hist(distances(),100,range=(0,30),cumulative=True)
       # plt.show()
    else:
        print 'gimme file'
       # makeDistanceHistograms()
       # makeDistanceHistograms2()
    #    makePDFhist()
        makePDFhist2()


if __name__ == "__main__":
    main()