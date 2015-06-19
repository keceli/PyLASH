#!/usr/bin/env python
import logging
import numpy as np
import matplotlib.pyplot as plt


def getEigenvalues(eigenfile):
    eigs = np.loadtxt( eigenfile, unpack=True)
    print "\nAnalyzing eigenvalues in ", eigenfile
    analyzeEigs(eigs)
    return eigs

def analyzeEigs(evec):
    print 'n: ',len(evec)
    print 'min max: ',min(evec),max(evec)
    print 'condition number max(abs(eigs))/min(abs(eigs)):',max(abs(evec))/min(abs(evec))

    """
    # I don't know why below is not working, there should be no need for absolute value

    for i,j in enumerate(evec):
        distvec=np.zeros(len(evec)-1)
        if i>0: distvec[i-1]=j-evec[i-1]
    print 'min and max seperation',':', min(distvec), max(distvec)
    """
    dist = [abs(evec[j] - evec[j + 1]) for j in range(len(evec) - 1)]
    print 'min  max (delta): ', min(dist), max(dist)
    return 0

def getMultVec(evec,thresh):
    mvec=[]
    m=1
    for i,j in enumerate(evec):
        if i>0 and abs(j-evec[i-1]) < thresh:
            m=m+1
            if i==len(evec)-1:
                mvec.append(m)
        else:
            mvec.append(m)
            m=1
    if sum(mvec) != len(evec): logging.error("multiplicity count error: {0} vs {1}".format(sum(mvec),len(evec)))
    return mvec

def plotEigs(evec,nbins):
    if nbins==100: nbins=len(evec)/10
    plt.figure()
    plt.hist(evec,bins=nbins)
    plt.xlabel("Eigenvalue")
    plt.ylabel("Number of eigenvalues")
    plt.title("Eigenvalue histogram with {0} bins".format(nbins))
 #   plt.show()
    return 0

def plotMult(mvec,thresh):
    plt.figure()
    plt.plot(mvec,ls='None',marker='o',mfc='b',mec='b')
    plt.xlabel("")
    plt.ylabel("Multiplicity")
    plt.title("Multiplicity plot with a threshold {0}".format(thresh))
 #   plt.show()
    return 0

def compareEigs(base,evec, basefile):
    diffvec=base-evec
    print 'max diff', max(diffvec)
    plt.figure()
    plt.plot(diffvec,ls='None',marker='o',mfc='k',mec='k')
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Difference')
    plt.title('Difference based on {0}'.format(basefile))
#    plt.show()
    return 0

def initializeLog(debug=False,warning=False,silent=False):
    import sys
    logging.basicConfig(format='%(levelname)s: %(message)s')
    if debug: logLevel = logging.DEBUG
    elif warning: logLevel = logging.WARNING
    elif silent: logLevel = logging.ERROR
    else: logLevel = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(logLevel)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logLevel)
    logging.debug("Debugging messages will be printed...")
    logging.warning("Warning messages will be printed ...")
    return 0

def getArgs():
    import argparse
    parser = argparse.ArgumentParser(description=
    """
    June 17, 2015
    Murat Keceli
    Analyzes and plots one or many files consisting of only numbers.
    """)
    parser.add_argument('input', metavar='FILE', type=str, nargs='*', help='input arguments')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')
    parser.add_argument('-w', '--warning', action='store_true', help='Print warnings and errors.')
    parser.add_argument('-s', '--silent', action='store_true', help='Print only errors.')
    parser.add_argument("-f", dest="filename", required=False, help="input file", metavar="FILE")
    parser.add_argument("-b", dest="nbins", required=False, type=int, default=100, help="number of bins for the histogram")
    parser.add_argument("-t", dest="thresh", required=False, type=float, default=1.e-6, help="threshold for multiplicity")
    return parser.parse_args()

def main():
    args = getArgs()
    initializeLog(debug=args.debug,warning=args.warning,silent=True)
    if len(args.input) > 0 :
        for i,f in enumerate(args.input):
            logging.debug('Given input argument:{0}'.format(f))
            print "file: ",f
            eigs = np.loadtxt(f, unpack=True)
            analyzeEigs(eigs)
            mvec=getMultVec(eigs, args.thresh)
            print 'max multiplicity:', max(mvec)
            plotEigs(eigs,args.nbins)
            plotMult(mvec,args.thresh)
            if i==0:
                base=eigs
            else:
                compareEigs(base,eigs,args.input[0])
        plt.show()
    else:
        logging.debug('No input arguments given')
    return 0

if __name__ == "__main__":
    main()