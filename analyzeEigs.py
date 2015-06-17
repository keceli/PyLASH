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
    print 'number of eigenvalues:',len(evec)
    print 'min, max, range of the spectrum:',min(evec),max(evec)
    print 'condition number (abs(max)/abs(min)):',max(abs(evec))/min(abs(evec))
    for i,j in enumerate(evec):
        distvec=np.zeros(len(evec)-1)
        if i>0: distvec[i-1]=j-evec[i-1]
    print 'min and max seperation',':', min(distvec), max(distvec)
    return 0

def plotEigs(evec):
    plt.figure()
    plt.hist(evec,bins=len(evec)/10)
 #   plt.show()
    return 0

def compareEigs(base,evec):
    diffvec=base-evec
    print 'max diff', max(diffvec)
    plt.figure()
    plt.plot(diffvec,ls='None',marker='o',mfc='k',mec='k')
    plt.ylabel('Difference')
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
    parser.add_argument("-f", dest="filename", required=False, help=" input file", metavar="FILE")
    return parser.parse_args()

def main():
    args = getArgs()
    initializeLog(debug=args.debug,warning=args.warning,silent=True)
    if len(args.input) > 0 :
        for i,f in enumerate(args.input):
            logging.debug('Given input argument:{0}'.format(f))
            eigs = np.loadtxt(f, unpack=True)
            analyzeEigs(eigs)
            plotEigs(eigs)
            if i==0:
                base=eigs
            else:
                compareEigs(base,eigs)
        plt.show()
    else:
        logging.debug('No input arguments given')
    return 0

if __name__ == "__main__":
    main()