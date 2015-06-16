#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
import itertools
from math import ceil
import math
import scipy.optimize as opt
import sys

# font = {'family' : 'times',
#        'weight' : 'normal',
#        'size'   : 16}
font = {'weight' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)
legendsize = 12
ms = 8
# matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['mathtext.default'] = 'regular'

# Define function for string formatting of scientific notation. from http://stackoverflow.com/questions/18311909/how-do-i-annotate-with-power-of-ten-formatting

def getSciNotation(num, decimal_digits=2, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    http://stackoverflow.com/questions/18311909/how-do-i-annotate-with-power-of-ten-formatting
    """
    if not exponent:
        exponent = int(math.floor(math.log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if not precision:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)

def getPowerFitLabel(a,b):
    """
    Returns a fit label in the form
    $y=a \cdot x^b$
    a is written with a power of 10 instead of E notation
    """
    return r'$y=$'+getSciNotation(a)+r'$x^{{{0:.2f}}}$'.format(b)

def getCubicFunction(x,a,b,c):
    return a*x**3.0 + b*x**2.0 + c*x

def getCubicFit(x,y):
    """
    Returns (a,b,c) of a cubic fit with 0 intercept
    $ ax^3 + bx^2 + cx $
    """
    popt, pcov = opt.curve_fit(getCubicFunction, x, y)
    return popt

def getExtendedArray(x,maxValue):
    if len(x)<2:
        logging.error('x should be an array of length > 1')
        return x
    factor=x[-1]/x[-2]
    logging.debug('factor: {}'.format(factor))
    if factor<1.0:
        logging.error('values of x should be increasing with index')
        return x
    y=x
    while y[-1]<maxValue:
        y = np.append(y,y[-1]*factor)
    return y

def getAmdahlLaw(t,ts,n):
    logging.debug('\n getAmdahlLaw...')
    if (ts>t): logging.error('ts > t')
    s=ts/t
    p=1.0-s
    logging.info('Serial portion:{0:.3f}'.format(s))
    logging.info('Max speedup:{0:.2f}'.format(1.0/s))
    return t*(s+p/n.astype(float))

def getPowerLaw(x,a,b):
    return a*x**b

def getPowerFit(x, y):
    from scipy import optimize, log10
    logx = log10(x)
    logy = log10(y)

    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y: (y - fitfunc(p, x))

    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit,
                           args=(logx, logy), full_output=1)

    pfinal = out[0]
    covar = out[1]

    index = pfinal[1]
    amp = 10.0 ** pfinal[0]

    print '\n Power fit output:'
    print 'amp:', amp, 'index', index, 'covar', covar

    return [amp * (x ** index), amp, index]

def plotSIPsNNnzScalingFigure(embed=0):
    """
    Plot  y axis SIPs time to solution (tall)
    x axis is $ N \times N_nz $
    """
    # strong scaling plots
    font = {'weight' : 'normal',
        'size'   : 14}
    matplotlib.rc('font', **font)
    legendsize = 14
    ms = 8
    TmatrixSize, TnCores, Tratios, TsolveTime, TtotalTime, Tncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_T.txt', unpack=True, usecols=(1, 3, 6, 9, 9, 10))
    WmatrixSize, WnCores, Wratios, WsolveTime, WtotalTime, Wncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_W.txt', unpack=True, usecols=(1, 3, 6, 9, 9, 10))
    DmatrixSize, DnCores, Dratios, DsolveTime, DtotalTime, Dncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_D.txt', unpack=True, usecols=(1, 3, 6, 9, 9, 10))

    matrixSize,Tnnz,Wnnz,Dnnz = np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_fillin.txt", unpack=True, usecols=(0,4,5,6))

    matlistD = [8000, 16000, 32000, 64000]
    matlistW = [8000, 16000, 32000, 64000, 128000]
    matlistT = [8000, 16000, 32000,64000, 128000,256000, 512000]

    xT=np.zeros(len(matlistT))
    xT2=np.zeros(len(matlistT))
    yT=np.zeros(len(matlistT))

    for i in range(len(matlistT)):
        yT[i] = min(TsolveTime[np.logical_and(TnCores == 16384,TmatrixSize == matlistT[i])])
        xT[i] = matlistT[i]*Tnnz[matrixSize == matlistT[i]]
        xT2[i] = matlistT[i]

    xW=np.zeros(len(matlistW))
    xW2=np.zeros(len(matlistW))
    yW=np.zeros(len(matlistW))
    for i in range(len(matlistW)):
        yW[i] = min(WsolveTime[np.logical_and(WnCores == 16384,WmatrixSize == matlistW[i])])
        xW[i] = matlistW[i]*Wnnz[matrixSize == matlistW[i]]
        xW2[i] = matlistW[i]

    xD=np.zeros(len(matlistD))
    xD2=np.zeros(len(matlistD))
    yD=np.zeros(len(matlistD))
    for i in range(len(matlistD)):
        yD[i] = min(DsolveTime[np.logical_and(DnCores == 16384,DmatrixSize == matlistD[i])])
        xD[i] = matlistD[i]*Dnnz[matrixSize == matlistD[i]]
        xD2[i] = matlistD[i]

    xTW=np.concatenate([xT,xW])
    xTW2=np.concatenate([xT2,xW2])
    yTW=np.concatenate([yT,yW])

    if not embed: plt.figure()

    xlabel = r"$N \cdot N_{nz}$"
    ylabel = r"Time to solution, $t$ (s)"
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='major', axis='y')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(1e10, 1.8e14)
    plt.ylim(1, 2e3)

    p1,= plt.plot(xT, yT, linestyle='None', label="nanotube", marker='o', markersize=ms, mfc='b', mec='b')
    p2,= plt.plot(xW, yW, linestyle='None', label='nanowire', marker='s', markersize=ms, mfc='g', mec='g')
    p3,= plt.plot(xD, yD, linestyle='None', label='diamond', marker='d', markersize=ms, mfc='r', mec='r')

    plegend = plt.legend(handles=[p1,p2,p3], loc='upper left', prop={'size':legendsize})
    plt.gca().add_artist(plegend)

    myfit = getPowerFit(xD, yD)
    mylabel=r'$t=$'+getSciNotation(myfit[1])+r'$(N\cdot N_{{nz}})^{{{0:.2f}}}$'.format(myfit[2])
    f1,=plt.plot(xD, myfit[0], label=mylabel, linestyle='--', color='red')

    myfit = getPowerFit(xTW, yTW)
    mylabel=r'$t=$'+getSciNotation(myfit[1])+r'$(N\cdot N_{{nz}})^{{{0:.2f}}}$'.format(myfit[2])
    f2,=plt.plot(xTW, myfit[0], label=mylabel, linestyle='-', color='blue')

    plt.legend(handles=[f1,f2],loc='lower right', prop={'size':legendsize})

    if not embed: plt.show() #plt.savefig("SIPs_NNz_scaling.png")

    return

def plotSIPsElementalScaling(embed=0):
    # strong scaling plots
    font = {'weight' : 'normal',
            'size'   : 12}
    matplotlib.rc('font', **font)
    legendsize = 11
    ms = 6
    if not embed:
        font = {'weight' : 'normal',
            'size'   : 14}
        matplotlib.rc('font', **font)
        legendsize = 14
        ms = 6
    TmatrixSize, TnCores, Tratios, TsolveTime, TtotalTime, Tncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_T.txt', unpack=True, usecols=(1, 3, 6, 9, 9, 10))
    WmatrixSize, WnCores, Wratios, WsolveTime, WtotalTime, Wncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_W.txt', unpack=True, usecols=(1, 3, 6, 9, 9, 10))
    DmatrixSize, DnCores, Dratios, DsolveTime, DtotalTime, Dncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_D.txt', unpack=True, usecols=(1, 3, 6, 9, 9, 10))
    EmatrixSize, ETtotalTime,EWtotalTime,EDtotalTime= np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_El.txt', unpack=True, usecols=(0,1,2,3))
    matrixSize,Tnnz,Wnnz,Dnnz = np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_fillin.txt", unpack=True, usecols=(0,4,5,6))

    matlistD = [8000, 16000, 32000, 64000]
    matlistW = [8000, 16000, 32000, 64000, 128000]
    matlistT = [8000, 16000, 32000,64000, 128000,256000, 512000]
    x=np.arange(8000,5000000,1000)

    xT=np.zeros(len(matlistT))
    xT2=np.zeros(len(matlistT))
    yT=np.zeros(len(matlistT))
    if not embed: plt.figure()

    for i in range(len(matlistT)):
        yT[i] = min(TsolveTime[np.logical_and(TnCores == 16384,TmatrixSize == matlistT[i])])
        xT[i] = matlistT[i]*Tnnz[matrixSize == matlistT[i]]
        xT2[i] = matlistT[i]

    xW=np.zeros(len(matlistW))
    xW2=np.zeros(len(matlistW))
    yW=np.zeros(len(matlistW))
    for i in range(len(matlistW)):
        yW[i] = min(WsolveTime[np.logical_and(WnCores == 16384,WmatrixSize == matlistW[i])])
        xW[i] = matlistW[i]*Wnnz[matrixSize == matlistW[i]]
        xW2[i] = matlistW[i]

    xD=np.zeros(len(matlistD))
    xD2=np.zeros(len(matlistD))
    yD=np.zeros(len(matlistD))
    for i in range(len(matlistD)):
        yD[i] = min(DsolveTime[np.logical_and(DnCores == 16384,DmatrixSize == matlistD[i])])
        xD[i] = matlistD[i]*Dnnz[matrixSize == matlistD[i]]
        xD2[i] = matlistD[i]

    xTW=np.concatenate([xT,xW])
    xTW2=np.concatenate([xT2,xW2])
    yTW=np.concatenate([yT,yW])
    if embed:
        xlabel = r"$N$"
    else:
        xlabel = r"Number of basis functions, $N$"

    ylabel = r"Time to solution, $t$ (s)"

    plt.xscale('log')
    plt.yscale('log')
  #  plt.grid(which='major', axis='y')
    plt.xlabel(xlabel)
    if not embed:
        plt.ylabel(ylabel)
    plt.xlim(5e3, 1.1e6)
    plt.ylim(1, 1.1e4)

#    plotNNnz=1
#     if plotNNnz:
#         xT2=xT
#         xW2=xW
#         xD2=xD
#         xTW2=xTW
#         EmatrixSize= [EmatrixSize[i] * (EmatrixSize[i] +1) / 2 for i in range(len(EmatrixSize))]
#         x=np.asarray([x[i] * (x[i] +1) / 2 for i in range(len(x))])

    p1,=plt.plot(xT2, yT, linestyle='None', label="nanotube", marker='o', markersize=ms, mfc='b', mec='b')
    p2,=plt.plot(xW2, yW, linestyle='None', label='nanowire', marker='s', markersize=ms, mfc='g', mec='g')
    p3,=plt.plot(xD2, yD, linestyle='None', label='diamond', marker='d', markersize=ms, mfc='r', mec='r')
    p4,=plt.plot(EmatrixSize, ETtotalTime, linestyle='None', label="nanotube", marker='o', markersize=ms+2, mfc='none', mec='b')
    p5,=plt.plot(EmatrixSize, EWtotalTime, linestyle='None', label='nanowire', marker='s', markersize=ms+2, mfc='none', mec='g')
    p6,=plt.plot(EmatrixSize, EDtotalTime, linestyle='None', label='diamond', marker='d', markersize=ms+2, mfc='none', mec='r')
    #p4,=plt.plot(EmatrixSize, EtotalTime, linestyle='None', label='Elemental', marker='*', markersize=ms+2, mfc='k', mec='k')





#     myfit = getPowerFit(EmatrixSize, EtotalTime)
#     mylabel=r'$t_{El}=$'+getSciNotation(myfit[1])+r'$N^{{{0:.2f}}}$'.format(myfit[2])
#     f2,=plt.plot(x, getPowerLaw(x,myfit[1],myfit[2]), label=mylabel, linestyle='--', color='black')

#     myfit = getPowerFit(xW2, yW)
#     mylabel=r'$t=$'+getSciNotation(myfit[1])+r'$N^{{{0:.2f}}}$'.format(myfit[2])
#     f2,=plt.plot(x, getPowerLaw(x,myfit[1],myfit[2]), label=mylabel, linestyle='--', color='green')
#
#     myfit = getPowerFit(xT2, yT)
#     mylabel=r'$t=$'+getSciNotation(myfit[1])+r'$N^{{{0:.2f}}}$'.format(myfit[2])
#     f3,=plt.plot(x, getPowerLaw(x,myfit[1],myfit[2]), label=mylabel, linestyle='--', color='blue')



#    z = np.polyfit(EmatrixSize, EtotalTime, 3)
#    myfit = np.poly1d(z)

#     z = getCubicFit(xT2, yT)
#     mylabel=r'$t_{El}=$'+getSciNotation(z[0]) + r'$N^3+$' + getSciNotation(z[1]) + r'$N^2+$' + getSciNotation(z[2]) + r'$N$' #+ r'${{{0:.2f}}}$'.format(z[3])
#     f6,=plt.plot(x, getCubicFunction(x, *z), label=mylabel, linestyle='-', color='blue')
#
#     z = getCubicFit(xW2, yW)
#     mylabel=r'$t_{El}=$'+getSciNotation(z[0]) + r'$N^3+$' + getSciNotation(z[1]) + r'$N^2+$' + getSciNotation(z[2]) + r'$N$' #+ r'${{{0:.2f}}}$'.format(z[3])
#     f7,=plt.plot(x, getCubicFunction(x, *z), label=mylabel, linestyle='-', color='green')

#     z = getCubicFit(xD2, yD)
#     mylabel=r'$t_{SIPs}=$'+getSciNotation(z[0]) + r'$N^3+$' + getSciNotation(z[1]) + r'$N^2+$' + getSciNotation(z[2]) + r'$N$' #+ r'${{{0:.2f}}}$'.format(z[3])
#     f8,=plt.plot(x, getCubicFunction(x, *z), label=mylabel, linestyle='-', color='red')
    if embed:
       # plegend = plt.legend(handles=[f4,f1], loc='lower right',prop={'size':legendsize})
        #plt.gca().add_artist(plegend)
        plt.legend(handles=[f1,f4,p4],loc='upper left', prop={'size':legendsize},frameon=False)
    else:
        plegend = plt.legend(handles=[p1,p2,p3,p4,p5,p6], loc='lower right',prop={'size':legendsize},ncol=2, title="SIPs                 Elemental")
        plt.gca().add_artist(plegend)
        plt.savefig("SIPs_El_scaling1.png")
        myfit = getPowerFit(xD2, yD)
        mylabel=r'$t_{SIPs}=$'+getSciNotation(myfit[1])+r'$N^{{{0:.2f}}}$'.format(myfit[2])
        f1,=plt.plot(x, getPowerLaw(x,myfit[1],myfit[2]), label=mylabel, linestyle='--', color='red')
        myfit = getPowerFit(xTW2, yTW)
        mylabel=r'$t_{SIPs}=$'+getSciNotation(myfit[1])+r'$N^{{{0:.2f}}}$'.format(myfit[2])
        f4,=plt.plot(x, getPowerLaw(x,myfit[1],myfit[2]), label=mylabel, linestyle='--', color='blue')
      #  z = getCubicFit(EmatrixSize, EtotalTime)
       # mylabel=r'$t_{El}=$'+getSciNotation(z[0]) + r'$N^3+$' + getSciNotation(z[1]) + r'$N^2+$' + getSciNotation(z[2]) + r'$N$' #+ r'${{{0:.2f}}}$'.format(z[3])

      #  f5,=plt.plot(x, getCubicFunction(x, *z), label=mylabel, linestyle='-', color='black')
        plt.legend(handles=[f1,f4],loc='upper left', prop={'size':legendsize},frameon=False)
 #   plt.gca().get_yaxis().set_ticklabels([])
 #   fig.tight_layout()
    if not embed : plt.savefig("SIPs_El_scaling2.png")


    # plt.show()
    return

def binning(M, blockSize):
    size = M.shape[0]
    binNbr = int(ceil(size / blockSize))
    matSize = M.shape[0]
    blockSize = ceil(matSize / binNbr)
    bins = np.zeros((binNbr, binNbr), dtype=int)
    for i, j in itertools.izip(M.row, M.col):
        bins[(i // blockSize, j // blockSize)] += 1
    return bins

def doScalingFigure():
    # strong and weak scaling plots
    font = {'weight' : 'normal',
        'size'   : 20}
    matplotlib.rc('font', **font)
    TmatrixSize, TnCores, Tratios, TsetupTime,TsolveTime, TtotalTime, Tncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_T.txt', unpack=True, usecols=(1, 3, 6, 7, 8, 9, 10))
    WmatrixSize, WnCores, Wratios, WsetupTime,WsolveTime, WtotalTime, Wncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_W.txt', unpack=True, usecols=(1, 3, 6, 7, 8, 9, 10))
    DmatrixSize, DnCores, Dratios, DsetupTime,DsolveTime, DtotalTime, Dncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_D.txt', unpack=True, usecols=(1, 3, 6, 7, 8, 9, 10))

    ratiosT = [2000, 1000, 500, 250, 125, 62.5]  # ,192]
    ratiosW = [2000, 1000, 500, 250, 125, 62.5]  # ,192]
    ratiosD = [2000, 1000, 500, 250, 125, 62.5]  # ,192]
    ratiosT = [2000, 1000, 500, 250]  # ,192]
    ratiosW = [2000, 1000, 500, 250]  # ,192]
    ratiosD = [2000, 1000, 500, 250]  # ,192]
    xticklabel = [ 8000,32000, 128000,512000]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6.5))
    plt.subplot(1, 3, 1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(5000, 1e6)
    plt.ylim(10, 1e4)
    plt.grid(which='major', axis='y')
    ylabel = "Time to solution (s)"
    plt.ylabel(ylabel)
    mymarker = itertools.cycle(list('o^sd*h>v<*'))
    mycolorRGB = itertools.cycle(list('rgbk'))

    legendsize = 15
    ms = 8
    for i in range(len(ratiosT)):
        x = TmatrixSize[np.logical_and(Tratios == ratiosT[i], Tncoresperslice == 16)]
        y = TtotalTime[np.logical_and(Tratios == ratiosT[i], Tncoresperslice == 16)]
        x2 = TmatrixSize[np.logical_and(Tratios == ratiosT[i], Tncoresperslice == 32)]
        y2 = TtotalTime[np.logical_and(Tratios == ratiosT[i], Tncoresperslice == 32)]
        myLabel = "ratio = " + str(ratiosT[i])
        currentmarker = next(mymarker)
        currentcolor = next(mycolorRGB)
        plt.plot(x, y, linestyle='None', label=myLabel, marker=currentmarker, markersize=ms, mfc=currentcolor, mec=currentcolor)
        plt.plot(x2, y2, linestyle='None', marker=currentmarker, markersize=ms, mfc='none', mec=currentcolor)
    plt.legend(loc='upper right', prop={'size':legendsize})
    plt.tick_params(\
    axis='x',          # changes apply to the x-axis
    which='minor',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
    plt.xticks(xticklabel, map(lambda i : "%d" % i, xticklabel))


    plt.subplot(1, 3, 2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(5000, 1e6)
    plt.ylim(10, 1e4)
    plt.grid(which='major', axis='y')
    xlabel = "Number of basis functions"
    plt.xlabel(xlabel)
    mymarker = itertools.cycle(list('o^sd*h>v<*'))
    for i in range(len(ratiosW)):
        x = WmatrixSize[np.logical_and(Wratios == ratiosW[i], Wncoresperslice == 16)]
        y = WtotalTime[np.logical_and(Wratios == ratiosW[i], Wncoresperslice == 16)]
        x2 = WmatrixSize[np.logical_and(Wratios == ratiosW[i], Wncoresperslice == 32)]
        y2 = WtotalTime[np.logical_and(Wratios == ratiosW[i], Wncoresperslice == 32)]
        myLabel = "ratio = " + str(ratiosW[i])
        currentmarker = next(mymarker)
        currentcolor = next(mycolorRGB)
        plt.plot(x, y, linestyle='None', label=myLabel, marker=currentmarker, markersize=ms, mfc=currentcolor, mec=currentcolor)
        plt.plot(x2, y2, linestyle='None', marker=currentmarker, markersize=ms, mfc='none', mec=currentcolor)
    plt.legend(loc='upper right', prop={'size':legendsize})
    plt.xlim(5000, 1e6)
    plt.ylim(10, 1e4)
    plt.gca().get_yaxis().set_ticklabels([])
    plt.tick_params(\
    axis='x',          # changes apply to the x-axis
    which='minor',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
    plt.xticks(xticklabel, map(lambda i : "%d" % i, xticklabel))

    plt.subplot(1, 3, 3)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='major', axis='y')
    mymarker = itertools.cycle(list('o^sd*h>v<*'))
    for i in range(len(ratiosD)):
        x = DmatrixSize[np.logical_and(Dratios == ratiosD[i], Dncoresperslice == 16)]
        y = DtotalTime[np.logical_and(Dratios == ratiosD[i], Dncoresperslice == 16)]
        x2 = DmatrixSize[np.logical_and(Dratios == ratiosD[i], Dncoresperslice > 16)]
        y2 = DtotalTime[np.logical_and(Dratios == ratiosD[i], Dncoresperslice > 16)]
        myLabel = "ratio = " + str(ratiosD[i])
        currentmarker = next(mymarker)
        currentcolor = next(mycolorRGB)
        plt.plot(x, y, linestyle='None', label=myLabel, marker=currentmarker, markersize=ms, mfc=currentcolor, mec=currentcolor)
        plt.plot(x2, y2, linestyle='None', marker=currentmarker, markersize=ms, mfc='none', mec=currentcolor)

    plt.legend(loc='upper right', prop={'size':legendsize})
    plt.xlim(5000, 1e6)
    plt.ylim(10, 1e4)
    plt.gca().get_yaxis().set_ticklabels([])
    plt.tick_params(\
    axis='x',          # changes apply to the x-axis
    which='minor',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
    plt.xticks(xticklabel, map(lambda i : "%d" % i, xticklabel))

    fig.tight_layout()
    plt.savefig("SIPs_weak.png")

    # strong scaling
    # strong scaling
    # strong scaling
#     font = {'weight' : 'normal',
#         'size'   : 16}
#     matplotlib.rc('font', **font)
    matlistD = [8000, 16000, 32000, 64000]
    matlistW = [8000, 32000, 128000]
    matlistT = [8000, 64000, 512000]
    xticklabel=[16,64,256, 1024,4096,16384,65536 ,266144]
    myxlim=[12,4e6]
    nAmdahl=3
    xlabel = "Number of cores"
    ylabel = "Time to solution (s)"

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6.5))
  #  fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(9.5,3.5))

    plt.subplot(1, 3, 1)
    plt.xscale('log',basex=2)
    plt.yscale('log')
    plt.grid(which='major', axis='y')
    plt.ylabel(ylabel)
    plt.xlim(myxlim)
    plt.ylim(1, 1e4)
    mymarker = itertools.cycle(list('o^sd*h>v<*'))
    mycolorRGB = itertools.cycle(list('rgbk'))
    for i in range(len(matlistT)):
        myLabel = r'$N=$' + str(matlistT[i])
        currentmarker = next(mymarker)
        currentcolor = next(mycolorRGB)
        x = TnCores[np.logical_and(TmatrixSize == matlistT[i], Tncoresperslice == 16)]
        y = TtotalTime[np.logical_and(TmatrixSize == matlistT[i], Tncoresperslice == 16)]
        if (i<nAmdahl):
            tsVec=TsetupTime[np.logical_and(TmatrixSize == matlistT[i], Tncoresperslice == 16)]
            ts=sum(tsVec)/len(tsVec)
            xAmdahl=getExtendedArray(x, 1e7)
            yAmdahl=getAmdahlLaw(y[0], ts, xAmdahl/float(x[0]))
            if (nAmdahl==1):
                plt.plot(xAmdahl, yAmdahl, linestyle='--', label="Amdahl's law", marker='.', markersize=ms, mfc=currentcolor, mec=currentcolor,color=currentcolor)
            else:
                plt.plot(xAmdahl, yAmdahl, linestyle='--', marker='.', markersize=ms, mfc=currentcolor, mec=currentcolor,color=currentcolor)

        x1 = TnCores[(TmatrixSize == matlistT[i])]
        y1 = TtotalTime[(TmatrixSize == matlistT[i])]
        x2 = TnCores[np.logical_and(TmatrixSize == matlistT[i], Tncoresperslice == 32)]
        y2 = TtotalTime[np.logical_and(TmatrixSize == matlistT[i], Tncoresperslice == 32)]


        plt.plot(x, y, linestyle='None', label=myLabel, marker=currentmarker, markersize=ms, mfc=currentcolor, mec=currentcolor)
        plt.plot(x2, y2, linestyle='None', marker=currentmarker, markersize=ms, mfc='none', mec=currentcolor)
        plt.plot(sorted(x1), [sorted(y1, reverse=True)[0] / (sorted(x1)[i] / sorted(x1)[0]) for i in range(len(x1))], linestyle='-', color=currentcolor)

    plt.legend(loc='upper right', prop={'size':legendsize})
   # plt.xticks(xticklabel, map(lambda i : "%d" % i, xticklabel))

    plt.subplot(1, 3, 2)
    plt.xscale('log',basex=2)
    plt.yscale('log')
    plt.xlim(myxlim)
    plt.ylim(1, 1e4)
    plt.grid(which='major', axis='y')
    mymarker = itertools.cycle(list('o^sd*h>v<*'))
    mycolorRGB = itertools.cycle(list('rgbk'))
    for i in range(len(matlistW)):
        x = WnCores[np.logical_and(WmatrixSize == matlistW[i], Wncoresperslice == 16)]
        y = WtotalTime[np.logical_and(WmatrixSize == matlistW[i], Wncoresperslice == 16)]
        x1 = WnCores[(WmatrixSize == matlistW[i])]
        y1 = WtotalTime[(WmatrixSize == matlistW[i])]
        x2 = WnCores[np.logical_and(WmatrixSize == matlistW[i], Wncoresperslice == 32)]
        y2 = WtotalTime[np.logical_and(WmatrixSize == matlistW[i], Wncoresperslice == 32)]
        myLabel = r'$N=$' + str(matlistW[i])
        currentmarker = next(mymarker)
        currentcolor = next(mycolorRGB)
        if (i<nAmdahl):
            tsVec=WsetupTime[np.logical_and(WmatrixSize == matlistW[i], Wncoresperslice == 16)]
            ts=sum(tsVec)/len(tsVec)
            xAmdahl=getExtendedArray(x, 1e7)
            yAmdahl=getAmdahlLaw(y[0], ts, xAmdahl/float(x[0]))
            if (nAmdahl==1):
                plt.plot(xAmdahl, yAmdahl, linestyle='--', label="Amdahl's law", marker='.', markersize=ms, mfc=currentcolor, mec=currentcolor,color=currentcolor)
            else:
                plt.plot(xAmdahl, yAmdahl, linestyle='--', marker='.', markersize=ms, mfc=currentcolor, mec=currentcolor,color=currentcolor)

        plt.plot(x, y, linestyle='None', label=myLabel, marker=currentmarker, markersize=ms, mfc=currentcolor, mec=currentcolor)
        plt.plot(x2, y2, linestyle='None', marker=currentmarker, markersize=ms, mfc='none', mec=currentcolor)
        plt.plot(sorted(x1), [sorted(y1, reverse=True)[0] / (sorted(x1)[i] / sorted(x1)[0]) for i in range(len(x1))], linestyle='-', color=currentcolor)
    plt.gca().get_yaxis().set_ticklabels([])
    plt.legend(loc='upper right', prop={'size':legendsize})
    plt.xlabel(xlabel)
   # plt.xticks(xticklabel, map(lambda i : "%d" % i, xticklabel))

    plt.subplot(1, 3, 3)
    plt.xscale('log',basex=2)
    plt.yscale('log')
    plt.xlim(myxlim)
    plt.ylim(1, 1e4)
    plt.grid(which='major', axis='y')
    mymarker = itertools.cycle(list('o^sd*h>v<*'))
    mycolorRGB = itertools.cycle(list('rgbk'))
    for i in range(len(matlistD)):
        x = DnCores[np.logical_and(DmatrixSize == matlistD[i], Dncoresperslice == 16)]
        y = DtotalTime[np.logical_and(DmatrixSize == matlistD[i], Dncoresperslice == 16)]
        x1 = DnCores[(DmatrixSize == matlistD[i])]
        y1 = DtotalTime[(DmatrixSize == matlistD[i])]
        x2 = DnCores[np.logical_and(DmatrixSize == matlistD[i], Dncoresperslice > 16)]
        y2 = DtotalTime[np.logical_and(DmatrixSize == matlistD[i], Dncoresperslice > 16)]
        myLabel = r'$N=$' + str(matlistD[i])
        currentmarker = next(mymarker)
        currentcolor = next(mycolorRGB)
        if (i<nAmdahl):
            tsVec=DsetupTime[np.logical_and(DmatrixSize == matlistD[i], Dncoresperslice == 16)]
            ts=sum(tsVec)/len(tsVec)
            xAmdahl=getExtendedArray(x, 1e7)
            yAmdahl=getAmdahlLaw(y[0], ts, xAmdahl/float(x[0]))
            if (nAmdahl==1):
                plt.plot(xAmdahl, yAmdahl, linestyle='--', label="Amdahl's law", marker='.', markersize=ms, mfc=currentcolor, mec=currentcolor,color=currentcolor)
            else:
                plt.plot(xAmdahl, yAmdahl, linestyle='--', marker='.', markersize=ms, mfc=currentcolor, mec=currentcolor,color=currentcolor)
        if i==3:
            print matlistD[i]
            tsVec=DsetupTime[np.logical_and(DmatrixSize == matlistD[i], Dncoresperslice == 64)]
            ts=sum(tsVec)/len(tsVec)
            xAmdahl=getExtendedArray(x2, 1e7)
            yAmdahl=getAmdahlLaw(y2[0], ts, xAmdahl/float(x2[0]))
            plt.plot(xAmdahl, yAmdahl, linestyle='--', marker='.', markersize=ms, mfc=currentcolor, mec=currentcolor,color=currentcolor)

        plt.plot(x, y, linestyle='None', label=myLabel, marker=currentmarker, markersize=ms, mfc=currentcolor, mec=currentcolor)
        plt.plot(x2, y2, linestyle='None', marker=currentmarker, markersize=ms, mfc='none', mec=currentcolor)
        plt.plot(sorted(x1), [sorted(y1, reverse=True)[0] / (sorted(x1)[i] / sorted(x1)[0]) for i in range(len(x1))], linestyle='-', color=currentcolor)
    plt.gca().get_yaxis().set_ticklabels([])
    plt.legend(loc='lower left', prop={'size':legendsize})
  #  plt.xticks(xticklabel, map(lambda i : "%d" % i, xticklabel))

    fig.tight_layout()
    plt.savefig("SIPs_strong.png")

    # plt.show()
    return

def doTOCFigure():
    """
    Generate SIPs strong scaling plot for CNT512000, DNW64000, BDC8000 examples
    """
    font = {'weight' : 'normal',
        'size'   : 8,
        'family' : 'sans-serif'}
    matplotlib.rc('font', **font)
    mymarker = itertools.cycle(list('o^sd*h>v<*'))
    mycolorRGB = itertools.cycle(list('rgbk'))

    TmatrixSize, TnCores, Tratios, TsetupTime,TsolveTime, TtotalTime, Tncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_T2.txt', unpack=True, usecols=(1, 3, 6, 7, 8, 9, 10))
    WmatrixSize, WnCores, Wratios, WsetupTime,WsolveTime, WtotalTime, Wncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_W2.txt', unpack=True, usecols=(1, 3, 6, 7, 8, 9, 10))
    DmatrixSize, DnCores, Dratios, DsetupTime,DsolveTime, DtotalTime, Dncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_D.txt', unpack=True, usecols=(1, 3, 6, 7, 8, 9, 10))


    matlistD = [8000]
    matlistW = [64000]
    matlistT = [512000]
    xticklabel=[16,64,256, 1024,4096,16384,65536 ,266144]
    nAmdahl=3
    xlabel = "Number of cores"
    ylabel = "Time (s)"
    ms=4
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3, 1.75))
  #  fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(9.5,3.5))

    plt.subplot(1, 1, 1)
    plt.xscale('log',basex=10)
    plt.yscale('log')
   # plt.grid(which='major', axis='y')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([9.5,1.5e6])
    plt.ylim(10, 1e4)

    for i in range(len(matlistT)):
        myLabel = r'$N=$' + str(matlistT[i])
        x = TnCores[np.logical_and(TmatrixSize == matlistT[i], Tncoresperslice > 15)]
        y = TtotalTime[np.logical_and(TmatrixSize == matlistT[i], Tncoresperslice > 15)]
        plt.plot(x, y, linestyle='None', label=myLabel, marker='o', markersize=ms, mfc='b', mec='b')

    for i in range(len(matlistW)):
        x = WnCores[np.logical_and(WmatrixSize == matlistW[i], Wncoresperslice >15)]
        y = WtotalTime[np.logical_and(WmatrixSize == matlistW[i], Wncoresperslice > 15)]
        plt.plot(x, y, linestyle='None', label=myLabel, marker='s', markersize=ms, mfc='g', mec='g')

    for i in range(len(matlistD)):
        x = DnCores[np.logical_and(DmatrixSize == matlistD[i], Dncoresperslice == 16)]
        y = DtotalTime[np.logical_and(DmatrixSize == matlistD[i], Dncoresperslice == 16)]
        plt.plot(x, y, linestyle='None', label=myLabel, marker='d', markersize=ms, mfc='r', mec='r')

    fig.tight_layout()
    plt.savefig("SIPs_TOC.pdf")

    # plt.show()
    return

def doFactorizationStronScalingFigure():
    datafile='data_max_factorization.txt'
    print '\n\n Generating factorization scaling figure using data file:',datafile

    # strong scaling plots
    font = {'weight' : 'normal',
        'size'   : 12}
    matplotlib.rc('font', **font)
    legendsize = 8
    ms = 5
    nCores,matSize,symTime, numTime,fType = np.genfromtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/'+datafile, unpack=True,usecols=(1, 2,3,4,5))
   #  fName = np.genfromtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_max_factorization.txt', unpack=True, usecols=(0),dtype=None)


    xlabel = "Number of cores"
    ylabel = "Walltime (s)"


    fig, _ = plt.subplots(nrows=1, ncols=3, figsize=(9.5, 3.5))

    mymarker = itertools.cycle(list('osd*h>v<*'))
    mycolor = itertools.cycle(list('bgrgbk'))
  #  fList=['tube', 'wire', 'diamond']
    mList=[512000,128000,64000]
    for i in range(3):
        plt.subplot(1, 3, i+1)
    #    xMin = nCores[np.logical_and(matSize==8000,fList[i] in fName)] # gave false for fList in condition
        xMin = nCores[np.logical_and(matSize==8000,fType==i)]
        yMinSym = symTime[np.logical_and(matSize==8000,fType==i)]
        yMinNum = numTime[np.logical_and(matSize==8000,fType==i)]
        yMinTot = yMinSym + 2.0 * yMinNum
        xMax=nCores[np.logical_and(matSize==mList[i],fType==i)]
        yMaxSym = symTime[np.logical_and(matSize==mList[i],fType==i)]
        yMaxNum = numTime[np.logical_and(matSize==mList[i],fType==i)]
        yMaxTot = yMaxSym + 2.0 * yMaxNum
      #  currentmarker = mymarker.next()
        plt.xscale('log',basex=2)
        plt.yscale('log')
        plt.grid(which='major', axis='y')
        if (i==1): plt.xlabel(xlabel)
        if (i==0): plt.ylabel(ylabel)
        plt.xlim(10, 700)
        plt.ylim(0.5, 1500)
        plt.plot(xMin, yMinSym, linestyle='None', label='sym. 8000', marker='o', markersize=ms, mfc='blue', mec='blue')
        plt.plot(xMin, yMinNum, linestyle='None', label='num. 8000', marker='s', markersize=ms, mfc='blue', mec='blue')
        plt.plot(xMin, yMinTot, linestyle='None', label='tot. 8000', marker='d', markersize=ms, mfc='blue', mec='blue')
        plt.plot(xMax, yMaxSym, linestyle='None', label='sym. '+str(mList[i]), marker='o', markersize=ms, mfc='red', mec='red')
        plt.plot(xMax, yMaxNum, linestyle='None', label='num. '+str(mList[i]), marker='s', markersize=ms, mfc='red', mec='red')
        plt.plot(xMax, yMaxTot, linestyle='None', label='tot. '+str(mList[i]), marker='d', markersize=ms, mfc='red', mec='red')
        plt.legend(loc='upper left', prop={'size':legendsize},ncol=2)
        xticklabel = [ 16,32, 64,128, 256, 512]
        plt.xticks(xticklabel, map(lambda i : "%d" % i, xticklabel))


    fig.tight_layout()
    plt.savefig("SIPs_factorization_strong_scaling.png")

    # plt.show()
    return

def doFactorizationStronScalingFigure2():
    datafile='data_max_factorization.txt'
    print '\n\n Generating factorization scaling figure using data file:',datafile

    # strong scaling plots
    font = {'weight' : 'normal',
        'size'   : 12}
    matplotlib.rc('font', **font)
    legendsize = 11
    ms = 6
    nCores,matSize,symTime, numTime,fType = np.genfromtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/'+datafile, unpack=True,usecols=(1, 2,3,4,5))
   #  fName = np.genfromtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_max_factorization.txt', unpack=True, usecols=(0),dtype=None)


    xlabel = "Number of cores"
    ylabel = "Walltime (s)"


    fig, _ = plt.subplots(nrows=1, ncols=3, figsize=(9.5, 3.5))

    mymarker = itertools.cycle(list('osd*h>v<*'))
    mycolor = itertools.cycle(list('bgrgbk'))
  #  fList=['tube', 'wire', 'diamond']
    mList=[512000,128000,64000]
    for i in range(3):
        plt.subplot(1, 3, i+1)
    #    xMin = nCores[np.logical_and(matSize==8000,fList[i] in fName)] # gave false for fList in condition
        xMin = nCores[np.logical_and(matSize==8000,fType==i)]
        yMinSym = symTime[np.logical_and(matSize==8000,fType==i)]
        yMinNum = numTime[np.logical_and(matSize==8000,fType==i)]
        yMinTot = yMinSym + 2.0 * yMinNum
        xMax=nCores[np.logical_and(matSize==mList[i],fType==i)]
        yMaxSym = symTime[np.logical_and(matSize==mList[i],fType==i)]
        yMaxNum = numTime[np.logical_and(matSize==mList[i],fType==i)]
        yMaxTot = yMaxSym + 2.0 * yMaxNum
      #  currentmarker = mymarker.next()
        plt.xscale('log',basex=2)
        plt.yscale('log')
        plt.grid(which='major', axis='y')
        if (i==1): plt.xlabel(xlabel)
        if (i==0): plt.ylabel(ylabel)
        plt.xlim(10, 700)
#         plt.ylim(0.5, 1500)
        plt.ylim(1, 500)
#         plt.plot(xMin, yMinSym, linestyle='None', label='sym. 8000', marker='o', markersize=ms, mfc='blue', mec='blue')
#         plt.plot(xMin, yMinNum, linestyle='None', label='num. 8000', marker='s', markersize=ms, mfc='blue', mec='blue')
#         plt.plot(xMin, yMinTot, linestyle='None', label='tot. 8000', marker='d', markersize=ms, mfc='blue', mec='blue')
        plt.plot(xMax, yMaxSym, linestyle='None', label='sym. ', marker='o', markersize=ms, mfc='blue', mec='blue')
        plt.plot(xMax, yMaxNum, linestyle='None', label='num. ', marker='s', markersize=ms, mfc='green', mec='green')
        plt.plot(xMax, yMaxTot, linestyle='None', label='tot. ', marker='d', markersize=ms, mfc='red', mec='red')
        if i<2:
            plt.legend(loc='upper left', prop={'size':legendsize},ncol=1)
        else:
            plt.legend(loc='lower left', prop={'size':legendsize},ncol=1)
        xticklabel = [ 16,32, 64,128, 256, 512]
        plt.xticks(xticklabel, map(lambda i : "%d" % i, xticklabel))
        if i>0: plt.gca().get_yaxis().set_ticklabels([])



    fig.tight_layout()
    figureFile=sys._getframe().f_code.co_name[2:]+".eps"
    plt.savefig(figureFile)
    return

def doOneSliceFigure():
    nSlices, nCores,setupTime,solveTime, totalTime = np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_1slice.txt", unpack=True, usecols=(1, 2,6,7, 8))
    T_size, T_nSlices, T_nCores,T_setupTime,T_solveTime, T_totalTime = np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_T.txt", unpack=True, usecols=(1, 2, 3,7,8, 9))
    x = nCores[(nSlices == 1)]
    y = totalTime[(nSlices == 1)]
    x2 = nCores[(nSlices > 1)]
    y2 = totalTime[(nSlices > 1)]
    x3 = T_nCores[(T_size == 8000)]
    y3 = T_totalTime[(T_size == 8000)]
    x4 = [1, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384]

    tSetupVec=setupTime[(nSlices > 1)]
    tSetup=sum(tSetupVec)/len(tSetupVec)
    xAmdahl=getExtendedArray(x2, 40000)
    yAmdahl=getAmdahlLaw(y[0], tSetup, xAmdahl)

    tSetupVec2=T_setupTime[(T_size == 8000)]
    tSetup2=sum(tSetupVec2)/len(tSetupVec2)
    xAmdahl2=getExtendedArray(x3, 40000)
    yAmdahl2=getAmdahlLaw(y3[0], tSetup2, xAmdahl2/16.0)

    plt.figure()
    ls=14
    fs=14
    font = {'weight' : 'normal',
        'size'   : fs}
    matplotlib.rc('font', **font)

    plt.xscale('log',basex=2)
    plt.yscale('log')
    plt.xlabel('Number of cores')
    plt.ylabel('Time to solution (s)')
    plt.xticks(x4, map(lambda i : "%d" % i, x4))
    plt.xlim([0.5, 35000])
    plt.ylim([0.1, 5000])

    plt.plot(x, y, label='single slice', linestyle='None', marker='o', markersize=8, color='black')
    plt.plot(x4, [y[0] / (x4[i] / x4[0]) for i in range(len(x4))], linestyle='-', color='black')
    plt.plot(x2, y2, label='one core per slice', linestyle='None', marker='s', markersize=8, color='red')
    plt.legend(loc='lower left', prop={'size':ls})

    plt.savefig('SIPs_single_slice1.eps')
#    plt.plot(x2,tAmdahl , label="Amdahl's law", linestyle='--', marker='.', color='green')
    plt.plot(xAmdahl,yAmdahl , linestyle='--', marker='.', color='red')
    plt.legend(loc='lower left', prop={'size':ls})
    plt.savefig('SIPs_single_slice2.eps')

    plt.plot(x3, y3, label='16 cores per slice', linestyle='None', marker='d', markersize=8, color='blue')
    plt.plot(x3, [y3[0] / (x3[i] / x3[0]) for i in range(len(x3))], linestyle='-', color='blue')
    plt.plot(xAmdahl2,yAmdahl2 , linestyle='--', marker='.', color='blue')
    plt.legend(loc='lower left', prop={'size':ls})
    plt.savefig('SIPs_single_slice3.eps')

 #   plt.plot(x3,tAmdahl , label="Amdahl's law", linestyle='--', marker='.', color='red')


    return

def doOneSliceProfileFigure():
    nSlices, nCores, totalTime, solTime, symTime, numTime, ortTime = np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_1slice.txt", unpack=True, usecols=(1, 2, 8, 15, 16, 17, 18))
 #   mydata=loadData(dataFile)
    font = {'weight' : 'normal',
        'size'   : 12}
    matplotlib.rc('font', **font)
    myxlim = [0.8, 30000]
    myylim = [0.3, 7000]
    mylegendsize = 10
    myx = [1, 4, 16, 64, 256, 1024, 4096, 16394]
    xticklabel = [ 1, 4, 16, 64, 1024,16384]


    plt.figure()
    fig, ax = plt.subplots(nrows=2, ncols=2)  # ,figsize=(8,6))
    plt.setp(ax, xlim=myxlim, ylim=myylim, xscale='log', yscale='log')
    x = nCores[(nSlices == 1)]
    mymarker5 = itertools.cycle(list('osd^*>v<*'))
    mymarker6 = itertools.cycle(list('osd^*>v<*'))
    mycolor5 = itertools.cycle(list('rgbkcmyk'))
    mycolor6 = itertools.cycle(list('rgbkcmyk'))

    plt.subplot(2, 2, 1)
    y = symTime[(nSlices == 1)]
    plt.plot(myx, [y[0] / myx[i] / myx[0] for i in range(len(myx))], linestyle='-', color='black')
    plt.plot(x, y, label='sym. (single slice)', linestyle='None', marker=next(mymarker5), markersize=5, mfc='black', mec='black')
    x2 = nCores[(nSlices > 1)]
    y2 = symTime[(nSlices > 1)]
    plt.plot(x2, y2, label='sym. (one core per slice)', linestyle='None', marker=next(mymarker6), markersize=5, mfc='red', mec='red')

    plt.ylabel('Walltime (s)')
    plt.legend(loc='upper right', prop={'size':mylegendsize})
    plt.xscale('log',basex=2)
    plt.yscale('log')
    plt.xlim(myxlim)
    plt.ylim(myylim)
    plt.gca().get_xaxis().set_ticklabels([])

    plt.subplot(2, 2, 2)
    y = numTime[(nSlices == 1)]
    plt.plot(myx, [y[0] / myx[i] / myx[0] for i in range(len(myx))], linestyle='-', color='black')
    plt.plot(x, y, label='num. (single slice)', linestyle='None', marker=next(mymarker5), markersize=5, mfc='black', mec='black')
    x2 = nCores[(nSlices > 1)]
    y2 = numTime[(nSlices > 1)]
    plt.plot(x2, y2, label='num. (one core per slice)', linestyle='None', marker=next(mymarker6), markersize=5, mfc='red', mec='red')

    xAmdahl=getExtendedArray(x, 2e5)
    tAmdahl=getAmdahlLaw(y[0], y[0]/64.0, xAmdahl)
    plt.plot(xAmdahl,tAmdahl , linestyle='--', marker='.', color='red')

    plt.legend(loc='upper right', prop={'size':mylegendsize})
    plt.xscale('log',basex=2)
    plt.yscale('log')
    plt.xlim(myxlim)
    plt.ylim(myylim)
    plt.gca().get_xaxis().set_ticklabels([])
    plt.gca().get_yaxis().set_ticklabels([])


    plt.subplot(2, 2, 3)
    y = solTime[(nSlices == 1)]
    plt.plot(myx, [y[0] / myx[i] / myx[0] for i in range(len(myx))], linestyle='-', color='black')
    plt.plot(x, y, label='sol. (single slice)', linestyle='None', marker=next(mymarker5), markersize=5, mfc='black', mec='black')
    x2 = nCores[(nSlices > 1)]
    y2 = solTime[(nSlices > 1)]
    plt.plot(x2, y2, label='sol. (one core per slice)', linestyle='None', marker=next(mymarker6), markersize=5, mfc='red', mec='red')
    plt.xlabel('Number of cores')
    plt.ylabel('Walltime (s)')
    plt.legend(loc='upper right', prop={'size':mylegendsize})
    plt.xscale('log',basex=2)
    plt.yscale('log')
    plt.xlim(myxlim)
    plt.ylim(myylim)
    plt.xticks(xticklabel, map(lambda i : "%d" % i, xticklabel))

    plt.subplot(2, 2, 4)
    y = ortTime[(nSlices == 1)]
    plt.plot(myx, [y[0] / myx[i] / myx[0] for i in range(len(myx))], linestyle='-', color='black')
    plt.plot(x, y, label='ort. (single slice)', linestyle='None', marker=next(mymarker5), markersize=5, mfc='black', mec='black')
    x2 = nCores[(nSlices > 1)]
    y2 = ortTime[(nSlices > 1)]
    plt.plot(x2, y2, label='ort. (one core per slice)', linestyle='None', marker=next(mymarker6), markersize=5, mfc='red', mec='red')
    plt.xscale('log',basex=2)
    plt.yscale('log')
    plt.xlim(myxlim)
    plt.ylim(myylim)
    plt.xlabel('Number of cores')
    plt.gca().get_yaxis().set_ticklabels([])
    plt.legend(loc='upper right', prop={'size':mylegendsize})
    plt.xticks(xticklabel, map(lambda i : "%d" % i, xticklabel))
    fig.tight_layout()

    plt.savefig('SIPs_single_slice_profile.eps')
    return

def makeElementalPlots():
    font = {'weight' : 'normal',
        'size'   : 17}
    matplotlib.rc('font', **font)
    legendsize = 12
    ms = 8
    xlabel = "Number of basis functions"
    ylabel = "Walltime (s)"
    mSize,t1024,t256,t64 = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_elemental.txt', unpack=True)
    mymarker = itertools.cycle(list('osd*h>v<*'))
    mycolor = itertools.cycle(list('bgrgbk'))
    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='major', axis='y')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(5e3, 5e5)
    plt.ylim(5, 8000)
    plt.plot(mSize,t1024,linestyle='None', label='16384 cores', marker=mymarker.next(), markersize=ms, mfc=mycolor.next())
    plt.plot(mSize,t256,linestyle='None', label='4096cores', marker=mymarker.next(), markersize=ms, mfc=mycolor.next())
    plt.plot(mSize,t64,linestyle='None', label='1024 cores', marker=mymarker.next(), markersize=ms, mfc=mycolor.next())
    plt.savefig("paper_elemental.png")
    plt.legend(loc='lower left', prop={'size':legendsize})

    return


def doSIPsScalingPlots():
    # strong scaling plots
    font = {'weight' : 'normal',
        'size'   : 12}
    matplotlib.rc('font', **font)
    legendsize = 10
    ms = 8
    TmatrixSize, TnCores, Tratios, TsolveTime, TtotalTime, Tncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_T.txt', unpack=True, usecols=(1, 3, 6, 9, 9, 10))
    WmatrixSize, WnCores, Wratios, WsolveTime, WtotalTime, Wncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_W.txt', unpack=True, usecols=(1, 3, 6, 9, 9, 10))
    DmatrixSize, DnCores, Dratios, DsolveTime, DtotalTime, Dncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_D.txt', unpack=True, usecols=(1, 3, 6, 9, 9, 10))

    matrixSize,Tnnz,Wnnz,Dnnz = np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_fillin.txt", unpack=True, usecols=(0,4,5,6))

    xlabel = r"$N \cdot N_{nz}$"
    ylabel = "Time to solution (s)"

    matlistD = [8000, 16000, 32000]
    matlistW = [8000, 16000, 32000, 64000, 128000]
    matlistT = [8000, 16000, 32000,64000, 128000,256000, 512000]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.5))

    plt.subplot(1,2,1)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='major', axis='y')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(1e10, 1.8e14)
    plt.ylim(1, 2e3)
    mymarker = itertools.cycle(list('o^sd*h>v<*'))
    mycolorRGB = itertools.cycle(list('rgbk'))

    myLabel = "nanotube"
    x=np.zeros(len(matlistT))
    y=np.zeros(len(matlistT))
    xx=np.zeros(12)
    yy=np.zeros(12)

    for i in range(len(matlistT)):
        y[i] = TsolveTime[np.logical_and(np.logical_and(TnCores == 16384,TmatrixSize == matlistT[i]), Tncoresperslice == 16)]
        x[i] = matlistT[i]*Tnnz[matrixSize == matlistT[i]]
        print x[i],y[i]
    currentmarker = 'o'
    currentcolor = 'b'
    plt.plot(x, y, linestyle='None', label=myLabel, marker=currentmarker, markersize=ms, mfc=currentcolor, mec=currentcolor)
    xx[0:7]=x
    yy[0:7]=y

    myLabel = "nanowire"
    x=np.zeros(len(matlistW))
    y=np.zeros(len(matlistW))
    for i in range(len(matlistW)):
        y[i] = WsolveTime[np.logical_and(np.logical_and(WnCores == 16384,WmatrixSize == matlistW[i]), Wncoresperslice == 16)]
        x[i] = matlistW[i]*Wnnz[matrixSize == matlistW[i]]
        print x[i],y[i]
    currentmarker = 's'
    currentcolor = 'g'
    plt.plot(x, y, linestyle='None', label=myLabel, marker=currentmarker, markersize=ms, mfc=currentcolor, mec=currentcolor)
    xx[7:12]=x
    yy[7:12]=y

    myLabel = "diamond"
    x=np.zeros(len(matlistD))
    y=np.zeros(len(matlistD))
    for i in range(len(matlistD)):
        y[i] = DsolveTime[np.logical_and(np.logical_and(DnCores == 16384,DmatrixSize == matlistD[i]), Dncoresperslice == 16)]
        x[i] = matlistD[i]*Dnnz[matrixSize == matlistD[i]]
        print x[i],y[i]
    currentmarker = 'd'
    currentcolor = 'r'
    plt.plot(x, y, linestyle='None', label=myLabel, marker=currentmarker, markersize=ms, mfc=currentcolor, mec=currentcolor)

    myfit = getPowerFit(x, y)
    plt.plot(x, myfit[0], label='$y={0:.2E}x^{{{1:.2f}}}$'.format(myfit[1], myfit[2]), linestyle='--', color='red')

    myfit = getPowerFit(xx, yy)
    plt.plot(xx, myfit[0], label='$y={0:.2E}x^{{{1:.2f}}}$'.format(myfit[1], myfit[2]), linestyle='-', color='blue')

    plt.legend(loc='lower right', prop={'size':legendsize},ncol=2)


    plt.subplot(1,2,2)

    xlabel = "$N $"
    ylabel = "Time to solution (s)"
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='major', axis='y')
    plt.xlabel(xlabel)
  #  plt.ylabel(ylabel)
    plt.xlim(5e3, 6e5)
    plt.ylim(1, 2e3)
    mymarker = itertools.cycle(list('o^sd*h>v<*'))
    mycolorRGB = itertools.cycle(list('rgbk'))

    myLabel = "nanotube"
    x=np.zeros(len(matlistT))
    y=np.zeros(len(matlistT))
    xx=np.zeros(12)
    yy=np.zeros(12)

    for i in range(len(matlistT)):
        y[i] = TsolveTime[np.logical_and(np.logical_and(TnCores == 16384,TmatrixSize == matlistT[i]), Tncoresperslice == 16)]
        x[i] = matlistT[i]
        print x[i],y[i]
    currentmarker = 'o'
    currentcolor = 'b'
    plt.plot(x, y, linestyle='None', label=myLabel, marker=currentmarker, markersize=ms, mfc=currentcolor, mec=currentcolor)
    xx[0:7]=x
    yy[0:7]=y

    myLabel = "nanowire"
    x=np.zeros(len(matlistW))
    y=np.zeros(len(matlistW))
    for i in range(len(matlistW)):
        y[i] = WsolveTime[np.logical_and(np.logical_and(WnCores == 16384,WmatrixSize == matlistW[i]), Wncoresperslice == 16)]
        x[i] = matlistW[i]
        print x[i],y[i]
    currentmarker = 's'
    currentcolor = 'g'
    plt.plot(x, y, linestyle='None', label=myLabel, marker=currentmarker, markersize=ms, mfc=currentcolor, mec=currentcolor)
    xx[7:12]=x
    yy[7:12]=y

    myLabel = "diamond"
    x=np.zeros(len(matlistD))
    y=np.zeros(len(matlistD))
    for i in range(len(matlistD)):
        y[i] = DsolveTime[np.logical_and(np.logical_and(DnCores == 16384,DmatrixSize == matlistD[i]), Dncoresperslice == 16)]
        x[i] = matlistD[i]
        print x[i],y[i]
    currentmarker = 'd'
    currentcolor = 'r'
    plt.plot(x, y, linestyle='None', label=myLabel, marker=currentmarker, markersize=ms, mfc=currentcolor, mec=currentcolor)

    myfit = getPowerFit(x, y)
    plt.plot(x, myfit[0], label='$y={0:.2E}x^{{{1:.2f}}}$'.format(myfit[1], myfit[2]), linestyle='--', color='red')

    myfit = getPowerFit(xx, yy)
    plt.plot(xx, myfit[0], label='$y={0:.2E}x^{{{1:.2f}}}$'.format(myfit[1], myfit[2]), linestyle='-', color='blue')

    plt.legend(loc='lower right', prop={'size':legendsize},ncol=2)

#     xticklabel = [ 8000,32000,128000, 512000]
#     plt.tick_params(\
#     axis='x',          # changes apply to the x-axis
#     which='minor',      # both major and minor ticks are affected
#     bottom='off',      # ticks along the bottom edge are off
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
#     plt.xticks(xticklabel, map(lambda i : "%d" % i, xticklabel))

    fig.tight_layout()
    plt.savefig("SIPs_scaling_1d-3d.png")

    # plt.show()
    return

def doSIPsScalingFigure():
    """
    Figure of 2 subplots with y axis SIPs time to solution (tall)
    Left: x axis is $ N \times N_nz $
    Right: x axis is $N$
    """
    # strong scaling plots
    font = {'weight' : 'normal',
        'size'   : 12}
    matplotlib.rc('font', **font)
    legendsize = 11
    ms = 6
    TmatrixSize, TnCores, Tratios, TsolveTime, TtotalTime, Tncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_T.txt', unpack=True, usecols=(1, 3, 6, 9, 9, 10))
    WmatrixSize, WnCores, Wratios, WsolveTime, WtotalTime, Wncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_W.txt', unpack=True, usecols=(1, 3, 6, 9, 9, 10))
    DmatrixSize, DnCores, Dratios, DsolveTime, DtotalTime, Dncoresperslice = np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_D.txt', unpack=True, usecols=(1, 3, 6, 9, 9, 10))

    matrixSize,Tnnz,Wnnz,Dnnz = np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_fillin.txt", unpack=True, usecols=(0,4,5,6))

    matlistD = [8000, 16000, 32000, 64000]
    matlistW = [8000, 16000, 32000, 64000, 128000]
    matlistT = [8000, 16000, 32000,64000, 128000,256000, 512000]

    xT=np.zeros(len(matlistT))
    xT2=np.zeros(len(matlistT))
    yT=np.zeros(len(matlistT))

    for i in range(len(matlistT)):
        yT[i] = min(TsolveTime[np.logical_and(TnCores == 16384,TmatrixSize == matlistT[i])])
        xT[i] = matlistT[i]*Tnnz[matrixSize == matlistT[i]]
        xT2[i] = matlistT[i]

    xW=np.zeros(len(matlistW))
    xW2=np.zeros(len(matlistW))
    yW=np.zeros(len(matlistW))
    for i in range(len(matlistW)):
        yW[i] = min(WsolveTime[np.logical_and(WnCores == 16384,WmatrixSize == matlistW[i])])
        xW[i] = matlistW[i]*Wnnz[matrixSize == matlistW[i]]
        xW2[i] = matlistW[i]

    xD=np.zeros(len(matlistD))
    xD2=np.zeros(len(matlistD))
    yD=np.zeros(len(matlistD))
    for i in range(len(matlistD)):
        yD[i] = min(DsolveTime[np.logical_and(DnCores == 16384,DmatrixSize == matlistD[i])])
        xD[i] = matlistD[i]*Dnnz[matrixSize == matlistD[i]]
        xD2[i] = matlistD[i]

    xTW=np.concatenate([xT,xW])
    xTW2=np.concatenate([xT2,xW2])
    yTW=np.concatenate([yT,yW])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.5))

    xlabel = r"$N \cdot N_{nz}$"
    ylabel = "Time to solution, $t$ (s)"
    plt.subplot(1,2,1)
    plt.xscale('log')
    plt.yscale('log')
#     plt.grid(which='major', axis='y')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(1e10, 1.8e14)
    plt.ylim(1, 2e6)

    p1,= plt.plot(xT, yT, linestyle='None', label="nanotube", marker='o', markersize=ms, mfc='b', mec='b')
    p2,= plt.plot(xW, yW, linestyle='None', label='nanowire', marker='s', markersize=ms, mfc='g', mec='g')
    p3,= plt.plot(xD, yD, linestyle='None', label='diamond', marker='d', markersize=ms, mfc='r', mec='r')

    plegend = plt.legend(handles=[p1,p2,p3], loc='upper left', prop={'size':legendsize})
    plt.gca().add_artist(plegend)

    myfit = getPowerFit(xD, yD)
    mylabel=r'$t_{SIPs}=$'+getSciNotation(myfit[1])+r'$(N\cdot N_{{nz}})^{{{0:.2f}}}$'.format(myfit[2])
    f1,=plt.plot(xD, myfit[0], label=mylabel, linestyle='--', color='red')

    myfit = getPowerFit(xTW, yTW)
    mylabel=r'$t_{SIPs}=$'+getSciNotation(myfit[1])+r'$(N\cdot N_{{nz}})^{{{0:.2f}}}$'.format(myfit[2])
    f2,=plt.plot(xTW, myfit[0], label=mylabel, linestyle='--', color='blue')

    plt.legend(handles=[f1,f2],loc='upper right', prop={'size':legendsize})


    plt.subplot(1,2,2)

#     xlabel = "$N$"
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.grid(which='major', axis='y')
#     plt.xlabel(xlabel)
#     plt.xlim(5e3, 6e5)
#     plt.ylim(1, 2e3)
#
#     p1,=plt.plot(xT2, yT, linestyle='None', label="nanotube", marker='o', markersize=ms, mfc='b', mec='b')
#     p2,=plt.plot(xW2, yW, linestyle='None', label='nanowire', marker='s', markersize=ms, mfc='g', mec='g')
#     p3,=plt.plot(xD2, yD, linestyle='None', label='diamond', marker='d', markersize=ms, mfc='r', mec='r')
#
#   #  plegend = plt.legend(handles=[p1,p2,p3], loc='upper left')
#    # plt.gca().add_artist(plegend)
#
#     myfit = getPowerFit(xD2, yD)
#     mylabel=r'$t=$'+getSciNotation(myfit[1])+r'$N^{{{0:.2f}}}$'.format(myfit[2])
#     f1,=plt.plot(xD2, myfit[0], label=mylabel, linestyle='--', color='red')
#
#     myfit = getPowerFit(xTW2, yTW)
#     mylabel=r'$t=$'+getSciNotation(myfit[1])+r'$N^{{{0:.2f}}}$'.format(myfit[2])
#     f2,=plt.plot(xTW2, myfit[0], label=mylabel, linestyle='-', color='blue')
#
#     plt.legend(handles=[f1,f2],loc='lower right', prop={'size':legendsize})
#     plt.gca().get_yaxis().set_ticklabels([])
    plotSIPsElementalScaling(1)
    plt.gca().get_yaxis().set_ticklabels([])



    fig.tight_layout()
    figureFile=sys._getframe().f_code.co_name[2:]+".eps"
    plt.savefig(figureFile)
    # plt.show()
    return



def doNonzerosFigure():
    func=sys._getframe().f_code.co_name[2:]
    datafile="/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_fillin.txt"
    print '\n\n Generating nonzeros figure using data file:',datafile
    mydata = np.loadtxt(datafile, unpack=True)
    mylabel =['nanotube','nanowire','diamond','nanotube','nanowire','diamond']
    mycolor=['blue','green','red','blue','green','red']
    mymarker=['o','s','d','o','s','d']
    mymfc=['none','none','none','blue','green','red']

    plt.figure()
    font = {'weight' : 'normal',
        'size'   : 14}
    legendsize=13

    matplotlib.rc('font', **font)
    x = mydata[0]
    plt.xscale('log',basex=2)
    plt.yscale('log')
#    plt.xlabel(r"Number of basis functions ($\times$ x 1000)")
    plt.xlabel(r"Number of basis functions")
    plt.ylabel("Number of nonzeros")
    plt.xlim([5E3, 1E6])
    plt.ylim([1E5, 1E12])
    plt.grid(which='major', axis='y')
    xticklabel = [ 8000,32000,128000, 512000]
   # xticklabel2 = [ 8,32,128, 512]
    plt.tick_params(\
    axis='x',          # changes apply to the x-axis
    which='minor',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
    plt.xticks(xticklabel, map(lambda i : "%d" % i, xticklabel))
    plt.legend(loc='upper left', prop={'size':legendsize},ncol=2)
    for n in range(6):#[5,4,3,2,1,0]:
        y=mydata[n+1]
        plt.plot(x, y, label=mylabel[n], linestyle='None', marker=mymarker[n], markersize=8, mfc=mymfc[n],mec=mycolor[n], color=mycolor[n])
        plt.plot(x, [y[0] * (x[i] / x[0]) for i in range(len(x))], linestyle='--', color=mycolor[n])
        if n==5:
            myfit = getPowerFit(x[:4], y[:4])
            amp=myfit[1]
            index=myfit[2]
            fitlabel=r'$N_{nz}=$'+r'${0:.2f}N^{{{1:.2f}}}$'.format(myfit[1], myfit[2])
            plt.plot(x, amp * (x ** index), label=fitlabel, linestyle=':', color='red')
        if n==2:
            plt.plot(x, [x[i] * (x[i]+1) / 2 for i in range(len(x))], marker='*', label='dense', ms=12, linestyle='-', color='black')
            plt.legend(loc='upper left', prop={'size':legendsize},ncol=1)
            figureFile=func+"1.eps"
            plt.savefig(figureFile)

    #matplotlib.rc('text', usetex=True)
    plt.legend(loc='upper left', prop={'size':legendsize},ncol=2)

    #plt.savefig("SIPs_nonzeros.png")
    figureFile=func+".eps"
    plt.savefig(figureFile)
    return

def plotNonzerosFigure():
    datafile="/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_fillin.txt"
    print '\n\n Generating nonzeros figure using data file:',datafile
    mydata = np.loadtxt(datafile, unpack=True)
    mylabel =['nanotube','nanowire','diamond','nanotube','nanowire','diamond']
    mycolor=['blue','green','red','blue','green','red']
    mymarker=['o','s','d','o','s','d']
    mymfc=['none','none','none','blue','green','red']

    plt.figure()
    x = mydata[0]
    for n in [5,4,3]:
        y=mydata[n+1]
        if n==5:
            myfit = getPowerFit(x[:4], y[:4])
            amp=myfit[1]
            index=myfit[2]
            fitlabel=r'$N_{nz}=$'+r'${0:.2f}N^{{{1:.2f}}}$'.format(myfit[1], myfit[2])
            plt.plot(x, amp * (x ** index), label=fitlabel, linestyle=':', color='red')
        plt.plot(x, y, label=mylabel[n], linestyle='None', marker=mymarker[n], markersize=8, mfc=mymfc[n],mec=mycolor[n], color=mycolor[n])
        plt.plot(x, [y[0] * (x[i] / x[0]) for i in range(len(x))], linestyle='--', color=mycolor[n])
        if n==3:    plt.plot(x, np.multiply(x, x), marker='*', label='dense', ms=12, linestyle='-', color='black')

    #matplotlib.rc('text', usetex=True)
    plt.xscale('log',basex=2)
    plt.yscale('log')
#    plt.xlabel(r"Number of basis functions ($\times$ x 1000)")
    plt.xlabel(r"Number of basis functions")
    plt.ylabel("Number of nonzeros")
    plt.xlim([5E3, 1E6])
    plt.ylim([1E6, 1E12])
    plt.grid(which='major', axis='y')
    xticklabel = [ 8000,32000,128000, 512000]
   # xticklabel2 = [ 8,32,128, 512]
    plt.tick_params(\
    axis='x',          # changes apply to the x-axis
    which='minor',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
    plt.xticks(xticklabel, map(lambda i : "%d" % i, xticklabel))
    plt.legend(loc='upper left', prop={'size':14},ncol=1)
    plt.savefig("SIPs_nonzeros.eps")
    return



def makeProfilePlots():
    data = np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_tube64k_profile.txt", unpack=True)
    x = data[1]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    plt.setp(axes, xlim=[0.8, 20], ylim=[0.1, 100])
    plt.subplot(2, 2, 1)
    plt.plot(x, data[3], label='sym. (AMD)', linestyle='None', marker='.', markersize=8, mfc='black', mec='black')
    plt.plot(x, [data[3][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='black')
    plt.plot(x, data[7], label='sym. (PM)', linestyle='None', marker='+', markersize=8, mfc='none', mec='red')
    plt.plot(x, [data[7][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='red')
    plt.plot(x, data[11], label='sym. (PS) ', linestyle='None', marker='*', markersize=8, mfc='none', mec='blue')
    plt.plot(x, [data[11][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Walltime (s)')
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticklabels([])
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.xlim([0.8, 20])
    plt.legend(loc='lower left', prop={'size':10})
 #   plt.savefig('sym_factor.png')
    plt.subplot(2, 2, 2)
    plt.plot(x, data[4], label='num. (AMD)', linestyle='None', marker='.', markersize=8, mfc='black', mec='black')
    plt.plot(x, [data[4][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='black')
    plt.plot(x, data[8], label='num. (PM)', linestyle='None', marker='+', markersize=8, mfc='none', mec='red')
    plt.plot(x, [data[8][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='red')
    plt.plot(x, data[12], label='num. (PS) ', linestyle='None', marker='*', markersize=8, mfc='none', mec='blue')
    plt.plot(x, [data[12][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='blue')
    plt.xscale('log')
    plt.yscale('log')
  #  plt.xlabel('Number of cores')
  #  plt.ylabel('Walltime (s)')
    frame2 = plt.gca()
    frame2.get_xaxis().set_ticklabels([])
    frame2.get_yaxis().set_ticklabels([])
    plt.legend(loc='lower left', prop={'size':10})

    plt.subplot(2, 2, 3)
    plt.plot(x, data[2], label='sol. (AMD)', linestyle='None', marker='.', markersize=8, mfc='black', mec='black')
    plt.plot(x, [data[2][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='black')
    plt.plot(x, data[6], label='sol. (PM)', linestyle='None', marker='+', markersize=8, mfc='none', mec='red')
    plt.plot(x, [data[6][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='red')
    plt.plot(x, data[10], label='sol. (PS) ', linestyle='None', marker='*', markersize=8, mfc='none', mec='blue')
    plt.plot(x, [data[10][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(x, map(lambda i : "%d" % i, x))
    plt.xlabel('Number of cores')
    plt.ylabel('Walltime (s)')
  #  frame1.get_xaxis().set_ticklabels([])
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.legend(loc='lower left', prop={'size':10})
 #   plt.savefig('sym_factor.png')
    plt.subplot(2, 2, 4)
    plt.plot(x, data[5], label='ort. (AMD)', linestyle='None', marker='.', markersize=8, mfc='black', mec='black')
    plt.plot(x, [data[5][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='black')
    plt.plot(x, data[9], label='ort. (PM)', linestyle='None', marker='+', markersize=8, mfc='none', mec='red')
    plt.plot(x, [data[9][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='red')
    plt.plot(x, data[13], label='ort. (PS) ', linestyle='None', marker='*', markersize=8, mfc='none', mec='blue')
    plt.plot(x, [data[13][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of cores')
  #  plt.ylabel('Walltime (s)')
    plt.xticks(x, map(lambda i : "%d" % i, x))
    frame4 = plt.gca()
    frame4.get_yaxis().set_ticklabels([])
    plt.legend(loc='lower left', prop={'size':10})

   # fig.tight_layout()
    plt.savefig('profile.png')
    return

def makeProfilePlots2():
    data = np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_tube64k_profile.txt", unpack=True)
    x = data[1]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    plt.setp(axes, xlim=[0.8, 20], ylim=[1, 100])
    plt.subplot(2, 2, 1)
    plt.plot(x, data[7], label='sym. (PM)', linestyle='None', marker='o', markersize=5, mfc='none', mec='red')
    plt.plot(x, [data[7][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='red')
    plt.plot(x, data[11], label='sym. (PS) ', linestyle='None', marker='o', markersize=5, mfc='blue', mec='blue')
    plt.plot(x, [data[11][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Walltime (s)')
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticklabels([])
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.xlim([0.8, 20])
    plt.legend(loc='lower left', prop={'size':10})
 #   plt.savefig('sym_factor.png')
    plt.subplot(2, 2, 2)
    plt.plot(x, data[8], label='num. (PM)', linestyle='None', marker='s', markersize=5, mfc='none', mec='red')
    plt.plot(x, [data[8][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='red')
    plt.plot(x, data[12], label='num. (PS) ', linestyle='None', marker='s', markersize=5, mfc='blue', mec='blue')
    plt.plot(x, [data[12][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='blue')
    plt.xscale('log')
    plt.yscale('log')
  #  plt.xlabel('Number of cores')
  #  plt.ylabel('Walltime (s)')
    frame2 = plt.gca()
    frame2.get_xaxis().set_ticklabels([])
    frame2.get_yaxis().set_ticklabels([])
    plt.legend(loc='lower left', prop={'size':10})

    plt.subplot(2, 2, 3)
    plt.plot(x, data[6], label='sol. (PM)', linestyle='None', marker='d', markersize=5, mfc='none', mec='red')
    plt.plot(x, [data[6][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='red')
    plt.plot(x, data[10], label='sol. (PS) ', linestyle='None', marker='d', markersize=5, mfc='blue', mec='blue')
    plt.plot(x, [data[10][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(x, map(lambda i : "%d" % i, x))
    plt.xlabel('Number of cores')
    plt.ylabel('Walltime (s)')
  #  frame1.get_xaxis().set_ticklabels([])
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.legend(loc='lower left', prop={'size':10})
 #   plt.savefig('sym_factor.png')
    plt.subplot(2, 2, 4)
    plt.plot(x, data[9], label='ort. (PM)', linestyle='None', marker='^', markersize=5, mfc='none', mec='red')
    plt.plot(x, [data[9][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='red')
    plt.plot(x, data[13], label='ort. (PS) ', linestyle='None', marker='^', markersize=5, mfc='blue', mec='blue')
    plt.plot(x, [data[13][0] / x[i] / x[0] for i in range(len(x))], linestyle='-', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of cores')
  #  plt.ylabel('Walltime (s)')
    plt.xticks(x, map(lambda i : "%d" % i, x))
    frame4 = plt.gca()
    frame4.get_yaxis().set_ticklabels([])
    plt.legend(loc='lower left', prop={'size':10})

   # fig.tight_layout()
    plt.savefig('profile.png')
    return

def doReorderingFigure():
    mytype, ncores, tsym, tnum = np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_chol.txt", unpack=True)

    x_PM_D = ncores[(mytype == 24)]
    x_PM_W = ncores[(mytype == 16)]
    x_PM_T = ncores[(mytype == 8)]
    x_PS_D = ncores[(mytype == 27)]
    x_PS_W = ncores[(mytype == 18)]
    x_PS_T = ncores[(mytype == 9)]
    y1_PM_D = tsym[(mytype == 24)]
    y1_PM_W = tsym[(mytype == 16)]
    y1_PM_T = tsym[(mytype == 8)]
    y1_PS_D = tsym[(mytype == 27)]
    y1_PS_W = tsym[(mytype == 18)]
    y1_PS_T = tsym[(mytype == 9)]
    y2_PM_D = tnum[(mytype == 24)]
    y2_PM_W = tnum[(mytype == 16)]
    y2_PM_T = tnum[(mytype == 8)]
    y2_PS_D = tnum[(mytype == 27)]
    y2_PS_W = tnum[(mytype == 18)]
    y2_PS_T = tnum[(mytype == 9)]

    fig, axes = plt.subplots(nrows=2, ncols=3)
#    axes.tick_params(axis='x',which='minor',bottom='off')
#    axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    font = {'weight' : 'normal',
        'size'   : 12}
    matplotlib.rc('font', **font)
    myxticks= [1, 2, 4, 8, 16, 32, 64]
    myxlim=[0.7,80]
    myylim=[0.7,400]
    for i in range(1,4):
        plt.subplot(2, 3, i)
        x1 = ncores[(mytype == 8*i)] #x_PM_T
        y1 = tsym[(mytype == 8*i)] #y1_PM_T
        x2 = ncores[(mytype == 9*i)] #x_PS_T
        y2 = tsym[(mytype == 9*i)] #y1_PS_T
        plt.plot(x1, y1, label='sym. (PM)', linestyle='None', marker='s', markersize=5, mfc='none', mec='red')
        plt.plot(x1, [y1[0] / x1[j] / x1[0] for j in range(len(x1))], linestyle='-', color='red')
        plt.plot(x2, y2, label='sym. (PS) ', linestyle='None', marker='o', markersize=5, mfc='blue', mec='blue')
        plt.plot(x2, [y2[0] / x2[j] / x2[0] for j in range(len(x2))], linestyle='-', color='blue')
        plt.xscale('log', basex=2)
        plt.yscale('log')
        plt.xlim(myxlim)
        plt.ylim(myylim)
        if i==1:    plt.ylabel('Walltime (s)')
        else:   plt.gca().get_yaxis().set_ticklabels([])
        plt.gca().get_xaxis().set_ticklabels([])
        plt.legend(loc='upper right', prop={'size':10})
        if i==3:    plt.legend(loc='lower left', prop={'size':10})


    for i in range(1,4):
        plt.subplot(2, 3, i+3)
        x1 = ncores[(mytype == 8*i)] #x_PM_T
        y1 = tnum[(mytype == 8*i)] #y1_PM_T
        x2 = ncores[(mytype == 9*i)] #x_PS_T
        y2 = tnum[(mytype == 9*i)] #y1_PS_T

        plt.plot(x1, y1, label='num. (PM)', linestyle='None', marker='s', markersize=5, mfc='none', mec='red')
        plt.plot(x1, [y1[0] / x1[j] / x1[0] for j in range(len(x1))], linestyle='-', color='red')
        plt.plot(x2, y2, label='num. (PS) ', linestyle='None', marker='o', markersize=5, mfc='blue', mec='blue')
        plt.plot(x2, [y2[0] / x2[j] / x2[0] for j in range(len(x2))], linestyle='-', color='blue')
        plt.xscale('log', basex=2)
        plt.yscale('log')
        plt.xticks(myxticks, map(lambda i : "%d" % i, myxticks))
        if i==1:    plt.ylabel('Walltime (s)')
        else:   plt.gca().get_yaxis().set_ticklabels([])
        if i==2:    plt.xlabel('Number of cores')
        plt.xlim(myxlim)
        plt.ylim(myylim)
        plt.legend(loc='upper right', prop={'size':10})
        if i==3:    plt.legend(loc='lower left', prop={'size':10})
    fig.tight_layout()

    plt.savefig('SIPs_reordering.eps')
    return

def plotScotchvsMetis():
    mytype, ncores, tsym, tnum = np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_chol.txt", unpack=True)

    x_PM_D = ncores[(mytype == 24)]
    x_PM_W = ncores[(mytype == 16)]
    x_PM_T = ncores[(mytype == 8)]
    x_PS_D = ncores[(mytype == 27)]
    x_PS_W = ncores[(mytype == 18)]
    x_PS_T = ncores[(mytype == 9)]
    y1_PM_D = tsym[(mytype == 24)]
    y1_PM_W = tsym[(mytype == 16)]
    y1_PM_T = tsym[(mytype == 8)]
    y1_PS_D = tsym[(mytype == 27)]
    y1_PS_W = tsym[(mytype == 18)]
    y1_PS_T = tsym[(mytype == 9)]

    fig, _ = plt.subplots(nrows=1, ncols=3, figsize=(9.5, 3.5))
#    axes.tick_params(axis='x',which='minor',bottom='off')
#    axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    font = {'weight' : 'normal',
        'size'   : 12}
    matplotlib.rc('font', **font)
    myxticks= [1, 2, 4, 8, 16, 32, 64]
    myxlim=[0.7,80]
    myylim=[0.7,400]
    for i in range(1,4):
        plt.subplot(1, 3, i)
        x1 = ncores[(mytype == 8*i)] #x_PM_T
        y1 = tsym[(mytype == 8*i)] #y1_PM_T
        x2 = ncores[(mytype == 9*i)] #x_PS_T
        y2 = tsym[(mytype == 9*i)] #y1_PS_T
        plt.plot(x1, y1, label='ParMETIS', linestyle='None', marker='s', markersize=5, mfc='none', mec='red')
        plt.plot(x1, [y1[0] / x1[j] / x1[0] for j in range(len(x1))], linestyle='-', color='red')
        plt.plot(x2, y2, label='PT-Scotch', linestyle='None', marker='o', markersize=5, mfc='blue', mec='blue')
        plt.plot(x2, [y2[0] / x2[j] / x2[0] for j in range(len(x2))], linestyle='-', color='blue')
        plt.xscale('log', basex=2)
        plt.yscale('log')
        plt.xlim(myxlim)
        plt.ylim(myylim)
        plt.xticks(myxticks, map(lambda i : "%d" % i, myxticks))

        if i==1:    plt.ylabel('Walltime (s)')
        else:   plt.gca().get_yaxis().set_ticklabels([])

        if i==2:
            plt.legend(loc='upper right', prop={'size':10})
            plt.xlabel('Number of cores')

        #if i==3:    plt.legend(loc='lower left', prop={'size':10})
    fig.tight_layout()

    plt.savefig('SIPs_PTScotchvsParMetis.eps')
    return


def doFactorizationScalingFigure():
    # symbolic and numeric factorization
    ptype, psize, tsym, tnum = np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_factorization.txt", unpack=True, usecols=(1, 2, 6, 7))
    font = {'weight' : 'normal',
        'size'   : 12}
    matplotlib.rc('font', **font)
    legendsize = 12
    ms = 5
    fig, _ = plt.subplots(nrows=1, ncols=3, figsize=(9.5, 3.5))

    myxlim=[5000,1.2e6]
    myylim=[1,8e3]
    for p in range(1,4):
        plt.subplot(1, 3, p)
        x = psize[ptype == p]
        y1 = tsym[ptype == p]
        y2 = tnum[ptype == p]

        p1,=plt.plot(x, y1, label='sym.', linestyle='None', marker='o', markersize=ms, mfc='none', mec='red')
        p2,=plt.plot(x, y2, label='num. ', linestyle='None', marker='o', markersize=ms, mfc='blue', mec='blue')
        plt.plot(x, [y1[0] * x[i] / x[0] for i in range(len(x))], linestyle='-', color='red')
        plegend=plt.legend(handles=[p1,p2],loc='upper left', prop={'size':legendsize})
        plt.gca().add_artist(plegend)
        if p<3:
            plt.plot(x, [y2[0] * x[i] / x[0] for i in range(len(x))], linestyle='-', color='blue')
        plt.xscale('log')
        plt.yscale('log')
        if p==1: plt.ylabel(r'Walltime, $t$ (s)')
        if p==2: plt.xlabel(r'Number of basis functions, $N$')
        if p==3:
            myfit = getPowerFit(x[0:3], y2[0:3])
            mylabel=r'$t=$'+getSciNotation(myfit[1])+r'$N^{{{0:.2f}}}$'.format(myfit[2])
            f1,=plt.plot(x[0:3], myfit[0], label=mylabel, linestyle='--', color='blue')
            plt.legend(handles=[f1],loc='lower right', prop={'size':legendsize})

        plt.xlim(myxlim)
        plt.ylim(myylim)
        if p>1: plt.gca().get_yaxis().set_ticklabels([])
        #myxticks=[8000,64000,512000]
        #plt.xticks(myxticks, map(lambda i : "%d" % i, myxticks))

    fig.tight_layout()
    figureFile=sys._getframe().f_code.co_name[2:]+".eps"
    plt.savefig(figureFile)
    return


def makeSparsityPlots():
    import scipy.io as spio
    import scipy.sparse as sp
    mmdir = "/Volumes/s/matrices/matrixmarket/"
    mmfiles = [mmdir + "nanotube2-r_P2_A.mm", mmdir + "nanowire25-2r_P2_A.mm", mmdir + "diamond-2r_P2_A.mm",
             mmdir + "rcm_tube_P2.mm", mmdir + "rcm_wire25-2r_P2.mm", mmdir + "rcm_diamond-2r_P2.mm",
             mmdir + "lower_tube_P2.mm", mmdir + "lower_wire25_2r_P2.mm", mmdir + "lower_diamond_2r_P2.mm"]
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9.5, 10.5))
    fig.tight_layout()
    for i in range(9):
        M = spio.mmread(mmfiles[i])
     #   M=sp.eye(8000)
        plt.subplot(3, 3, i + 1)
        plt.spy(M, marker='.', markersize=1)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])

    fig.tight_layout()
    plt.savefig("spartsity.png")
    return

def getEigenvalues(eigenfile):
    eigs = np.loadtxt( eigenfile, unpack=True)
    print "\nAnalyzing eigenvalues in ", eigenfile
    analyzeEigs(eigs)
    return eigs

def analyzeEigs(evec):
    print 'number of eigenvalues:',len(evec)
    print 'min, max, range of the spectrum:',min(evec),max(evec)
    print 'condition number (abs(max)/abs(min)):',max(abs(evec))/min(abs(evec))
    print 'gap=',evec[len(evec)/2]-evec[len(evec)/2-1]
    interval=[-0.8,0.2]
    eigs=evec[np.logical_and(evec>interval[0],evec<interval[1])]
    dist = [abs(eigs[j] - eigs[j + 1]) for j in range(len(eigs) - 1)]
    print 'number of eigenvalues in', interval,':', len(eigs)
    print 'min and max distance between eigenvalues in', interval,':', min(dist), max(dist)
    return 0

def makeEigenvalueSpectrumFigure():
#    eigsFile = ['eigs_diamond-2r_P4.txt', 'eigs_wire25-2r_P4.txt', 'eigs_nanotube2-r_P4.txt']
    eigsFile = ['eigs_diamond-2r_P2.txt', 'eigs_wire25-2r_P2.txt', 'eigs_nanotube2-r_P2.txt']
    print "\n\n Generating eigenvalue spectrum figure using data file:", eigsFile
    fig, axes = plt.subplots(nrows=3, ncols=1)
    plt.setp(axes, xlim=[-1, 2], ylim=[0, 100], yticks=[25, 50, 75])
    nBins=1500
    myRange=[-1.0, 5.5]
    myColor = itertools.cycle(list('rgb'))

    for i in range(3):
        eigs = getEigenvalues( eigsFile[i])
        plt.subplot(3, 1, i+1)
        theColor=myColor.next()
        plt.hist(eigs, bins=nBins, range=myRange, color=theColor, edgecolor=theColor)
        plt.xlim([-1, 2])
        plt.ylim([0, 75])
        plt.yticks([25, 50, 75])
        if i<2:
            frame1 = plt.gca()
            frame1.get_xaxis().set_ticklabels([])
        if i==2:
            plt.xlabel("Eigenvalue spectrum (au)")
        if i==1:
            plt.ylabel("Number of eigenvalues")

    fig.add_axes([0.55, 0.2, 0.3, 0.1])
    plt.hist(eigs, bins=nBins, range=myRange, color='blue', edgecolor='blue')
    plt.xlim([2.0, 5.5])
    plt.ylim([0, 10])
    plt.xticks([2.0, 3.0, 4.0, 5.0])
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticklabels(['2.0', '3.0', '4.0', '5.0'])

    plt.yticks([5, 10])

    plt.savefig("paper_spectra.png")
    return

def doEigenvalueSpectrumFigure():
#    eigsFile = ['eigs_diamond-2r_P4.txt', 'eigs_wire25-2r_P4.txt', 'eigs_nanotube2-r_P4.txt']
    eigsFile = ['eigs_diamond-2r_P2.txt', 'eigs_wire25-2r_P2.txt', 'eigs_nanotube2-r_P2.txt']
    print "\n\n Generating eigenvalue spectrum figure using data file:", eigsFile
    fig, axes = plt.subplots(nrows=3, ncols=1)
    plt.setp(axes, xlim=[-1, 2], ylim=[0, 100], yticks=[25, 50, 75])
    nBins=1024
    myRange=[-0.8, 0.2]
    myColor = itertools.cycle(list('rgb'))

    for i in range(3):
        eigs = getEigenvalues( eigsFile[i])
        plt.subplot(3, 1, i+1)
        theColor=myColor.next()
        plt.hist(eigs, bins=nBins, range=myRange, color=theColor, edgecolor=theColor)
       # plt.xlim([-1, 0.4])
        plt.ylim([0, 20])
        #plt.yticks([25, 50, 75])
        if i<2:
            frame1 = plt.gca()
            frame1.get_xaxis().set_ticklabels([])
        if i==2:
         #   plt.xlabel("Eigenvalue spectrum (au)")
            plt.xlabel("Eigenvalue histogram (Hartree)")
        if i==1:
            plt.ylabel("Number of eigenvalues")


    plt.savefig("SIPs_spectra.eps")
    return

def doEigenvalueSpectrumFigure3():
#    eigsFile = ['eigs_diamond-2r_P4.txt', 'eigs_wire25-2r_P4.txt', 'eigs_nanotube2-r_P4.txt']
    eigsFile = ['eig.8000.16384.txt', 'eig.16000.16384.txt', 'eig.32000.16384.txt']
    print "\n\n Generating eigenvalue spectrum figure using data file:", eigsFile
    fig, axes = plt.subplots(nrows=3, ncols=1)
    plt.setp(axes, xlim=[-1, 2], ylim=[0, 100], yticks=[25, 50, 75])
    nBins=100
   # myRange=[-0.8, 0.2]
    myColor = itertools.cycle(list('rgb'))

    for i in range(3):
        eigs = getEigenvalues( "/Volumes/s/keceli/Dropbox/work/SEMO/data/EL_eigs/" + eigsFile[i])
        plt.subplot(3, 1, i+1)
        theColor=myColor.next()
        plt.hist(eigs, bins=nBins, color=theColor, edgecolor=theColor)
       # plt.xlim([-1, 0.4])
        plt.ylim([0, 20])
        #plt.yticks([25, 50, 75])
        if i<2:
            frame1 = plt.gca()
            frame1.get_xaxis().set_ticklabels([])
        if i==2:
         #   plt.xlabel("Eigenvalue spectrum (au)")
            plt.xlabel("Eigenvalue histogram (Hartree)")
        if i==1:
            plt.ylabel("Number of eigenvalues")


    plt.show()
    return

def makeSpectrumClusterPlots2():
    fig, axes = plt.subplots(nrows=3, ncols=3)
    mycolor = itertools.cycle(list('rgb'))
    plt.setp(axes, xlim=[0.1, 1000], ylim=[1e-8, 0.5])
    plt.setp(axes, xscale='log')
   # eigsfile = ['eigs_diamond-r_P2.txt', 'eigs_wire25-2r_P2.txt', 'eigs_nanotube2-r_P2.txt', 'eigs_diamond_2r_P4.txt', 'eigs_wire25_2r_P4.txt', 'eigs_tube_P4.txt']
    eigsfile = ['eigs.diamond-2r.P2.1024.16.BCDEAT.26s21m08h.txt', 'eigs.nanowire25-2r.P2.1024.16.BCDEAT.23s29m23h.txt', 'eigs.nanotube2-r.P2.1.1.ABCDET.40s48m18h.txt',
                'eigs.diamond-2r.P4.1024.16.BCDEAT.02s22m08h.txt', 'eigs.nanowire25-2r.P4.256.16.ABCEDT.55s54m06h.txt', 'eigs.nanotube2-r.P4.256.16.ABCEDT.50s14m01h.txt',
                'eigs.diamond-2r.P6.1024.16.BCDEAT.26s06m02h.txt', 'eigs.nanowire25-2r.P6.1024.16.BCDEAT.02s17m23h.txt', 'eigs.nanotube2-r.P6.1024.16.BCDEAT.23s58m16h.txt']
    for i in range(9):
    #    eigs = np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/" + eigsfile[i], unpack=True)
        eigs = np.loadtxt("/Volumes/s/keceli/Dropbox/work/SEMO/data/EL_eigs/" + eigsfile[i], unpack=True)
        eigs = eigs[eigs < 0.2]
        dist = [abs(eigs[j] - eigs[j + 1]) for j in range(len(eigs) - 1)]
        plt.subplot(3, 3, i + 1)
        print min(dist), max(dist)
        clr = mycolor.next()
        plt.hist(dist, bins=10 ** np.linspace(np.log10(min(dist)), np.log10(max(dist)), 100), color=clr, edgecolor=clr)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim([0.1, 1000])
        plt.xlim([1e-8, 0.5])

    plt.show()
    return

def makeSpectrumClusterPlots3():
    fig, axes = plt.subplots(nrows=3, ncols=3)
    mycolor = itertools.cycle(list('rgb'))
    plt.setp(axes, xlim=[0.1, 1000], ylim=[1e-8, 0.5])
    plt.setp(axes, xscale='log')
    eigsFile = ['eig.8000.16384.txt', 'eig.16000.16384.txt', 'eig.32000.16384.txt']
    for i in range(3):
    #    eigs = np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/" + eigsfile[i], unpack=True)
        eigs = np.loadtxt("/Volumes/s/keceli/Dropbox/work/SEMO/data/EL_eigs/" + eigsFile[i], unpack=True)
        eigs = eigs[eigs < 0.2]
        dist = [abs(eigs[j] - eigs[j + 1]) for j in range(len(eigs) - 1)]
        plt.subplot(3, 1, i + 1)
        print min(dist), max(dist)
        clr = mycolor.next()
        plt.hist(dist, bins=10 ** np.linspace(np.log10(min(dist)), np.log10(max(dist)), 100), color=clr, edgecolor=clr)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim([0.1, 1000])
        plt.xlim([1e-8, 0.5])

    plt.show()
    return

def doSliceTimingFigure():
    fileDir='/Volumes/s/keceli/Dropbox/work/SEMO/data/logs/tube/iter/'
    logFile64=fileDir+'log.nanotube2-r_P2.nprocEps64p1024.c16.n64.05s01m06h'
    logFile1024=fileDir+'log.nanotube2-r_P2.nprocEps1024p16384.c16.n1024.24s01m06h'
    plt.figure()
    plotSliceTimings(logFile64, 1)
    plt.savefig('nonUniformSlice2.eps')
    plotSliceTimings(logFile1024, 1)
    plt.savefig('nonUniformSlice.eps')
    return

def plotSliceTimings(logFile,embed=0):
    """
    Plot  slice Timings for a given input file
    """
    import parseSIPs
    logging.debug("in plotSliceTimings")
    sliceData=parseSIPs.parseSlicingTimes(logFile, "0,")
    sliceTiming=np.asarray(sliceData[2])+sliceData[0]
    nSlice=len(sliceTiming)
    setupTime=[sliceData[0]]*nSlice

    sliceData2=parseSIPs.parseSlicingTimes(logFile, "2,")
    if len(sliceData2[2])>0:
        sliceTiming2=np.asarray(sliceData2[2])+setupTime
        nSlice2=len(sliceTiming2)


    print nSlice, "slices"
    print 'Iter 0', max(sliceTiming)/min(sliceTiming)
    if len(sliceData2[2])>0:print 'Iter 2', max(sliceTiming2)/min(sliceTiming2)

    # strong scaling plots
    if not embed: plt.figure()
    font = {'weight' : 'normal',
        'size'   : 14}
    matplotlib.rc('font', **font)
    legendsize = 14
    ms = 6
    x=np.arange(-0.8,0.2,1./float(len(sliceTiming)))
    if nSlice<100:
        mfc='none'
    else:
        mfc='r'
   # plt.plot(x,setupTime,ls='-',color='g', lw=3)
    plt.plot(x,sliceTiming,label=str(nSlice)+' uniform slices',ls='none',marker='s', markersize=ms, mfc=mfc, mec='r')
    plt.xlim([-0.85,0.25])
    plt.ylim([2,25]) #max(sliceTiming)*1.1])
    plt.xlabel('Slices')
    plt.ylabel('Time to solution (s)')
    plt.legend(loc='upper right', prop={'size':legendsize})
    if nSlice<100: plt.savefig('nonUniformSlice1.eps')



    if not embed:plt.savefig(logFile+'sliceTiming.png')
    if len(sliceData2[2])>0:
        if nSlice<100:
            mfc='none'
        else:
            mfc='b'
        plt.plot(x,sliceTiming2,label=str(nSlice)+' nonuniform slices',ls='none',marker='o', markersize=ms, mfc=mfc, mec='b')
        plt.legend(loc='upper right', prop={'size':legendsize})

        if not embed:plt.savefig(logFile+'sliceTiming2.png')

    if not embed: plt.show()
    return

def initializeLog(debug):
    import sys
    if debug: logLevel = logging.DEBUG
    else: logLevel = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(logLevel)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logLevel)

    logging.debug("Start in debug mode:")

def getArgs():
    import argparse
    parser = argparse.ArgumentParser(description=
    """
    July 11, 2014
    Murat Keceli
    Generates plots from the results of the SIPs solver @ https://bitbucket.org/hzhangIIT/dftb-eig14.
    """)
    parser.add_argument('input', metavar='FILE', type=str, nargs='?',
        help='Log file to be parsed. All log files will be read if a log file is not specified.')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')
    parser.add_argument('-n', '--nCoresPerSlice', type=int, default=0, nargs='?',
                        help='Speedup, efficiency and profile plots for specific number of cores per slice')
    return parser.parse_args()

def main():
    args = getArgs()
    initializeLog(args.debug)
    if args.input is not None:
        logFile = args.input
        logging.debug('input file given')
        plotSliceTimings(logFile)

    else:
       # makeNonzerosFigure()
       # doEigenvalueSpectrumFigure()
#         doNonzerosFigure()
     #   doFactorizationStronScalingFigure()
     #    doOneSliceFigure()
       #  doOneSliceProfileFigure()
         #doScalingFigure()
       #  plotScotchvsMetis()
        # doSliceTimingFigure()
        # doNonzerosFigure()
         #makeSpectrumClusterPlots2()
         #makeSpectrumClusterPlots3()
         #doEigenvalueSpectrumFigure3()
 #        doOneSliceProfileFigure()
      #  doReorderingFigure()
       # doFactorizationScalingFigure()
  #       doSIPsScalingFigure()
       #  plotSIPsElementalScaling()
         doTOCFigure()
   #      plotSIPsNNnzScalingFigure()
       # plotNonzerosFigure()
       # doReorderingFigure()
   #      doFactorizationStronScalingFigure2()
        #   plt.show()

if __name__ == "__main__":
    main()
