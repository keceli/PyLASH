#!/usr/bin/env python
import numpy as np
import logging

def Peter2MMC(datfile, mtxfile,NNZ):
    count=0
    mynnz=0
    trace=0.0
    with open(datfile) as fin, open(mtxfile,"w") as fout:
        a = fin.readline().split()
        N = int(a[4])
        fin.readline()
        fin.readline()
#        line=str(N)+" "+str(N)+" "+str(NNZ)+"\n"
#        fout.write(line)
        while True:
            line = fin.readline()
            if not line:
                print "End of file reached"
                break
            if line.startswith('#'):
                a=line.split()
                a = fin.readline().split()
                percent=float(a[3])
                print "Density percentage: read from file, calculated with total, calculated in triangular",percent,count/(float(N)**2)*100.0,mynnz/(float(N)*float(N-1)/2)*100
                print "Trace of the matrix=",trace
                print "Number of nozeros in dat file, total, in triangular (should be used) ",NNZ,count,mynnz
                break

            i=int(line[0:6])
            j=int(line[6:12])
            v=float(line[12:])

            if (int(i)>N) or (int(j)>N):
                print "One of the indices",i,j,"larger than matrix size", N
                return
            if (count> N*N):
                print "Number of nonzeros exceeded number of elements",count,">",N*N
                return
            mynnz=mynnz+1
            if i==j:
                count=count+1
                trace=trace+v
            else:
                count=count+2
            line=str(i)+" "+str(j)+" "+str(v)+"\n"
            fout.write(line)
        #To prepend the matrix size info to mtxfile
    line1="%%MatrixMarket matrix coordinate real symmetric \n"
    line2=str(N)+" "+str(N)+" "+str(mynnz)+"\n"
    with file(mtxfile, 'r') as original: data = original.read()
    with file(mtxfile, 'w') as modified: modified.write(line1 + line2 + data)
    return

def getNumberofNonzeros(filename):
    NNZ=0
    with open(filename) as fin:
        for line in fin:
            if "nnz" in line:
                a=line.split()
                NNZ=int(a[3])
    if NNZ ==0:
        print "nnz not found"
    return NNZ

def getmtxfile(datfile):
    import subprocess
    NNZ=getNumberofNonzeros(datfile)
    mtxfile=datfile[0:3]+".mtx"
    binfile=datfile[0:3]+".bin"
    Peter2MMC(datfile,mtxfile,NNZ)
    executable="~/Dropbox/work/inertia/petscIO/x.mtx2bin"
    command= [executable,"-fin",mtxfile,"-fout",binfile]
    print executable+" -fin "+mtxfile+" -fout ",binfile
    #subprocess.call(command)
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
     April 21, 2014
     Converts the DAT files to MatrixMarket format.
     """
     )
    parser.add_argument('input', metavar='FILE', type=str, nargs='?',
        help='Run it in the folder with *.DAT files, it will convert them all to *.mtx files.')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')
    return parser.parse_args()

def main():
    args=getArgs()
    initializeLog(args.debug)
    NNZ=0
    if args.input is not None:
        getmtxfile(args.input)
    else:
        import glob
        global filename
        for datfile in glob.glob("*.DAT"):
            getmtxfile(datfile)

if __name__ == "__main__":
    main()
