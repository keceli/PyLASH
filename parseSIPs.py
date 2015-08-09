#!/usr/bin/env python
import numpy as np
import logging
import itertools

keys=[
      "A:",
      " 2:       SIPSSetUp:",
      " 3:      SIPSSolve0:",
      " 4:      SIPSSolve1:",
      " 5:      SIPSSolve2:",
      " 6:         SIPSDMA:",
      "Memory:",
      "Flops/sec:",
      "-mat_mumps_icntl_7",
      "-mat_mumps_icntl_29",
      "-mat_mumps_icntl_23",
      "-eps_krylovschur_nev",
      "-sips_interval",
      "-eps_tol",
      "Using PETSc arch:",
      "Total "
      ]

long_keys=[
           "PETSc version:",
           "SLEPc version:",
           ]

profile_keys=[
             "MatMult",
             "MatSolve",
             "MatCholFctrSym",
             "MatCholFctrNum",
             "BVOrthogonalize",
             ]

def list2Str(list1):
    return str(list1).replace('[','').replace(']','').replace(',','').replace("'","").replace(":","")

def readLogDirectory():
    import glob
    global filename
    for myfile in glob.glob("log.*"):
        readLogFile(myfile)
    return 0

def parseFileName(filename):
    a=filename.split('.')
    s=a[2].split('p')[2].replace('s','')
    p=a[2].split('p')[3].replace('p','')
    c=a[3].replace('c','')
    n=a[4].replace('n','')
    return [s,p,c,n]

def prune(mystr):
    return mystr.replace('-0.80.2','D').replace('100','D').replace('1.e-8','D').replace(' ','-').replace('-arch-xl-scotch','D')
    
def getRatio(matSize,nRanks):
    return matSize/nRanks**0.5


def parseSlicingTimes(logFile,iter="0,"):
    sliceEig=[]
    sliceTiming=[]
    logging.debug("in parseSlicingTimes")
    with open(logFile) as f:
        while True:          
            line = f.readline()
            if not line: 
                break
            elif line.startswith("Iter"):
                logging.debug("found Iter")
                a=line.split()
                if a[1]==iter:
                    logging.debug("in Iter")
                    line=f.readline()
                    while True: 
                        line=f.readline()
                        if line.startswith("["):
                            logging.debug("found [")
                            a=line.split()
                            sliceTiming.append(float(a[-1]))
                            sliceEig.append(int(a[1]))
                        else:
                            break
            elif "2:       SIPSSetUp" in line:
                a=line.split()
                setupTime=float(a[2])        
    return [setupTime,sliceEig,sliceTiming]  

def readLogFile(logfile):
    import socket
    errorCode="OK"
    values=["NA"]*len(keys)
    long_values=["NA"]*len(long_keys)
    profile_time=[0.0]*len(profile_keys)
    profile_count=[0]*len(profile_keys)
    #log.nanotube2-r_P6.nprocEps16p256.c16.n16.03s30m23h
    spcn="NA"
    spcn=parseFileName(logfile)
    
    logging.debug("Reading file {0}".format(logfile))
    with open(logfile) as f:
        while True:          
            line = f.readline()
            if not line: 
                break
            elif line.startswith("--- Event Stage 2: "):
                while True:
                    line = f.readline()
                    if not line: 
                        break
                    elif line.startswith("--- Event Stage 4:"):
                        break
                    else:
                        for i in range(len(profile_keys)):
                            if line.startswith(profile_keys[i]):
                                profile_count[i]=profile_count[i]+int(line[18:24])
                                profile_time[i]=profile_time[i]+ float(line[29:39])
                    
            else:
                for i in range(len(keys)):
                    if line.startswith(keys[i]):
                        a=line.replace(keys[i],'').split()
                        values[i]=a[0]
                        if i==0: ratio=getRatio(int(a[0]),int(spcn[1]))
                for i in range(len(long_keys)):
                    if line.startswith(long_keys[i]):
                        a=line.split()[-3]+"-"+line.split()[-2]
                        long_values[i]=a
                                                
                if "Error" in line or "ERROR" in line:
                    errorCode="ER"
                    print errorCode, logfile
                if "Performance may be degraded" in line:
                    errorCode="SL"   
                    print errorCode, logfile
        tAll=float(values[1])+float(values[2]) 
        #print logfile,values[0],list2Str(spcn),ratio,list2Str(values[1:3]),tAll,list2Str(profile_count),\
        #    list2Str(profile_time),list2Str(values[3:7]),prune(list2Str(values[7:])),\
        #    prune(list2Str(long_values)),socket.gethostname()   
        print logfile, list2Str(spcn), list2Str(values),list2Str(profile_count),list2Str(profile_time),prune(list2Str(long_values))         
           
    return 0     
    

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
    This code parses SIPs log files, log.*. 
    """
    )
    parser.add_argument('input', metavar='FILE', type=str, nargs='?',
        help='Log file to be parsed. All log files will be read if a log file is not specified.')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')
    parser.add_argument('-n','--nCoresPerSlice', type=int, default=0,nargs='?', 
                        help='Speedup, efficiency and profile plots for specific number of cores per slice')
    return parser.parse_args()  
          
def main():
    args=getArgs()
    initializeLog(args.debug)        
    #print logfile, list2Str(spcn), list2Str(values),list2Str(profile_count),list2Str(profile_time),prune(list2Str(long_values))         

   # print "file s p c n",list2Str(keys),list2Str(profile_keys),list2Str(profile_keys),list2Str(long_keys)
    print "file s p c n A SetUp Solve0 Solve1 Solve2 DMA Memory Flops/sec -mat_mumps_icntl_7 -mat_mumps_icntl_29 -mat_mumps_icntl_23 \
    -eps_krylovschur_nev -sips_interval -eps_tol arch Total  MatMult MatSolve MatCholFctrSym MatCholFctrNum BVOrthogonalize MatMult \
    MatSolve MatCholFctrSym MatCholFctrNum BVOrthogonalize PETScVer SLEPcVer"
    if args.input is not None:
        logFile=args.input
        readLogFile(logFile)
    else:
        readLogDirectory()
        

if __name__ == "__main__":
    main()
