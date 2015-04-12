#!/usr/bin/env python
import numpy as np
import logging
import itertools

#  Grid info: 1x1
#   Reading matrices...
# /home/keceli/scratch/data/matrix/diamond-2r_P2_A.mm.bin , /home/keceli/scratch/data/matrix/diamond-2r_P2_B.mm.bin was read in (s): 24.4631
#   Matrix heights: 8000 , 8000
#   Memory size: 64000000
#   Redundant size: 1
#   Solving matrix pencil...
#   Solved in (s): 977.946
#   Found 4907 eigenpairs in interval -0.8,0.2

keys=[
      "Grid info:",
      " was read in (s):",
      " Matrix heights:",
      " Memory size:",
      " Solved in (s):",
      "interval",
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
    ptype=a[0].replace('log','')
    n=a[2]
    c=a[3]
    mapping=a[4]
    return [ptype,n,c,mapping]

def readLogFile(logfile):
    import socket
    values=["NA"]*(len(keys)+1)

    nameInfo=parseFileName(logfile)
    
    logging.debug("Reading file {0}".format(logfile))
    with open(logfile) as f:
        while True:          
            line = f.readline()
            if not line: 
                break                    
            else:
                for i in range(len(keys)):
                    if keys[i] in line:
                        a=line.split()
                        values[i]=a[-1]
                        if i==len(keys)-1: values[i+1]=a[1]


        print logfile,list2Str(nameInfo),list2Str(values),socket.gethostname()              
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
    Jan 22, 2015
    Murat Keceli
    Parses Elemental output files. More info:  https://github.com/elemental and https://github.com/keceli/eigensolvers
    """)
    parser.add_argument('input', metavar='FILE', type=str, nargs='?',
        help='Log file to be parsed. All log files will be read if a log file is not specified.')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')
    return parser.parse_args()  
          
def main():
    args=getArgs()
    initializeLog(args.debug)
    print "file error s p c n",list2Str(keys)
    if args.input is not None:
        logFile=args.input
        readLogFile(logFile)
    else:
        readLogDirectory()
        

if __name__ == "__main__":
    main()
