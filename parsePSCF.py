#!/usr/bin/env python
import numpy as np
import logging
import itertools

keys=[
      "Number of atoms:",
      "Distance cutoff:",
      "Number of basis functions  :",
      "Nonzero density percent :",
      "Time (sec):",
      " 0:      Main Stage:",
      " 1:           Input:",
      " 2:       mindo3AIJ: ",
      " 3:            Sort:",
      " 4:      Initialize:",
      " 5:        getGamma:",
      " 6:              F0:",
      " 7:              D0:",
      " 8:               G: ",
      " 9:               H:",
      "10:              FD:"
      "11:           Solve:"
      ]

def list2Str(list1):
    return str(list1).replace('[','').replace(']','').replace(',','').replace("'","").replace(":","")

def readLogDirectory():
    import glob
    global filename
    for myfile in glob.glob("*log*"):
        readLogFile(myfile)
    return 0

def readLogFile(logfile):
    import socket
    errorCode="OK"
    values=["NA"]*len(keys)
    
    logging.debug("Reading file {0}".format(logfile))
    with open(logfile) as f:
        while True:          
            line = f.readline()
            if not line: 
                break
            else:
                for i in range(len(keys)):
                    if line.startswith(keys[i]):
                        a=line.replace(keys[i],'').split()
                        values[i]=a[0] 
        print logfile, list2Str(values)        
           
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
    Nov 30, 2015
    Murat Keceli
    This code parses PSCF log files, log.*. 
    """
    )
    parser.add_argument('input', metavar='FILE', type=str, nargs='?',
        help='Log file to be parsed. All log files (any file with "log" in the filename) will be read if a log file is not specified.')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')

    return parser.parse_args()  
          
def main():
    args=getArgs()
    initializeLog(args.debug)        
    if args.input is not None:
        logFile=args.input
        readLogFile(logFile)
    else:
        readLogDirectory()
        

if __name__ == "__main__":
    main()
