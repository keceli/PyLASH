#!/usr/bin/env python
import numpy as np
import logging
import itertools


def list2Str(list1):
    return str(list1).replace('[','').replace(']','').replace(',','')

def createXYZFile(filename):
    import subprocess
    xyzLines=[]
    with open(filename) as f:
        while True:          
            line = f.readline()
            if not line: 
                return
            if ("No.       Tag          Charge          X              Y              Z") in line:
                line = f.readline()                    
                while True:          
                    line = f.readline()                    
                    if not line.strip (): #line == "\n":                        
                        break
                    xyzLines.append(line)

                break    
    xyzFile=filename.split(".")[1]+".xyz"       
    nLines=len(xyzLines)
    with file(xyzFile, 'w') as out:
        out.write(str(nLines))
        out.write("\n\n")
        for i in range(nLines):
            xyzLine=xyzLines[i].split()
            out.write(xyzLine[1]+"\t"+xyzLine[3]+"\t"+xyzLine[4]+"\t"+xyzLine[5]+"\n")
    pngFile="PNG:"+filename.split(".")[1]+".png"
    command=["s.jmol",xyzFile,"-ion","-q9","-w",pngFile]
    subprocess.call(command)    
    return                         

def createXYZFiles():
    import glob
    for file in glob.glob("log.*d"):
        createXYZFile(file)
    return 0

def readLogDirectory():
    import glob
    for file in glob.glob("log.*"):
        readLogFile(file)
    return 0
def readLogFile(file):
    logging.debug("Reading file {0}".format(file))
    
    with open(file) as f:
        while True:          
            line = f.readline()
            if not line:
                return 
            if "nproc" in line:
                print "Ncores", line.split()[2]            
            if "Group name" in line:
                print "Sym", line.split()[2]
            elif "symmetry detected" in line:
                print "Sym", line.split()[0]
            if "heap     =" in line:
                print "heap",line.split()[5]
            if "stack    =" in line:
                print "stack",line.split()[5]
            if "global   =" in line:
                print "global", line.split()[5]
            if "total    =" in line:
                print "total", line.split()[5]
            if "functions       =" in line:
                print "basis", line.split()[2]
            if "iter       energy          gnorm     gmax       time" in line:
                while "Final RHF  results" not in line:
                    line = f.readline()
                    if len(line.split())==5:
                        if line.split()[0].isdigit():
                            scf_iter= int(line.split()[0])
                print "SCFiter", scf_iter
            if "closed shells   =" in line:
                homo_pos= int(line.split()[3])
            if "Final eigenvalues" in line:
                while "ROHF Final Molecular Orbital Analysis" not in line:
                    line = f.readline()
                    if len(line.split())==2:
                        if line.split()[0].isdigit():
                            eig_pos = int(line.split()[0])
                            if eig_pos==homo_pos:
                                print "HOMO", float(line.split()[1])
                            elif eig_pos==homo_pos+1:
                                print "LUMO", float(line.split()[1])
            if "Total SCF energy =" in line:
                print "SCF", line.split()[4]
            if "Time for solution =" in line:
                print "Tscf", line.split()[4]
#            if "Total energy =" in line:
#                print line.split()[3]
            if "CCSD iterations" in line:
                while "Iterations converged" not in line:
                    line = f.readline()
                    if len(line.split())==6:
                        if line.split()[0].isdigit():
                            ccsd_iter= int(line.split()[0])
                            ccsd_time= float(line.split()[3])
                print "CCSDiter", ccsd_iter
                print "CCSDitertime", ccsd_time
            if "iter     correlation     delta       rms       T2     Non-T2      Main" in line:
                while "*************converged*************" not in line:
                    line = f.readline()
                    if len(line.split())==7:
                        if line.split()[0].isdigit():
                            ccsd_iter= int(line.split()[0])
                            ccsd_time= float(line.split()[6])
                print "CCSDiter", ccsd_iter
                print "CCSDitertime", ccsd_time
            if "CCSD total energy / hartree       =" in line:
                print "CCSD", line.split()[6]
            if "Total CCSD energy:" in line:
                print "CCSD", line.split()[3]
            if "CCSD(T) total energy / hartree       =" in line:
                print "CCSD(T)", line.split()[6]
            if "Total CCSD(T) energy:" in line:
                print "CCSD(T)", line.split()[3]
            if "Max memory consumed for GA by this process:" in line:
                print "MemGA", float(line.split()[8])/1.e6
            if "maximum total M-bytes" in line:
                print "MemHeap",line.split()[3]
                print "MemStack",line.split()[4]
            if "Task  times  cpu:" in line:
                print "Time" ,line.split()[3]
#             if "" in line:
#                 print line.split()[6]
#             if "" in line:
#                 print line.split()[6]
                

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
    July 25, 2014
    Murat Keceli
    Parser for Nwchem log files, log.*. 
    """
    )
    parser.add_argument('input', metavar='FILE', type=str, nargs='?',
        help='Log file to be parsed. All log files (log.*) will be read if a log file is not specified.')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')
    parser.add_argument('-n','--nCoresPerSlice', type=int, default=0,nargs='?', 
                        help='Speedup, efficiency and profile plots for specific number of cores per slice')
    return parser.parse_args()  
          
def main():
    args=getArgs()
    initializeLog(args.debug)
    nCoresPerSlice=args.nCoresPerSlice
    if args.input is not None:
        logFile=args.input
        readLogFile(logFile)
      #  createXYZFile(logFile)
        
    else:
        createXYZFiles()
        


if __name__ == "__main__":
    main()
