#!/usr/bin/env python
import logging
from genericpath import isdir

def writeInputFile(templateFile, xyzFile):
    xyzName = xyzFile.replace('.xyz','')
    xyzKey1='put_xyz'
    inputFile= xyzName + '_' + templateFile
    with  open(templateFile,'r') as tmp, open(inputFile,'w') as inp:
        logging.debug("Read {0}, write {1}".format(templateFile,inputFile))
        while True:
            tmpLine=tmp.readline()
            logging.debug("{0} file, line: {1}".format(templateFile,tmpLine))
            if xyzKey1 in tmpLine:
                with open(xyzFile,'r') as xyz:
                    while True:
                        xyzLine=xyz.readline()
                        if not xyzLine:
                            logging.debug("End of file {0}".format(xyzFile))
                            break
                        else:
                            xyzList=xyzLine.split()
                            if len(xyzList)==0:
                                pass
                            elif len(xyzList)==1:
                                nAtoms=int(xyzList[0])
                                logging.debug("File {0} states {1} atoms".format(xyzFile,nAtoms))
                            elif len(xyzList)==4:
                                inp.write(xyzLine)
                            else:
                                logging.info("Weird line in {0} file".format(xyzFile))
            elif not tmpLine:
                logging.debug("End of file {}".format(templateFile))
                break
            else:
                inp.write(tmpLine)
    return 0
                         
def batchWriter(templateFile,xyzDir):
    import glob
    global filename
    xyzPattern=xyzDir + '/*.xyz'
    fileCounter=0
    for xyzFile in glob.iglob(xyzPattern):
        logging.debug('{0}:'.format(xyzFile))
        fileCounter += 1
        writeInputFile(templateFile,xyzFile)
    if fileCounter > 0:
        logging.info("{0} files written in {1}".format(fileCounter,xyzDir)) 
    else:
        logging.info("No *.xyz files found in {1}".format(fileCounter,xyzDir)) 
    return 0    

def initializeLog(debug):
    import sys
    logging.basicConfig(format='%(levelname)s: %(message)s')
    if debug: logLevel = logging.DEBUG
    else: logLevel = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(logLevel)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logLevel)
    logging.debug("Start in debug mode:") 
    return 0

def getArgs():
    import argparse
    import os
    parser = argparse.ArgumentParser(description=
    """
    April 15, 2015
    Murat Keceli
    Generates input files using given xyz files and an input template for electronic structure calculations.
    The geometry is inserted into the place where put_xyz keyword is written.
    """)
    parser.add_argument('tmpFile', metavar='FILE', type=str, help='required nput template file')
    parser.add_argument('xyzPath', metavar='FILE', type=str, nargs='?', default=os.getcwd(), 
                        help='path for the xyz file or directory, default is the current directory')
    parser.add_argument('-d', '--debug', action='store_true', help='print debug information.')
    return parser.parse_args()  
          
def main():
    import os
    args = getArgs()
    initializeLog(args.debug)
    if os.path.isfile(args.tmpFile):
        templateFile = args.tmpFile
        logging.debug("Given template file: {0}".format(templateFile))
        if os.path.isdir(args.xyzPath):
            logging.debug("Given xyz directory: {0}".format(args.xyzPath))
            xyzDir=args.xyzPath
            batchWriter(templateFile, xyzDir)
        elif os.path.isfile(args.xyzPath):
            logging.debug("Given xyz file: {0}".format(args.xyzPath))
            xyzFile=args.xyzPath
            writeInputFile(templateFile, xyzFile)
    else:
        logging.error("First input should be an input template file")
    return 0

if __name__ == "__main__":
    main()