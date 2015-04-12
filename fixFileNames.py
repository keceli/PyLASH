#!/usr/bin/env python
import logging

def fixFileName(fileName):
    import os
    charsToBeRemoved='`~!@#$%^&*()[]{}|:;,<>? '
    newFileName=fileName.translate(None, ''.join(charsToBeRemoved))
    if newFileName.startswith('_'):
        newFileName=newFileName[1:]
    while '__' in newFileName:
        newFileName=newFileName.replace('__','_')
    if not fileName==newFileName:  
            logging.info('{0} --> {1}'.format(fileName,newFileName))
            userInput=raw_input('Rename the file? (press enter for yes, enter anything else to cancel): ')
            if userInput=='':
                try:  
                    os.rename(fileName, newFileName)
                except:
                    logging.info('Problem in renaming')
            else:
                logging.info('Nothing changed')
    return 0
  
def fixDirectory():
    import glob
    global filename
    for theFile in glob.iglob("*"):
        logging.debug('{0}:'.format(theFile))
        fixFileName(theFile)
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
    return 0

def getArgs():
    import argparse
    parser = argparse.ArgumentParser(description=
    """
    April 12, 2015
    Murat Keceli
    This code renames files with special characters, blanks, etc.
    A file name can be given as an argument, or all files in the current directory is scanned.
    It asks user before renaming.
    """)
    parser.add_argument('input', metavar='FILE', type=str, nargs='?', help='info')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')
    return parser.parse_args()  
          
def main():
    args = getArgs()
    initializeLog(args.debug)
    if args.input is not None:
        inputFile = args.input
        logging.debug('Given input file:{0}'.format(inputFile))
        fixFileName(inputFile)
    else:
        fixDirectory()
    return 0

if __name__ == "__main__":
    main()
