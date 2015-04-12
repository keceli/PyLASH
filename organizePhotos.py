#!/usr/bin/env python
import logging
import exifread

def getPhotoDate(fileName):
    with open(fileName) as f:
        tags = exifread.process_file(f, details = False, stop_tag = 'DateTimeOriginal')
    try:    
        photoDate=tags['EXIF DateTimeOriginal']
        #photoDate=tags['IMAGE DateTime']
    except:
        photoDate='no_date'   
    return str(photoDate) 

def printEXIF(fileName):  
    with open(fileName) as f:
        tags = exifread.process_file(f, details = False,)
    print tags 

def renamePhoto(fileName,option=1):
    import os
    dateString = getPhotoDate(fileName)
    dirName = dateString.split()[0] + '/'
    if dirName.startswith('20') and option == 3:
        if not os.path.exists(dirName):
            os.makedirs(dirName)
    newFileName = dateString.replace(':', '_').replace(' ', '_') 
    if not fileName.startswith(newFileName):
        newFileName +=  "_" + fileName
        logging.info('{0} --> {1}'.format(fileName, newFileName))
        if option == 2:
            os.rename(fileName, newFileName)
        elif option == 3:
            os.rename(fileName, dirName + newFileName)
                
    return 0   
    
def organizeDirectory(ext,option):
    import glob
    fileType='*'+ext
    counter=0
    for photoFile in glob.iglob(fileType):
        renamePhoto(photoFile,option)
        counter +=1
    logging.info('{0} files scanned with extension {1}'.format(counter,ext))    
    return 0

def organizeDirectoryRecursively(ext,option):
    """Not tested"""
    import os
    counter=0
    for path, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith(ext):
                renamePhoto(os.path.join(path, f),option)
                counter +=1
    logging.info('{0} files scanned with extension {1}'.format(counter,ext))    
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
    April 12, 2015
    Murat Keceli
    This code renames photos based on the date given in EXIF data.
    OLDNAME ---> YYYY_MM_DD_OLDNAME 
    A file name can be given as an argument, or all files in the current directory is scanned.
    User has three options:
    1) Dry run to see files to be renamed
    2) Rename files
    3) Create folders with YY_MM_DD and move renamed files into these folders.
    """)
    parser.add_argument('input', metavar='FILE', type=str, nargs='?', help='info')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')
    return parser.parse_args()  
          
def main():
    args = getArgs()
    initializeLog(args.debug)
    option = raw_input("""First, make sure you have a backup. \n
    Then enter one of three options \n
    1) Dry run to see files to be renamed \n
    2) Rename files \n
    3) Create folders with YY_MM_DD and move renamed files into these folders.\n
    Enter your choice: """)
    extension = raw_input("""Enter the extension of files you want to be scanned\n
    You can simply press enter if you want all files (not recursive) to be scanned\n
    If you want to scan subdirectories you can enter */*.jpg etc.
    Enter the extension: """)
     
    if args.input is not None:
        inputFile = args.input
        logging.debug('Given input file:{0}'.format(inputFile))
        renamePhoto(inputFile, option)
        printEXIF(inputFile)
    else:
        organizeDirectory(extension,option)
        
    return 0

if __name__ == "__main__":
    main()
