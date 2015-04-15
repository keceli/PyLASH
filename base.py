#!/usr/bin/env python
import logging

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
    parser = argparse.ArgumentParser(description=
    """
    April 12, 2015
    Murat Keceli
    This is only a template.
    """)
    parser.add_argument('input', metavar='FILE', type=str, nargs='?', help='info')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')
    parser.add_argument("-f", dest="filename", required=True, help="required input file", metavar="FILE")
    return parser.parse_args()  
          
def main():
    args = getArgs()
    initializeLog(args.debug)
    if args.input is not None:
        inputFile = args.input
        logging.debug('Given input file:{0}'.format(inputFile))

    return 0

if __name__ == "__main__":
    main()