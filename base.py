#!/usr/bin/env python
import logging

def initializeLog(debug=False,warning=False,silent=False):
    import sys
    logging.basicConfig(format='%(levelname)s: %(message)s')
    if debug: logLevel = logging.DEBUG
    elif warning: logLevel = logging.WARNING
    elif silent: logLevel = logging.ERROR
    else: logLevel = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(logLevel)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logLevel)
    logging.debug("Debugging messages on...") 
    logging.info("Informational messages on...") 
    logging.warning("Warning messages on ...") 
    logging.error("Error messages on ...") 
    return 0

def getArgs():
    import argparse
    parser = argparse.ArgumentParser(description=
    """
    April 12, 2015
    Murat Keceli
    This is only a template.
    """)
    parser.add_argument('input', metavar='FILE', type=str, nargs='*', help='input arguments')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')
    parser.add_argument('-w', '--warning', action='store_true', help='Print warnings and errors.')
    parser.add_argument('-s', '--silent', action='store_true', help='Print only errors.')
    parser.add_argument("-f", dest="filename", required=False, help=" input file", metavar="FILE")
    return parser.parse_args()  
          
def main():
    args = getArgs()
    initializeLog(debug=args.debug,warning=args.warning,silent=args.silent)
    if len(args.input) > 0 :
        for i in args.input:
            logging.debug('Given input argument:{0}'.format(i))
    else:
        logging.debug('No input arguments given')
    return 0

if __name__ == "__main__":
    main()