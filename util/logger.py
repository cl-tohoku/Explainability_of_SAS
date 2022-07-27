import re
import logging


HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
WHITE = '\033[37m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
BLUE = '\033[34m'
CYAN = '\033[36m'
RED = '\033[31m'
MAGENTA = '\033[35m'
BLACK = '\033[30m'
BHEADER = BOLD + '\033[95m'
BOKBLUE = BOLD + '\033[94m'
BOKGREEN = BOLD + '\033[92m'
BWARNING = BOLD + '\033[93m'
BFAIL = BOLD + '\033[91m'
BUNDERLINE = BOLD + '\033[4m'
BWHITE = BOLD + '\033[37m'
BYELLOW = BOLD + '\033[33m'
BGREEN = BOLD + '\033[32m'
BBLUE = BOLD + '\033[34m'
BCYAN = BOLD + '\033[36m'
BRED = BOLD + '\033[31m'
BMAGENTA = BOLD + '\033[35m'
BBLACK = BOLD + '\033[30m'


COLORS = {
    'WARNING': YELLOW,
    'INFO': OKGREEN,
    'DEBUG': MAGENTA,
    'CRITICAL': HEADER,
    'ERROR': RED
}

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLORS[levelname] + \
                '[' + levelname + ']' + RESET_SEQ
            record.levelname = levelname_color
        else:
            levelname_color = '[' + levelname + ']'
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


def set_logger(out_dir=None, dof="_", debug=False):
    console_format = '%(levelname)s' + ' (%(name)s) %(message)s'

    logger = logging.getLogger()
    if debug == False:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    if debug == False:
        console.setLevel(logging.INFO)
    else:
        console.setLevel(logging.DEBUG)
    console.setFormatter(ColoredFormatter(console_format))
    logger.addHandler(console)
    if out_dir:
        console_format = '%(levelname)s' + ' (%(name)s) %(message)s'
        log_file = logging.FileHandler(out_dir + dof + 'log.txt', mode='w')
        log_file.setLevel(logging.DEBUG)
        log_file.setFormatter(ColoredFormatter(
            console_format, use_color=False))
        logger.addHandler(log_file)
