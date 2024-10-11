"""Detailed logger"""
import sys
import logging
from utils.util import boolean_string


LOG = logging.getLogger(__name__)


def cli(parser):
    group = parser.add_argument_group('logging')
    group.add_argument('--logging-output', default=None, type=str,
                       help='path to write the log')
    group.add_argument('--logging-stdout', default=False, type=boolean_string,
                       help='print the detailed log at stdout stream')
    group.add_argument('--logging-write', default=True, type=boolean_string,
                       help='write the detailed log into log file')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    group.add_argument('-q', '--quiet', default=False, action='store_true',
                       help='only show warning messages or above')
    group.add_argument('--shut-data-logging', default=True, type=boolean_string,
                       help='shut up the logging info during data preparing')


def configure(args):
    # refer to https://blog.csdn.net/xiaojiajia007/article/details/104749529
    from pythonjsonlogger import jsonlogger
    # logging信息打印输出顺序就是程序的执行顺序。
    file_handler = logging.FileHandler(args.logging_output + '.log', mode='a')
    file_handler.setFormatter(
        jsonlogger.JsonFormatter('(message) (levelname) (name) (asctime)'))
    stdout_handler = logging.StreamHandler(sys.stdout)

    # logging.warning('Who am I?')  # 这样调用其实是在root logger位置输出
    log_level = logging.INFO  # default is WARING
    if args.debug:
        log_level = logging.DEBUG
    if args.quiet:
        log_level = logging.WARNING

    # if no name is given, then name will be root;
    # if we give the name such as myfolder,
    # then other loggers under myfolder is controlled by this current log_level
    logger = logging.getLogger('')  # control level at the top root
    logger.setLevel(log_level)

    if args.logging_stdout:
        logger.addHandler(stdout_handler)  # bound handler to logger
    if args.logging_write:
        logger.addHandler(file_handler)

    if args.shut_data_logging:
        logging.getLogger("data").setLevel(logging.WARNING)
        logging.getLogger("transforms").setLevel(logging.WARNING)
        logging.getLogger("encoder").setLevel(logging.WARNING)

    LOG.info({
        'type': 'process',
        'local_rank': args.local_rank,
        'argv': sys.argv,  # return a list
        'args': vars(args),  # return a dict
    })
    return log_level
