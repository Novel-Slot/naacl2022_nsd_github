import logging
import sys
def get_train_logger(log_path = 'log.txt'):
    '''
    %(asctime)s 日期和时间
    %(threadName)s 线程名称
    %(levelname)s 日志记录级别的文本名称
    %(message)s 记录的消息
    '''
    # logger = logging.getLogger('time-{}'.format(__name__))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #控制台
    handler_stream = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler_stream)

    #日志文件
    handler_file = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(message)s')
    # formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    return logger
logger = get_train_logger()