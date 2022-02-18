# encoding:utf-8


import logging

def logger(file_name):
    # create logger
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    info_form = '[%(asctime)s]-[%(name)s:%(lineno)d]-[%(levelname)s] - %(message)s'
    formatter = logging.Formatter(info_form)
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger