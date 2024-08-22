import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .model import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

def create_model_val(opt):
    from model.Diff_RDA.model.model import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
