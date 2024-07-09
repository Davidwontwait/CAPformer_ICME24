import logging
logger = logging.getLogger('base')


def create_model(opt,mode=1):
    if mode == 0 :
        from .stage0_model import enhancement_model as M
    elif mode == 1:
        from .enhancement_model import enhancement_model as M
    else:
        raise  NotImplementedError(f'Wrong Mode :{mode}')
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
