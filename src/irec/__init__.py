from loguru import logger

try:
    import torch
except ImportError:
    logger.error('torch is not available, please install torch to use irec framework')
else:

    __all__ = [
        'callbacks',
        'data',
        'runners',
        'utils'
    ]
