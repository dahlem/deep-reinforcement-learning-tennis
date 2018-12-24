import pandas as pd
import numpy as np

import logging

logger = logging.getLogger(__name__)


class Seeds(object):
    def __init__(self, path):
        self.seeds = pd.read_csv(path, header=None, names=['seed'], index_col=None, dtype=np.int32)
        self.idx = 0
        
    def next(self):
        self.idx = self.idx + 1
        seed = self.seeds.seed.iloc[self.idx-1]
        logger.debug('Next random number seed: %d', seed)
        return seed
