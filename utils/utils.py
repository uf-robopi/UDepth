"""
Functions of average meter, logger, guidedfilter, and result output
"""
import logging
from cv2.ximgproc import guidedFilter
import numpy as np


class AverageMeter(object):
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

 
def get_logger(filename, verbosity=1, name=None):
    """Get logger"""
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger


def gf(mask, img):
    """Apply guidedfilter"""
    return guidedFilter(mask, img, 4, 0.2, -1)


def output_result(out, mask):
    """Prepare data and apply guidedfilter to depth map"""
    mask = np.float32(mask)
    img=out.detach().cpu().numpy()
    img_resized=img.reshape(240,320)
    max_item = max(max(row) for row in img_resized)
    img_resized = img_resized / max_item * 255

    result = gf(mask, img_resized)
    
    return result

