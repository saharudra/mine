import numpy as np
import yaml
import os
import errno

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def info_dict(key, value, dict):
    """
    Create an info_dict, and use the info_dict utility to keep track of 
    anything that has to be passed to the logger in every epoch. Losses created on the fly will be automatically added to the
    dict. Just don't go too overboard with this.
    """
    if key in dict:
        dict[key] += value
    else:
        dict[key] = value
    return dict