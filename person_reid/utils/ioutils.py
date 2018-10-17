"""
io utils functions
Quan Yuan
2018-10-10
"""
import yaml

def parse_config(args, config_file):

    with open(config_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    for k, v in cfg.iteritems():
        if not hasattr(args, k):
            print ('Ignore option: {}'.format(k))
        else:
            setattr(args, k, v)
    return args