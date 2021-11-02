import argparse

##### helper function #####
def str2bool(input_):
    if input_.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input_.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')