import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-path', required=True, type=str)

    args = parser.parse_args()

    # if not os.path.exists(args.out_path):
    #     os.mkdir(args.out_path)
    
    return args
