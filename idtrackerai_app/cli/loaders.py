import argparse
import os.path

import numpy as np


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    return ap


def load_video(chunk):
    return np.load(os.path.join("idtrackerai", f"session_{str(chunk).zfill(6)}", "video_object.npy"), allow_pickle=True).item()

def main():
    ap = get_parser()
    args = ap.parse_args()
    video = load_video(args.chunk)
    import ipdb; ipdb.set_trace()
