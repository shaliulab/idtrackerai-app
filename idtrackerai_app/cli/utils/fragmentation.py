import argparse
import os.path
import numpy as np
import ipdb
from idtrackerai.fragmentation.fragmentation import show_fragment_structure

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--seconds", type=float, required=False, default=1.0, help="Minimal duration of a significant fragment (seconds)")
    return ap


def main():
    ap = get_parser()
    args = ap.parse_args()
    chunk = args.chunk
    seconds = args.seconds
    
    show_fragment_structure(chunk, seconds)
    

def load_list_of_blobs():
    ap = get_parser()
    args = ap.parse_args()
    chunk = args.chunk
    blobs_collection_file = os.path.join(f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")
    assert os.path.exists(blobs_collection_file), f"{blobs_collection_file} not found"
    list_of_blobs = np.load(blobs_collection_file, allow_pickle=True).item()
    ipdb.set_trace()




if __name__ == "__main__":
    main()
