import argparse
import os.path
import numpy as np


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    return ap


def main():
    ap = get_parser()
    args = ap.parse_args()
    chunk = args.chunk
    fragment_file = os.path.join(f"session_{str(chunk).zfill(6)}", "preprocessing", "fragments.npy")
    video_file = os.path.join(f"session_{str(chunk).zfill(6)}", "video_object.npy")
    assert os.path.exists(fragment_file), f"{fragment_file} not found"
    list_of_fragments = np.load(fragment_file, allow_pickle=True).item()
    #video_object = np.load(video_file, allow_pickle=True).item()
    framerate = 160
    for fragment in list_of_fragments.fragments:
        length = fragment.start_end[1] - fragment.start_end[0]
        print(fragment.start_end, length, length > framerate, fragment.identity)


def load_list_of_blobs():
    ap = get_parser()
    args = ap.parse_args()
    chunk = args.chunk
    blobs_collection_file = os.path.join(f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")
    assert os.path.exists(blobs_collection_file), f"{blobs_collection_file} not found"
    list_of_blobs = np.load(blobs_collection_file, allow_pickle=True).item()
    import ipdb; ipdb.set_trace()




if __name__ == "__main__":
    main()
