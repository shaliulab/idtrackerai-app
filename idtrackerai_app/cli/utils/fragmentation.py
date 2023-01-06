import argparse
import os.path
import numpy as np
import ipdb
from imgstore.stores.utils.mixins.extract import _extract_store_metadata

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--seconds", type=float, required=False, default=1.0, help="Minimal duration of a significant fragment (seconds)")
    return ap


def main():
    ap = get_parser()
    args = ap.parse_args()
    chunk = args.chunks
    seconds = args.seconds
    
    show_fragment_structure(chunk, seconds)
    

def show_fragment_structure(chunk, seconds):
    

    fragment_file = os.path.join(f"session_{str(chunk).zfill(6)}", "preprocessing", "fragments.npy")
    video_file = os.path.join(f"session_{str(chunk).zfill(6)}", "video_object.npy")
    video_object = np.load(video_file, allow_pickle=True).item()
    assert os.path.exists(fragment_file), f"{fragment_file} not found"
    list_of_fragments = np.load(fragment_file, allow_pickle=True).item()
    store_md = os.path.realpath(video_object.video_path)
    assert os.path.exists(store_md), f"Path to metadata.yaml ({store_md}) not found"
    metadata = _extract_store_metadata(store_md)
    framerate = metadata["framerate"]
    structure = []

    for fragment in list_of_fragments.fragments:
        length = fragment.start_end[1] - fragment.start_end[0]
        followed = fragment.start_end[1] !=  video_object.episodes_start_end[-1][-1]
        
        structure.append((fragment.start_end, length, length > (framerate * seconds), fragment.identity, followed))
        for field in structure[-1]:
            print(field, end=" ")
        print("")

    return structure


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
