import argparse
import os.path
import logging
import itertools

import numpy as np
from tqdm import tqdm
import joblib
import pandas as pd
import h5py
from idtrackerai.list_of_blobs import ListOfBlobs
from imgstore.interface import VideoCapture
from .utils import validate_store

logger = logging.getLogger(__name__)

def detect_incomplete_frames_in_chunk_from_h5(store_path, chunk):

    incomplete_frames=[]
    
    segmentation_data = os.path.join(os.path.dirname(store_path), "idtrackerai", f"session_{str(chunk).zfill(6)}", "segmentation_data")
    h5_files = sorted([file for file in os.listdir(segmentation_data) if file.startswith("episode_pixels")])

    for file in tqdm(h5_files):
        with h5py.File(os.path.join(segmentation_data, file)) as f:
            keys = list(f.keys())
            missing = detect_incomplete_frames_in_episode(keys, number_of_animals)
            if missing:
                incomplete_frames.extend(missing)

    return incomplete_frames



def detect_incomplete_frames_in_episode(keys, number_of_animals):
    frame_numbers = {}
    missing = []
    
    identities = set(range(1, number_of_animals+1))
    
    for key in keys:
        fn, identity = key.split("-")
        fn=int(fn)
        identity = int(identity)
        identity+=1
        
        if fn not in frame_numbers:
            frame_numbers[fn] = []
        
        frame_numbers[fn].append(identity)
    
    for fn in frame_numbers:
        if identities != set(frame_numbers[fn]):
            missing.append(fn)
            
    return missing        

def detect_incomplete_frames_in_chunk_from_blobs_collection(store_path, chunk, output=None):

    raise NotImplementedError
    incomplete_frames=[]

    session_folder = os.path.join(os.path.dirname(store_path), "idtrackerai", f"session_{str(chunk).zfill(6)}")
    # blobs_collection_path = os.path.join(session_folder, "preprocessing", "blobs_collection.npy")
    list_of_blobs = ListOfBlobs.load(blobs_collection_path)
    video_object = np.load(os.path.join(session_folder, "video_object.npy"), allow_pickle=True).item()

    store = VideoCapture(store_path, chunk)
    fns = store._index.get_chunk_metadata(chunk)["frame_number"]
    start, end = fns[0], fns[-1]+1
    incomplete_frames = []
    number_of_animals = video_object._user_defined_parameters["number_of_animals"]
    fn = start

    for blobs_in_frame in list_of_blobs.blobs_in_video[start:end]:
        if len(blobs_in_frame) != number_of_animals:
            incomplete_frames.append(fn)
        fn += 1

    assert fn == end

    if output is not None:
        with open(os.path.join(output, f"{str(chunk).zfill(6)}_incomplete-frames.txt"), "w") as filehandle:
            for fn in incomplete_frames:
                filehandle.write(f"{fn}\n")

    return incomplete_frames



def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", help="path to imgstore metadata.yaml")
    ap.add_argument("--output", default=None, help="output folder to save results to")
    ap.add_argument("--chunks", type=int, nargs="+", required=False, default=None)
    ap.add_argument("--n-jobs",type=int, required=False, default=1)
    return ap



def main():

    ap = get_parser()
    args = ap.parse_args()

    store = validate_store(args.store_path)

    if args.chunks is not None:
        chunks = args.chunks
    else:
        chunks = list(store._index._chunks)

    output = args.output
    if output is not None:
        counter = 1
        while True:
            if os.path.exists(output):
                output = args.output+str(counter)
                counter += 1
            else:
               os.makedirs(output)
               break
   
    incomplete_frames_by_chunks=joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(detect_incomplete_frames_in_chunk_from_blobs_collection)(args.store_path, chunk, output=output)
        for chunk in chunks
    )
    
    incomplete_frames=list(itertools.chain(*incomplete_frames_by_chunks))
    with open(os.path.join(output, "all.txt"), "w") as filehandle:
        for fn in incomplete_frames:
             print(fn)
             filehandle.write(f"{fn}\n")


if __name__ == "__main__":
    main()
