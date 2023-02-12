import argparse
import os.path
import itertools

import pandas as pd
import joblib

from idtrackerai.list_of_blobs.overlap import compute_overlapping_between_two_subsequent_frames
from idtrackerai.list_of_blobs import ListOfBlobs

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", type=str, help="path to metadata.yaml")
    ap.add_argument("--chunks", type=int, nargs="+")
    return ap


def process_chunk(store_path, chunk):

    list_of_blobs_path = os.path.join(os.path.dirname(store_path), "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")
    list_of_blobs_path_next = os.path.join(os.path.dirname(store_path), "idtrackerai", f"session_{str(chunk+1).zfill(6)}", "preprocessing", "blobs_collection.npy")


    if os.path.exists(list_of_blobs_path) and os.path.exists(list_of_blobs_path_next):
        list_of_blobs=ListOfBlobs.load(list_of_blobs_path)
        list_of_blobs_next=ListOfBlobs.load(list_of_blobs_path_next)

        overlap_pattern=compute_overlapping_between_two_subsequent_frames(list_of_blobs.blobs_in_video[-1], list_of_blobs_next.blobs_in_video[0], queue=None, do=False)
        pattern=[]
        for ((blob_before, _), (blob_after, _)) in overlap_pattern:
            
            pattern.append((chunk, blob_before.in_frame_index, blob_after.in_frame_index))
        
        return pattern

    else:
        return []


def main():


    ap = get_parser()
    args = ap.parse_args()

    if args.chunks is None:
        raise NotImplementedError
    else:
        chunks = args.chunks
    
    overlap_pattern = joblib.Parallel(n_jobs=args.n_jobs)(joblib.delayed(process_chunk)(
        args.store_path, chunk 
    )
        for chunk in chunks
    )

    records=itertools.chain(*overlap_pattern)
    data=pd.DataFrame.from_records(records)
    data.columns=["chunk", "in_frame_index_before", "in_frame_index_after"]
    
    basedir = os.path.dirname(args.store_path)
    csv_file=os.path.join(basedir, "idtrackerai", "concatenation-overlap.csv")
    data.to_csv(csv_file)