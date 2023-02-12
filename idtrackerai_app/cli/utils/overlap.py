import argparse
import os.path
import itertools

import numpy as np
import pandas as pd
import joblib

from idtrackerai.list_of_blobs.overlap import compute_overlapping_between_two_subsequent_frames
from idtrackerai.list_of_blobs import ListOfBlobs

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", type=str, help="path to metadata.yaml")
    ap.add_argument("--chunks", type=int, nargs="+")
    ap.add_argument("--n-jobs", type=int, default=1)
    return ap


def process_chunk(store_path, chunk):

    video_object_path = os.path.join(os.path.dirname(store_path), "idtrackerai", f"session_{str(chunk).zfill(6)}", "video_object.npy")
    list_of_blobs_path = os.path.join(os.path.dirname(store_path), "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")
    list_of_blobs_path_next = os.path.join(os.path.dirname(store_path), "idtrackerai", f"session_{str(chunk+1).zfill(6)}", "preprocessing", "blobs_collection.npy")


    pattern=[]
    if os.path.exists(list_of_blobs_path) and os.path.exists(list_of_blobs_path_next):
        video=np.load(video_object_path, allow_pickle=True).item()
        frame_number = video.episodes_start_end[-1][-1]
    

        list_of_blobs=ListOfBlobs.load(list_of_blobs_path)
        list_of_blobs_next=ListOfBlobs.load(list_of_blobs_path_next)

        frame_before=list_of_blobs.blobs_in_video[frame_number-1]
        frame_after=list_of_blobs_next.blobs_in_video[frame_number]

        overlap_pattern=compute_overlapping_between_two_subsequent_frames(frame_before, frame_after, queue=None, do=False)
        print(overlap_pattern)
        for ((fn, i), (fnp1, j)) in overlap_pattern:
            blob_before=frame_before[i]
            blob_after=frame_after[j]
            pattern.append((chunk, blob_before.in_frame_index, blob_after.in_frame_index, blob_before.identity, blob_after.identity))
        
    else:
        print(f"Cannot compute overlap between chunks {chunk} and {chunk+1} for experiment {store_path}")
    return pattern


def main():

    ap = get_parser()
    args = ap.parse_args()

    if args.chunks is None:
        raise NotImplementedError
    else:
        chunks = args.chunks

    process_all_chunks(args.store_path, chunks, n_jobs=args.n_jobs)


def process_all_chunks(store_path, chunks, n_jobs=1):
    
    overlap_pattern = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(process_chunk)(
        store_path, chunk 
    )
        for chunk in chunks
    )

    records=itertools.chain(*overlap_pattern)
    data=pd.DataFrame.from_records(records)
    data.columns=["chunk", "in_frame_index_before", "in_frame_index_after", "local_identity", "local_identity_after"]

    data=data.loc[~np.isnan(data["local_identity"])]

    data["identity"]=0

    for chunk in data["chunk"]:
        current_chunk=data.loc[data["chunk"] == chunk]
        if chunk == 50:
            data[current_chunk.index, "identity"]=data[current_chunk.index, "local_identity"]
        
        else:
            for local_identity in current_chunk["local_identity"]:
                identity=data.loc[(data["chunk"] == chunk-1) & (data["local_identity_after"] == local_identity)]["identity"]

                data[
                    data.loc[(data["chunk"] == chunk) & (data["local_identity"] == local_identity)].index,
                    "identity"
                ] = identity



    basedir = os.path.dirname(store_path)
    csv_file=os.path.join(basedir, "idtrackerai", "concatenation-overlap.csv")
    data.to_csv(csv_file)
