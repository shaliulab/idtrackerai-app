import argparse
import os.path
import itertools
import warnings
import re
import numpy as np
import pandas as pd
import joblib
from tqdm.auto import tqdm

from idtrackerai.list_of_blobs.overlap import compute_overlapping_between_two_subsequent_frames
from idtrackerai.list_of_blobs import ListOfBlobs

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", type=str, help="path to metaidentity_table.yaml")
    ap.add_argument("--chunks", type=int, nargs="+")
    ap.add_argument("--n-jobs", type=int, default=1)
    return ap


def process_chunk(store_path, chunk):

    if chunk % 10 == 0:
        print(f"Processing chunk {chunk}")

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

        for ((fn, i), (fnp1, j)) in overlap_pattern:
            blob_before=frame_before[i]
            blob_after=frame_after[j]
            identity_before=blob_before.identity
            if identity_before is None:
                identity_before=0
            identity_after=blob_after.identity
            if identity_after is None:
                identity_after=0

            pattern.append((
                chunk,
                blob_before.in_frame_index, blob_after.in_frame_index,
                identity_before, identity_after
            ))

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


def process_all_chunks(store_path, chunks, n_jobs=1, ref_chunk=50):

    basedir = os.path.dirname(store_path)
    temp_csv_file=os.path.join(basedir, "idtrackerai", "temp_concatenation-overlap.csv")
    csv_file=os.path.join(basedir, "idtrackerai", "concatenation-overlap.csv")


    overlap_pattern = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(process_chunk)(
        store_path, chunk
    )
        for chunk in chunks
    )

    records=itertools.chain(*overlap_pattern)
    identity_table=pd.DataFrame.from_records(records)
    identity_table.columns=["chunk", "in_frame_index_before", "in_frame_index_after", "local_identity", "local_identity_after"]
    identity_table.to_csv(temp_csv_file)
    number_of_animals=int(re.search("FlyHostel[0-9]/([0-9]*)X/", store_path).group(1))


    identity_table=identity_table.loc[pd.notna(identity_table["local_identity"])]
    assert identity_table.shape[0] > 0, f"Corruped identity table (see {temp_csv_file})"
    identity_table["identity"]=0

    chunks=sorted(list(set(identity_table["chunk"].values.tolist())))
    identity_table = propagate_identities(
        identity_table,
        chunks,
        ref_chunk=ref_chunk,
        number_of_animals=number_of_animals
    )
    identity_table.to_csv(csv_file)


def propagate_identities(identity_table, chunks, ref_chunk=50, number_of_animals=None):
    """
    Propagate the identity of the blobs in the reference chunk
    to the overlapping blobs in future chunks of the recording

    This ensures identity is consistent not only within chunk, but also across chunks
    This step is required because idtrackerai analyzes the chunks independently,
    and so the identities are not guaranteed to be consistent across chunks

    We follow a recursive programming solution, where we move through the chunks
    always looking at the identity assignments in the previous chunk.
    We start at the reference chunk i.e. the ref_chunk identities will be propagated through the dataset

    If a chunk-specific identity of 0 (local_identity) is detected, it is ignored

    Args:
        identity_table (pd.DataFrame): Table with columns chunk, local_identity and local_identity_after
            chunk is an integer marking the "left" or "before" chunk being concatenated
            local_identity is a column with integers marking the identity of each blob in the "left" chunk
            local_identity_after is a column with integers marking the identity of each blob in the "right" or "after" chunk

            There should be a one-to-one match both ways but this is not checked but the function

        chunks (list): Chunks to propagate the identities through. All chunks because the ref_chunk are ignored
        ref_chunk (int): Chunk to use as reference
        number_of_animals (int): Optional, if passed and different from 1,
           the function emits a warning when an animal with an assigned identity of 0
           is detected (which means it does not have a cross-chunk identity)

    Returns:
        identity_table (pd.DataFrame): Same table as before with new column "identity" which contains a consistent identity across chunks
    """
    if ref_chunk not in chunks:
        ref_chunk=chunks[0]

    ignored_chunks = chunks[:chunks.index(ref_chunk)]

    if len(ignored_chunks) != 0:
        warnings.warn(f"Ignoring chunks {ignored_chunks}")

    chunks = chunks[chunks.index(ref_chunk):]

    for chunk in tqdm(chunks, desc="Propagating identities", unit="chunk"):
        current_chunk=identity_table.loc[identity_table["chunk"] == chunk]
        if chunk == ref_chunk:
            identity_table.loc[current_chunk.index, "identity"]=identity_table.loc[current_chunk.index, "local_identity"]

        else:
            for local_identity in current_chunk["local_identity"]:
                if local_identity == 0:
                    if number_of_animals > 1:
                        warnings.warn(f"Missing identification in chunk {chunk}")
                    # whatever the number of animals,
                    # the temporary identity in the data frame (0)
                    # is left unchanged
                    # to represent that a cross-chunk identity cannot be assigned
                    # to this local identity in this chunk
                    continue
                # get the identity of the blob in the previous chunk that overlapped with a blob in this chunk with the current identity
                # i.e. the current blob
                identity=identity_table.loc[(identity_table["chunk"] == chunk-1) & (identity_table["local_identity_after"] == local_identity), "identity"]
                if len(identity) == 0:
                    identity = 0
                else:
                    identity=identity.item()


                # assign to the current blob that identity
                identity_table.loc[identity_table.loc[(identity_table["chunk"] == chunk) & (identity_table["local_identity"] == local_identity)].index, "identity"] = identity


    return identity_table
