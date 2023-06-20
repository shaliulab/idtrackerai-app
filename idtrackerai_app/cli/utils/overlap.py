import argparse
import os.path
import itertools
import warnings
import logging

import numpy as np
import pandas as pd
import joblib
from tqdm.auto import tqdm

from idtrackerai.list_of_blobs.overlap import compute_overlapping_between_two_subsequent_frames
from idtrackerai.list_of_blobs import ListOfBlobs
from imgstore.stores.utils.mixins.extract import _extract_store_metadata

logger=logging.getLogger(__name__)


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", type=str, help="path to metaidentity_table.yaml")
    ap.add_argument("--chunks", type=int, nargs="+")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--strict", action="store_true", default=True, help=
    """Whether strict identity propagation policy should be used or not.
    If True, an identity is propagated to the next chunk if and only if a single blob in the last frame overlaps with a blob in the first frame of the next chunk
    and both have a non-zero final identity
    """)
    ap.add_argument("--not-strict", dest="strict", action="store_false", default=True)
    return ap


def process_chunk(store_path, chunk):

    if chunk % 10 == 0:
        print(f"Processing chunk {chunk}")

    input_step = "tracking"

    video_object_path = os.path.join(os.path.dirname(store_path), "idtrackerai", f"session_{str(chunk).zfill(6)}", "video_object.npy")
    list_of_blobs_path = os.path.join(os.path.dirname(store_path), "idtrackerai", f"session_{str(chunk).zfill(6)}", input_step, "blobs_collection.npy")
    if not os.path.exists(list_of_blobs_path):
        list_of_blobs_path = os.path.join(os.path.dirname(store_path), "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")
    
    list_of_blobs_path_next = os.path.join(os.path.dirname(store_path), "idtrackerai", f"session_{str(chunk+1).zfill(6)}", input_step, "blobs_collection.npy")
    if not os.path.exists(list_of_blobs_path_next):
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
        identities = []

        for ((fn, i), (fnp1, j)) in overlap_pattern:
            blob_before=frame_before[i]
            blob_after=frame_after[j]
            ai_identity=blob_before.identity
            identity=blob_before.final_identities[-1]
            if identity is None:
                identity=0
            identity_after=blob_after.final_identities[-1]
            ai_identity_after=blob_after.identity
            if identity_after is None:
                identity_after=0

            pattern.append((
                chunk,
                blob_before.in_frame_index,
                blob_after.in_frame_index,
                ai_identity, ai_identity_after,
                identity, identity_after
            ))
            identities.append(identity)

        identities, counts = np.unique(identities, return_counts=True)
        if not all(counts == 1):
            warnings.warn(f"Identities repeated in chunk {chunk}. Identities {identities}, counts {counts}")

    else:
        print(f"Cannot compute overlap between chunks {chunk} and {chunk+1} for experiment {store_path}")

    print(chunk, pattern)
    return pattern


def main():

    ap = get_parser()
    args = ap.parse_args()

    if args.chunks is None:
        raise NotImplementedError
    else:
        chunks = args.chunks

    print(f"Strict policy: {args.strict}")

    process_all_chunks(args.store_path, chunks, ref_chunk=sorted(chunks)[0], n_jobs=args.n_jobs, strict=args.strict)



def compute_identity_table(store_path, chunks, n_jobs=1):
    basedir = os.path.dirname(store_path)
    temp_csv_file=os.path.join(basedir, "idtrackerai", "temp_concatenation-overlap.csv")


    overlap_pattern = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(process_chunk)(
        store_path, chunk
    )
        for chunk in chunks
    )

    records=itertools.chain(*overlap_pattern)
    identity_table=pd.DataFrame.from_records(records)
    identity_table.columns=["chunk", "in_frame_index_before", "in_frame_index_after", "ai_identity","ai_identity_after", "local_identity", "local_identity_after"]
    identity_table.to_csv(temp_csv_file)


    strict=_extract_store_metadata(store_path).get("strict_identity", True)

    identity_table=identity_table.loc[pd.notna(identity_table["local_identity"])]
    assert identity_table.shape[0] > 0, f"Corrupted identity table (see {temp_csv_file})"
    identity_table["identity"]=0
    identity_table["is_inferred"]=False
    identity_table["is_broken"]=False

    return identity_table


def process_all_chunks(store_path, chunks, n_jobs=1, ref_chunk=50, strict=True):
    basedir = os.path.dirname(store_path)
    idtrackerai_folder = os.path.join(basedir, "idtrackerai")
    csv_file=os.path.join(idtrackerai_folder, "concatenation-overlap.csv")
    vo_path=os.path.join(idtrackerai_folder, f"session_{str(chunks[0]).zfill(6)}", "video_object.npy")
    assert os.path.exists(vo_path), f"{vo_path} not found"
    video_object=np.load(vo_path, allow_pickle=True).item()
    number_of_animals=video_object.user_defined_parameters["number_of_animals"]
    identity_table=compute_identity_table(store_path, chunks, n_jobs=n_jobs)

    chunks=sorted(list(set(identity_table["chunk"].values.tolist())))
    identity_table = propagate_identities(
        identity_table,
        chunks,
        ref_chunk=ref_chunk,
        number_of_animals=number_of_animals,
        strict=strict
    )
    identity_table.to_csv(csv_file)



def get_identity_of_overlapping_blob_in_previous_chunk(identity_table, chunk, local_identity, strict=True):
    """
    Returns the cross-chunk identity of the blob in the previous chunk that overlaps with a blob in the present chunk
    If a past blob that overlaps with the passed local_identity is not found, and strict is False, the function assumes
    there must be one that overlaps with a blob with local identity 0 (which later becomes the passed local_identity),
    and so, uses the identity transferred via the misidentified blob

    If more than 1 blob with local identity 0 is found in the present chunk, strict has no effect, which means the returned identity will be 0

    is_inferred: Whether at least one of the local identities of this track in any past chunk was 0. A human intervention is needed at some point in the past,
    but not necessarily here.
    is_broken: Whether an identity in the immediately past chunk can be matched (is_broken=False) or not (is_broken=True).
    This is where human intervention is needed.
    """
    assert local_identity != 0

    # the cross-chunk identity of this blob is:
    #    1) that of the blob in the previous chunk
    #    2) which overlaps with a blob in this chunk with the local identity of this chunk i.e. this blob
    # = that of the blob in the previous chunk which overlaps with this chunk

    identity=identity_table.loc[(identity_table["chunk"] == chunk-1) & (identity_table["local_identity_after"] == local_identity), "identity"]

    if len(identity) == 0:
        is_broken=True
        if strict:
            identity = 0
            is_inferred=False

        else:
            identity = identity_table.loc[(identity_table["chunk"] == chunk-1) & (identity_table["local_identity_after"] == 0), "identity"]
            is_inferred=True
            if len(identity) == 1:
                identity = identity.item()
                is_inferred=True
            else:
                identity = 0
                is_inferred=False

    elif len(identity) > 1:
        is_broken=True
        identity=0
        is_inferred=False

    else:
        is_broken=False
        identity=identity.item()
        is_inferred=identity_table.loc[(identity_table["chunk"] == chunk-1) & (identity_table["local_identity_after"] == local_identity), "is_inferred"].item()

    return identity, is_inferred, is_broken

def propagate_identities(identity_table, chunks, ref_chunk=50, number_of_animals=None, strict=True):
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

        strict (bool): If True, the identity is propagated only if the overlapping blobs in both chunks have an assigned identity,
        if False, it is inferred from the missing identity in the next chunk

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
        logger.debug(f"Propagating identities for chunk {chunk}")
        current_chunk=identity_table.loc[identity_table["chunk"] == chunk]

        if chunk == ref_chunk:
            identity_table.loc[current_chunk.index, "identity"]=identity_table.loc[current_chunk.index, "local_identity"]

        else:
            for i in range(current_chunk.shape[0]):

                local_identity = current_chunk.iloc[i]["local_identity"]
                indexer = identity_table.loc[identity_table["chunk"] == chunk].iloc[i].name

                if local_identity == 0:
                    if number_of_animals > 1:
                        warnings.warn(f"Missing identification in chunk {chunk}")
                        is_broken=True
                        identity=None
                    else:
                        identity = 0
                    # whatever the number of animals,
                    # the temporary identity in the data frame (0)
                    # is left unchanged
                    # to represent that a cross-chunk identity cannot be assigned
                    # to this local identity in this chunk
                else:
                    # get the identity of the blob in the previous chunk that overlapped with a blob in this chunk with the current identity
                    # i.e. the current blob
                    identity, is_inferred, is_broken=get_identity_of_overlapping_blob_in_previous_chunk(identity_table, chunk, local_identity, strict=strict)                  
                    # assign to the current blob that identity
                    identity_table.loc[indexer, "identity"] = identity
                    identity_table.loc[indexer, "is_inferred"] = is_inferred
                    identity_table.loc[indexer, "is_broken"] = is_broken


            if not strict and chunk != chunks[-1]:
                current_chunk = identity_table.loc[(identity_table["chunk"] == chunk),]
                found_lids, target_lids, missing_lids = feature_stats(identity_table, chunk, "local_identity", number_of_animals)
                found_ids, target_ids, missing_ids = feature_stats(identity_table, chunk, "identity", number_of_animals)
                found_lidas, target_lidas, missing_lidas = feature_stats(identity_table, chunk-1, "local_identity_after", number_of_animals)
                if current_chunk.shape[0] == number_of_animals and len(missing_lids)>0:
                    for i, (missing_local_identity, missing_identity) in enumerate(zip(missing_lids, missing_ids)):
                        truth_table=current_chunk["local_identity"].isin([None, 0])
                        truth_table=truth_table[truth_table]
                        indexer=truth_table.index[0]
                        current_chunk.loc[indexer, "is_inferred"]=True
                        current_chunk.loc[indexer, "is_broken"]=True
                        current_chunk.loc[indexer, "identity"]=missing_identity
                        current_chunk.loc[indexer, "local_identity"]=missing_local_identity
                    
                    identity_table=identity_table.loc[identity_table["chunk"] != chunk,]
                    identity_table=pd.concat([identity_table, current_chunk])
                
                elif current_chunk.shape[0] == number_of_animals and len(missing_lidas)>0:
                    for i, (missing_local_identity_after, missing_identity) in enumerate(zip(missing_lidas, missing_ids)):
                        truth_table=current_chunk["identity"].isin([None, 0])
                        truth_table=truth_table[truth_table]
                        indexer=truth_table.index[0]
                        current_chunk.loc[indexer, "is_inferred"]=True
                        current_chunk.loc[indexer, "is_broken"]=True
                        current_chunk.loc[indexer, "identity"]=missing_identity

                    identity_table=identity_table.loc[identity_table["chunk"] != chunk,]
                    identity_table=pd.concat([identity_table, current_chunk])

                else:

                    warnings.warn("Missing ids: {missing_ids} in chunk {chunk}")
                    for missing_local_identity, missing_identity, missing_local_identity_after in zip(missing_lids, missing_ids, missing_lidas):
                        template = identity_table.iloc[0].copy()
                        template["chunk"]=chunk
                        template["local_identity"]=missing_local_identity
                        template["local_identity_after"]=missing_local_identity_after
                        template["identity"]=missing_identity
                        template["is_inferred"]=True
                        template["is_broken"]=True
                        identity_table=pd.concat([
                            identity_table[identity_table.columns],
                            pd.DataFrame(template).T[identity_table.columns]
                        ], axis=0)

        identity_table.reset_index(inplace=True)
        identity_table.drop("index", inplace=True, axis=1)
        identity_table=identity_table.sort_values(["chunk", "local_identity"],)

    return identity_table


def feature_stats(identity_table, chunk, feature, number_of_animals):
    current_chunk=identity_table.loc[identity_table["chunk"] == chunk]
    found = set(current_chunk[feature].values.tolist())
    target = set(list(range(1, number_of_animals+1)))
    missing = list(target.difference(found))
    return found, target, missing
