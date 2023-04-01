import argparse
import os.path
import joblib
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai_app.cli.utils.blobs import get_blobs_collection


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", required=True)
    ap.add_argument("--chunks", type=int, nargs="+", required=True)
    ap.add_argument("--n-jobs", type=int, default=1)
    return ap

def main():

    ap = get_parser()

    args = ap.parse_args()
    chunks = args.chunks
    output = joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(
            verify_list_of_blobs_identity
        )(
            store_path=args.store_path, chunk=chunk
        ) for chunk in chunks
    )

    for i, chunk in enumerate(chunks):
        print(chunk, output[i])

def verify_list_of_blobs_identity(store_path, chunk):

    experiment=os.path.dirname(store_path)
    print(experiment)
    blobs_collection_file = get_blobs_collection("/", experiment, chunk)
    if os.path.exists(blobs_collection_file):
        list_of_blobs = ListOfBlobs.load(blobs_collection_file)
        missing_identification=[]

        for blobs_in_frame in list_of_blobs.blobs_in_video:
            identities=[]
            blob=None
            for blob in blobs_in_frame:
                identities.append(blob.final_identities[0])

            if blob is not None and all([identity is None for identity in identities]):
                frame_number = blob.frame_number
                missing_identification.append(frame_number)

    else:
        missing_identification=[None]

    return missing_identification
