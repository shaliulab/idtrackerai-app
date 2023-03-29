import argparse
import os.path
import joblib
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai_app.cli.utils.blobs import get_blobs_collection


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--chunks", type=int, nargs="+", required=True)
    return ap

def main():

    ap = get_parser()

    args = ap.parse_args()
    output = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            verify_list_of_blobs_identity
        )(
            experiment=args.experiment, chunk=chunk
        ) for chunk in args.chunks
    )

    for i, chunk in enumerate(chunks):
        print(chunk, output[i])

def verify_list_of_blobs_identity(experiment, chunk):

    blobs_collection_file = get_blobs_collection(os.environ["FLYHOSTEL_VIDEOS"], experiment, chunk)
    list_of_blobs = ListOfBlobs.load(blobs_collection_file)
    missing_identification=[]

    for blobs_in_frame in list_of_blobs.blobs_in_video:
        identities=[]
        for blob in blobs_in_frame:
            identities.append(blob.final_identities[0])

        if all([identity is None for identity in identities]):
            frame_number = blob.frame_number
            missing_identification.append(frame_number)

    return missing_identification

