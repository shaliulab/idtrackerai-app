import argparse
import os.path

import pandas as pd

from idtrackerai_app.cli.transfer.transfer import transfer_file

from .utils import get_blobs_collection

FLYHOSTEL_VIDEOS=os.environ["FLYHOSTEL_VIDEOS"]
REMOTE_VIDEOS="/staging/leuven/stg_00115/Data/flyhostel_data/videos"

def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    # ap.add_argument("--chunks", type=int, nargs="+")
    return ap


def main():

    ap=get_parser()
    args = ap.parse_args()


    human_corrections_file = os.path.join(
        os.environ["FLYHOSTEL_VIDEOS"],
        args.experiment,
        "video-annotator",
        "human-corrections.csv"
    )
    if os.path.exists(human_corrections_file):

        corrections = pd.read_csv(human_corrections_file)
        chunks = sorted(list(set(corrections["chunk"].tolist())))
        upload_to_vsc(args.experiment, chunks)

    else:
        print(f"{human_corrections_file} not found")


def upload_to_vsc(experiment, chunks):
    
    for chunk in chunks:
        blobs_collection=get_blobs_collection(FLYHOSTEL_VIDEOS, experiment, chunk)
        remote_blobs_collection=get_blobs_collection(REMOTE_VIDEOS, experiment, chunk)
        print(f"{blobs_collection} -> login:{remote_blobs_collection}")
        transfer_file(blobs_collection, "login:" + remote_blobs_collection, update=True)


if __name__ == "__main__":
    main()
