import argparse
import os.path

import pandas as pd

from idtrackerai_app.cli.transfer.transfer import transfer_file

from .utils import get_blobs_collection

FLYHOSTEL_VIDEOS=os.environ["FLYHOSTEL_VIDEOS"]
REMOTE_VIDEOS="/Users/FlySleepLab_Dropbox/Data/flyhostel_data/videos"

def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    # ap.add_argument("--chunks", type=int, nargs="+")
    return ap


def main():
    ap=get_parser()
    args = ap.parse_args()


    concatenation_file = os.path.join(
        os.environ["FLYHOSTEL_VIDEOS"],
        args.experiment,
        "idtrackerai",
        "concatenation-overlap.csv"
    )
    if os.path.exists(concatenation_file):

        concatenation = pd.read_csv(concatenation_file)
        chunks0 = sorted(list(set(concatenation.loc[concatenation["local_identity"] == 0, "chunk"].tolist())))
        
        chunks1= concatenation.loc[concatenation["local_identity_after"] == 0, "chunk"]+1
        chunks1=sorted(list(set(chunks1.tolist())))
        chunks = chunks0 + chunks1
        download_from_vsc(args.experiment, chunks)

    else:
        print(f"{human_corrections_file} not found")


def send_from_vsc(experiment, chunks):
    server = "cv3"
    
    for chunk in chunks:
        blobs_collection=get_blobs_collection(FLYHOSTEL_VIDEOS, experiment, chunk)
        remote_blobs_collection=get_blobs_collection(REMOTE_VIDEOS, experiment, chunk)
        print(f"{blobs_collection} -> {server}:{remote_blobs_collection}")
        transfer_file(blobs_collection, server + ":" + remote_blobs_collection, update=True)


def download_from_vsc(experiment, chunks):
    server = "login"
    
    for chunk in chunks:
        blobs_collection=get_blobs_collection(FLYHOSTEL_VIDEOS, experiment, chunk)
        remote_blobs_collection=get_blobs_collection(REMOTE_VIDEOS, experiment, chunk)
        print(f"{server}{remote_blobs_collection} -> {blobs_collection}")
        transfer_file(server + ":" + remote_blobs_collection, blobs_collection, update=True)



if __name__ == "__main__":
    main()
