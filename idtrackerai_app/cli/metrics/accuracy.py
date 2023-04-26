import argparse
import glob
import os.path

import numpy as np
import joblib


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", required=True, type=str)
    ap.add_argument("--n-jobs", required=False, default=-2, type=int)
    return ap

def main():

    ap = get_parser()
    args=ap.parse_args()
    list_accuracy(args.store_path, args.n_jobs)


def list_accuracy(store_path, n_jobs):

    idtrackerai_folder = os.path.join(os.path.dirname(store_path), "idtrackerai")
    vos = sorted(glob.glob(
        os.path.join(idtrackerai_folder, "session_*/video_object.npy")
    ))


    output = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
           list_accuracy_single_chunk
        )(
        vo_path
    )  for vo_path in vos
    )

    output = sorted(output, key = lambda x: x[0])
    for chunk, accuracy in output:
        print(f"{chunk},{round(accuracy, 2)}")


def list_accuracy_single_chunk(vo_path):

    vo = np.load(vo_path, allow_pickle=True).item()
    accuracy = getattr(vo, "estimated_accuracy", None)
    chunk = vo._chunk
    return chunk, accuracy


if __name__ == "__main__":
    main()
