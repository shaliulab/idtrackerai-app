import argparse
import os.path
import glob
import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
import cv2
plt.set_cmap("gray")
from scipy.spatial.distance import euclidean
logging.getLogger("idtrackerai.animals_detection.yolov7").setLevel(logging.DEBUG)
logging.getLogger("idtrackerai.list_of_blobs.modifiable").setLevel(logging.DEBUG)

import torch
from idtrackerai.animals_detection.yolov7 import annotate_chunk_with_yolov7, yolov7_repo

from .utils import validate_store


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", required=True)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--input", required=True, help="Folder with labels produced by YOLOv7")
    ap.add_argument("--output", default="idtrackerai", help="Output folder")
    ap.add_argument("--chunks", nargs="+", type=int, default=None)
    return ap


def main():

    ap = get_parser()
    args = ap.parse_args()
    integrate_yolov7(store_path=args.store_path, n_jobs=n_jobs, input=input, output=args.output, chunks=args.chunks)


def integrate_yolov7(store_path, n_jobs, input, output, chunks):
    """
    Assumes all problematic frames have a label in the yolov7 repo
    """

    assert os.path.exists(input), f"{input} not found (wd: {os.getcwd()}"
    assert os.path.exists(output), f"{output} not found"
    assert os.path.isdir(output), f"{output} is not a directory"

    filename = f"session_{str(chunk).zfill(6)}_integration_output.txt"
    logfile = os.path.join(output, filename)
    assert os.path.exists(logfile), f"{logfile} does not exist, did you run the YOLOv7 integration step already?"

    allowed_classes={0: "fly"}
    store = validate_store(store_path)

    count = 0
    original_output = output



   
    Output = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(process_chunk)(
            store_path, chunk, input, allowed_classes=allowed_classes, logfile=logfile
        )
        for chunk in chunks
    )


def process_chunk(store_path, chunk, input, allowed_classes=None, logfile=None):

    regex=os.path.join(input, f"*_{chunk}-*.txt")
    labels = sorted(glob.glob(regex))
    frames = [(
            int(os.path.basename(label).split("_")[0]),
            int(os.path.splitext(os.path.basename(label))[0].split("_")[1].split("-")[1])
        )
        for label in labels
    ]

    logger.debug(f"Processing {len(frames)} for {store_path} chunk {chunk}")
    
    if frames:
        _, processed_successfully, failed_frames = annotate_chunk_with_yolov7(store_path, chunk, frames, input, allowed_classes=allowed_classes, exclusive=False, save=True)
        success_rate=round(processed_successfully/len(frames), 3)
        
    else:
        success_rate= "OK"
        processed_successfully=0
        failed_frames = []

    if logfile is not None:
        with open(logfile, "a") as filehandle:
            filehandle.write(f"Processed frames: {processed_successfully}\n")
            filehandle.write(f"Total frames: {len(frames)}\n")
            filehandle.write(f"Success rate: {success_rate}\n")
            for frame_number in failed_frames:
                filehandle.write(f"Fail: {frame_number}\n")

    return chunk, success_rate

if __name__ == "__main__":
     main()
