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
    ap.add_argument("--output", default=None, help="Output folder")
    ap.add_argument("--chunks", nargs="+", type=int, default=None)
    return ap


def main():
    """
    Assumes all problematic frames have a label in the yolov7 repo
    """

    ap = get_parser()
    args = ap.parse_args()

    allowed_classes={0: "fly"}

    store = validate_store(args.store_path)

    if args.chunks is None:
         chunks = store._index._chunks
    else:
         chunks = args.chunks
         
         
    assert os.path.exists(args.input)


    output = args.output
    count = 0
    if output is not None:
        while True:
            if os.path.exists(output):
                count+=1
                output = args.output + str(count)
            else:
                os.makedirs(output) 
                break
    
    output = joblib.Parallel(n_jobs=args.n_jobs)(joblib.delayed(process_chunk)(
            args.store_path, chunk, args.input, allowed_classes=allowed_classes, output=output
        )
        for chunk in chunks
    )


def process_chunk(store_path, chunk, input, allowed_classes=None, output=None):

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

    if output is not None:
        with open(os.path.join(output, f"{str(chunk).zfill(6)}_success.txt"), "w") as filehandle:
            filehandle.write(f"Processed/Total,Ratio\n")
            filehandle.write(f"{processed_successfully}/{len(frames)},{success_rate}\n")
        with open(os.path.join(output, f"{str(chunk).zfill(6)}_failure.txt"), "w") as filehandle:
            for frame_number in failed_frames:
                filehandle.write(f"{frame_number}\n")

    return chunk, success_rate

if __name__ == "__main__":
     main()
