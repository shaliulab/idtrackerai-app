"""
Save specific frames of an imgstore as .png

Reads every .txt file in the input folder with a name matching chunk_*.txt
(where chunk is padded with 6 zeroes).
Every line in the file should be a frame number beloging to the corresponding chunk
"""

import argparse
import os.path
import logging
import glob

from tqdm import tqdm
import joblib
import cv2
from .utils import validate_store


logger = logging.getLogger(__name__)


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", help="path to imgstore metadata.yaml")
    ap.add_argument("--input", required=True, help="folder with frame numbers per chunk to be saved")
    ap.add_argument("--output", default=None, help="output folder to save .png files to")
    ap.add_argument("--chunks", type=int, nargs="+", required=False, default=None)
    ap.add_argument("--n-jobs",type=int, required=False, default=1)
    return ap


def read_frame_numbers(path):

    with open(path, "r") as filehandle:
        fns = filehandle.readlines()
        fns = [int(fn.strip()) for fn in fns]

    return fns


def main():

    ap = get_parser()
    args = ap.parse_args()

    store = validate_store(args.store_path)


    assert os.path.exists(args.input)

    if args.chunks is not None:
        chunks = args.chunks
    else:
        chunks = list(store._index._chunks)

    if args.output is not None:
        counter = 1
        output = os.path.realpath(args.output)
        while True:
            if os.path.exists(output):
                output = os.path.realpath(args.output)+str(counter)
                counter += 1
            else:
               os.makedirs(output)
               break


    joblib.Parallel(args.n_jobs)(joblib.delayed(process_chunk)(
        args.store_path, chunk, args.input, output
        )
        for chunk in chunks
    )



def process_chunk(store_path, chunk, input, output):

    store = validate_store(store_path)
    fn_metadata = store._index.get_chunk_metadata(chunk)["frame_number"]

    fn_paths = glob.glob(os.path.join(input, f"{str(chunk).zfill(6)}_*.txt"))
    assert len(fn_paths) == 1
    fn_path = fn_paths[0]

    fns = read_frame_numbers(fn_path)

    f_idxs = [fn_metadata.index(frame_number) for frame_number in fns]

    for i, frame_number in enumerate(tqdm(fns, desc=f"Processing chunk {chunk}")):
        frame, _ = store.get_image(frame_number)
        frame_idx = f_idxs[i]
        key = f"{frame_number}_{chunk}-{frame_idx}" 
        output_path = os.path.join(output, f"{key}.png")
        print(output_path)
        cv2.imwrite(output_path, frame)






