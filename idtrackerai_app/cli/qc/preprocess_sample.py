"""
Sample random frames to verify segmentation parameters

Sometimes the config produced by the user might work
in a decent fraction of frames but not high enough (<99%).
This module systematically checks samples-per-chunk frames per chunk
by applying the segmentation criteria and flags frames
where the number of animals is not the expected one
This could only be because in that particular frame
1) there are crossings (only in experiments with groups)
2) an animal is missing

Call this script like so:

preprocess-sample --store-path /flyhostel_data/videos/FlyHostel1/1X/2023-10-31_16-00-00/metadata.yaml

It will output for every chunk between start-chunk and end-chunk, the frames that dont have the expected number of animals
At the end, the total count of such frames is shown
"""

import os.path
import argparse
import json
import numpy as np

from imgstore.interface import VideoCapture
from idtrackerai.utils.utils import load_mask_img
from idtrackerai.animals_detection.segmentation import _process_frame
from idtrackerai.animals_detection.segmentation_utils import apply_segmentation_criteria

def preprocess_sample(store_path, start_chunk, end_chunk, samples_per_chunk):

    last_chunk=start_chunk-1


    cap = VideoCapture(store_path, 1)
    try:
        chunksize=int(cap._metadata["chunksize"])
        basedir=os.path.dirname(store_path)

        config_file = os.path.join(
            basedir,
            os.path.basename(basedir) + ".conf"
        )

        assert os.path.exists(config_file), f"{config_file} not found"
        with open(config_file, "r") as filehandle:
            config = json.load(filehandle)
            number_of_animals=int(float(config["_number_of_animals"]["value"]))

        frame_numbers=np.arange(start_chunk*chunksize, end_chunk*chunksize, chunksize//samples_per_chunk)
        frame_numbers=[int(fn) for fn in frame_numbers]

        total_count=0
        problematic_frames=[]
        for frame_number in frame_numbers:
            chunk = frame_number // chunksize
            if last_chunk != chunk:
                print(f"Chunk {chunk}: {problematic_frames}")
                total_count+=len(problematic_frames)
                problematic_frames.clear()
                last_chunk=chunk

            cap.set(1, frame_number)
            ret, frame = cap.read()
            frame=frame[:,:,0]
            assert ret
            found_animals=count_animals_in_frame(frame, config)
            if found_animals!=number_of_animals:
                problematic_frames.append(frame_number)

        print(f"Total problematic frames = {total_count}")


    finally:
        cap.release()


def count_animals_in_frame(frame, parameters):
    (
        _,
        good_contours_in_full_frame,
    ) = apply_segmentation_criteria(
        frame,
        parameters,
    )

    number_of_animals=len(good_contours_in_full_frame)
    return number_of_animals


def get_parser():

    ap=argparse.ArgumentParser()
    ap.add_argument("--store-path")
    ap.add_argument("--start-chunk", default=10, type=int)
    ap.add_argument("--end-chunk", default=350, type=int)
    ap.add_argument("--samples-per-chunk", default=10, type=int)
    return ap


def main():

    ap = get_parser()
    args=ap.parse_args()
    store_path = args.store_path
    start_chunk=args.start_chunk
    end_chunk=args.end_chunk
    samples_per_chunk=args.samples_per_chunk

    preprocess_sample(store_path, start_chunk, end_chunk, samples_per_chunk)

if __name__ == "__main__":
    main()