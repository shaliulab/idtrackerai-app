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
The frames are also saved to --output (default: preprocess_sample.csv next to the store), one row per animal found
(x, y empty if none was found), written chunk by chunk so the file can be checked while it runs
"""

import os.path
import argparse
import json
import numpy as np
import pandas as pd

COLUMNS = ["frame_number", "chunk", "video", "frame_idx", "n", "x", "y"]

from imgstore.interface import VideoCapture
from idtrackerai.utils.utils import load_mask_img
from idtrackerai.animals_detection.segmentation import _process_frame
from idtrackerai.animals_detection.segmentation_utils import apply_segmentation_criteria

def preprocess_sample(store_path, start_chunk, end_chunk, samples_per_chunk, output=None):

    last_chunk=start_chunk-1
    store_path=os.path.realpath(store_path)
    if output is None:
        output=os.path.join(os.path.dirname(store_path), "preprocess_sample.csv")
    pd.DataFrame(columns=COLUMNS).to_csv(output, index=False)

    cap = VideoCapture(store_path, 10)
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
                total_count+=visualize_problems(problematic_frames, output)
                # print(f"Chunk {chunk-1}: {problematic_frames}")
                problematic_frames.clear()
                last_chunk=chunk

            cap.set(1, frame_number)
            ret, frame = cap.read()
            frame=frame[:,:,0]
            assert ret

            (
                _,
                contours,
            ) = apply_segmentation_criteria(
                frame,
                config,
            )
            found_animals=len(contours)
            # found_animals=count_animals_in_frame(frame, config)

            if found_animals!=number_of_animals:
                # import ipdb; ipdb.set_trace()
                centroids=[contour.mean(axis=0).flatten() for contour in contours]
                if not centroids:
                    centroids=[(np.nan, np.nan)]
                for centroid in centroids:
                    problematic_frames.append(
                        (
                            frame_number, frame_number//chunksize,
                            str(frame_number//chunksize).zfill(6) + ".mp4", 
                            frame_number%chunksize, found_animals,
                            *centroid
                        )
                    )

        total_count+=visualize_problems(problematic_frames, output)
        print(f"Total problematic frames = {total_count}")
        print(f"Saved to {output}")


    finally:
        cap.release()

def visualize_problems(records, output):

    all_rows=pd.DataFrame.from_records(records, columns=COLUMNS)
    all_rows.to_csv(output, mode="a", header=False, index=False)
    df=all_rows.groupby("frame_number").first().reset_index()

    if df.shape[0]>0:
        print(df)
    count=len(df["frame_number"].unique())
    return count




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
    ap.add_argument("--output", default=None, help="csv with the problematic frames (default: preprocess_sample.csv next to the store)")
    return ap


def main():

    ap = get_parser()
    args=ap.parse_args()
    store_path = args.store_path
    start_chunk=args.start_chunk
    end_chunk=args.end_chunk
    samples_per_chunk=args.samples_per_chunk

    preprocess_sample(store_path, start_chunk, end_chunk, samples_per_chunk, output=args.output)

if __name__ == "__main__":
    main()