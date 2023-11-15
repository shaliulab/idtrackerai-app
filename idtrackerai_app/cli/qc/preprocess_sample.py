"""
Sample random frames to verify segmentation parameters

Sometimes the config produced by the user might work
in a decent fraction of frames but not high enough (<99%).
This module systematically checks 10 frames per chunk
by applying the segmentation criteria and flags frames
where the number of animals is not the expected one
This could only be because in that particular frame
1) there are crossings (only in experiments with groups)
2) an animal is missing

Call this script like so:

preprocess-sample --store-path /flyhostel_data/videos/FlyHostel1/1X/2023-10-31_16-00-00/metadata.yaml

It will output for every chunk between 10 and 350, the frames that dont have the expected number of animals
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

SAMPLES_PER_CHUNK=10
def preprocess_sample(store_path):

    start_chunk=10
    end_chunk=350
    last_chunk=start_chunk-1


    cap = VideoCapture(store_path, 1)
    try:
        chunksize=cap._metadata["chunksize"]
        basedir=os.path.dirname(store_path)

        config_file = os.path.join(
            basedir,
            os.path.basename(basedir) + ".conf"
        )

        assert os.path.exists(config_file), f"{config_file} not found"
        with open(config_file, "r") as filehandle:
            config = json.load(filehandle)
            number_of_animals=int(float(config["_number_of_animals"]["value"]))

        frame_numbers=np.arange(start_chunk*chunksize, end_chunk*chunksize, chunksize//SAMPLES_PER_CHUNK)


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



# def preprocess_sample(store_path):

#     start_chunk=10
#     end_chunk=350


#     cap = VideoCapture(store_path, 1)
#     try:
#         chunksize=cap._metadata["chunksize"]
#         basedir=os.path.dirname(store_path)

#         config_file = os.path.join(
#             basedir,
#             os.path.basename(basedir) + ".conf"
#         )

#         assert os.path.exists(config_file), f"{config_file} not found"
#         with open(config_file, "r") as filehandle:
#             config = json.load(filehandle)
#             number_of_animals=int(float(config["_number_of_animals"]["value"]))
#             self={}
#             for k in config:
#                 if isinstance(config[k], dict):
#                     self[k]=SimpleNamespace(**config[k])
#                 else:
#                     self[k]=config[k]

#             self=SimpleNamespace(**self)

#         user_defined_parameters = {
#                     "number_of_animals": int(self._number_of_animals.value),
#                     "min_threshold": self._intensity.value[0],
#                     "max_threshold": self._intensity.value[1],
#                     "min_area": self._area.value[0],
#                     "max_area": self._area.value[1],
#                     "check_segmentation": self._chcksegm.value,
#                     # "tracking_interval": self._tracking_interval,
#                     "apply_ROI": self._applyroi.value,
#                     "rois": self._roi.value,
#                     # "mask": self._mask_img,
#                     "subtract_bkg": self._bgsub.value,
#                     # "bkg_model": self._background_img,
#                     "resolution_reduction": self._resreduct.value,
#                     "track_wo_identification": self._no_ids.value,
#                     "setup_points": None,
#                     "sigma_gaussian_blurring": None,
#                     "knowledge_transfer_folder": None,
#                     "identity_transfer": False,
#                     "identification_image_size": None,
#                 }

#         user_defined_parameters["mask"]=load_mask_img(cap, config)

#         # parameters={}
#         # for k, v in config.items():
#         #     if isinstance(v, dict):
#         #         parameters[k]=v.get("value", None)
#         #     else:
#         #         parameters[k]=v


#         user_defined_parameters["bkg_model"]=None
#         # parameters["resolution_reduction"]=1.0

#         parameters=user_defined_parameters


#         frame_numbers=np.arange(start_chunk*chunksize, end_chunk*chunksize, chunksize//10)
#         print(parameters)
#         import ipdb; ipdb.set_trace()

#         for frame_number in frame_numbers:
#             chunk = frame_number // chunksize
#             cap.set(1, frame_number)
#             ret, frame = cap.read()
#             assert ret
#             found_animals=count_animals_in_frame(frame, frame_number, parameters)
#             if found_animals != number_of_animals:
#                 print(f"frame {frame_number} (chunk {chunk}) has {found_animals} animals")

#     finally:
#         cap.release()



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
    return ap


def main():

    ap = get_parser()
    args=ap.parse_args()
    store_path = args.store_path
    preprocess_sample(store_path)

if __name__ == "__main__":
    main()