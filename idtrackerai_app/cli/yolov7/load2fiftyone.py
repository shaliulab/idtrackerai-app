import os.path
import glob
import shutil
import argparse

import fiftyone as fo

from idtrackerai.animals_detection.yolov7.yolov7 import load_detections

FIFTYONE_DATASETS="/Users/FlySleepLab_Dropbox/Data/flyhostel_data/fiftyone"

def load_dataset_to_fiftyone(results_folder, dataset_name, class_id=None, count=None):

    labels_folder = os.path.join(os.path.realpath(results_folder), "labels")
    assert os.path.exists(labels_folder)

    label_files = sorted(glob.glob(os.path.join(labels_folder, "*")))

    try:
        dataset = fo.load_dataset(dataset_name)
    except Exception as error:
        print(error)
        raise Exception(f"Cannot load dataset {dataset_name}. Does it exist?")

    def add_to_dataset(label_file):

        key = os.path.splitext(os.path.basename(label_file))[0]
        image_file = os.path.join(results_folder, key+".png")
        image_file_in_fiftyone = os.path.join(FIFTYONE_DATASETS, dataset_name, "images", key+".png")
        shutil.copy(image_file, image_file_in_fiftyone)

        sample = fo.Sample(filepath=image_file_in_fiftyone)
        sample.tags.append("load2fiftyone")
        dataset.add_sample(sample)
        dataset.save()


    detections = load_detections(label_files, class_id=class_id, count=count, false_action=add_to_dataset, true_action=None)
    return detections


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="name of fiftyone dataset")
    ap.add_argument("--results_folder", help="path to folder containing YOLOv7 results: should contain .png files and a labels/ folder with a corresponding .txt file for each .png")
    ap.add_argument("--count", type=int, help="target number of detections in each .png", default=None)
    ap.add_argument("--class-id", type=int, help="target class id", default=None)
    return ap

def main():

    ap = get_parser()
    args = ap.parse_args()

    detections=load_dataset_to_fiftyone(results_folder=args.results_folder, dataset_name=args.dataset, class_id=args.class_id, count=args.count)
    return detections
