import os.path
import glob
import shutil

import fiftyone as fo

from idtrackerai.animals_detection.yolov7.yolov7 import load_detections

FIFTYONE_DATASETS="/Users/FlySleepLab_Dropbox/Data/flyhostel_data/fiftyone"

def load_dataset_to_fiftyone(results_folder, dataset_name):

    labels_folder = os.path.join(os.path.realpath(results_folder), "labels")
    assert os.path.exists(labels_folder)

    label_files = sorted(glob.glob(os.path.join(labels_folder, "*")))

    dataset = fo.load_dataset(dataset_name)

    def add_to_dataset(label_file):

        key = os.path.splitext(os.path.basename(label_file))[0]
        image_file = os.path.join(results_folder, key+".png")
        image_file_in_fiftyone = os.path.join(FIFTYONE_DATASETS, dataset_name, "images", key+".png")
        shutil.copy(image_file, image_file_in_fiftyone)

        sample = fo.Sample(filepath=image_file_in_fiftyone)
        sample.tags.append("load2fiftyone")
        dataset.add_sample(sample)
        dataset.save()


    detections = load_detections(label_files, class_id=0, count=6, false_action=add_to_dataset, true_action=None)



