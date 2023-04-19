import os.path


def get_blobs_collection(root, experiment, chunk, step):
    raise NotImplementedError
    # path = os.path.join(root, experiment, "idtrackerai", f"session_{str(chunk).zfill(6)}", step, "blobs_collection.npy")
    # return path

