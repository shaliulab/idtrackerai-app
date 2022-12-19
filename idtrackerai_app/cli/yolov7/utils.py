import os.path
from imgstore.interface import VideoCapture

def validate_store(store_path):
    """Raise a human friendly error if the store cannot be found, is invalid or cannot be opened"""

    assert os.path.exists(store_path), f"Passed store {store_path} not found"
    assert store_path.endswith(".yaml"), f"Passed store {store_path} not valid. You should pass a .yaml file"
    try:
        store = VideoCapture(store_path, 0)
    except Exception as error:
        logger.error(error)
        raise Exception(f"Passed store {store_path} cannot be opened. See error")

    return store
