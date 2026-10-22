import os.path
import logging
from imgstore.interface import VideoCapture

logger = logging.getLogger(__name__)

def validate_store(store_path, chunk=0):
    """Raise a human friendly error if the store cannot be found, is invalid or cannot be opened"""

    assert os.path.exists(store_path), f"Passed store {store_path} not found"
    assert store_path.endswith(".yaml"), f"Passed store {store_path} not valid. You should pass a .yaml file"
    try:
        store = VideoCapture(store_path, chunk)
    except Exception as error:
        logger.error(error)
        raise Exception(f"Passed store {store_path} cannot be opened. See error") from error

    return store
