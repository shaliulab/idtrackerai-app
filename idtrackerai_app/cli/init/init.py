import os


local_settings_file = """
SETTINGS_PRIORITY=1
COLOR=False
READ_FORMAT="imgstore"
MULTI_STORE_ENABLED=False
NUMBER_OF_JOBS_FOR_BACKGROUND_SUBTRACTION=6
NUMBER_OF_JOBS_FOR_CONNECTING_BLOBS=6
NUMBER_OF_JOBS_FOR_SEGMENTATION=6
NUMBER_OF_JOBS_FOR_SETTING_ID_IMAGES=1
RECONNECT_BLOBS_FROM_CACHE=True
POSTPROCESS_IMPOSSIBLE_JUMPS=False
DISABLE_PROTOCOL_3=True
IDTRACKERAI_IS_TESTING=False
SKIP_SAVING_IDENTIFICATION_IMAGES=False
TIME_FORMAT="H|CF"
DATA_POLICY="remove_segmentation_data"
SKIP_EVERY_FRAME=1
CHUNK=0
"""

def init_idtrackerai():

    os.makedirs("idtrackerai")
    with open("idtrackerai/local_settings.py", "w") as filehandle:
        filehandle.write(local_settings_file)

    os.remove("index.db")

