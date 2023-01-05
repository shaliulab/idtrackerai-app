import argparse
import shutil
import json
import os.path
import logging

import yaml

from imgstore.constants import STORE_MD_FILENAME
from idtrackerai.constants import ANALYSIS_FOLDER

from .utils import write_shell_script, ts_sub

TS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ts.json")

assert os.path.exists(TS_JSON)
logger = logging.getLogger(__name__)


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--chunk", required=True, type=int)
    ap.add_argument("--command", required=False)
    ap.add_argument("--wait_for", type=int, nargs="+", required=False, default=[])
    ap.add_argument("--reconnect-blobs-from-cache", action="store_true", default=None)
    ap.add_argument("--submit", action="store_true", dest="submit", default=True)
    ap.add_argument("--no-submit", action="store_false", dest="submit", default=True)
    ap.add_argument("--skip-every-frame", default=1, type=int, help="If differnt from 1, only every skip-every-frameth is analyzed, and all others will have no blobs")
    return ap


def main():

    ap = get_parser()
    args = ap.parse_args()
    if args.command is not None:
        if not args.command in ["preprocessing", "tracking", "track_video", "crossings_detection_and_fragmentation"]:
            raise Exception("Invalid")
        
        
    root_dir = args.root_dir
    
    chunk = args.chunk
    experiment = args.experiment
    reconnect_blobs_from_cache=args.reconnect_blobs_from_cache
    skip_every_frame=args.skip_every_frame
    submit=args.submit
    command=args.command
    wait_for=args.wait_for
    store_path=os.path.join(root_dir, experiment, STORE_MD_FILENAME)
    skip_saving_identification_images=False


    # IMPORTANT
    # reconnect_blobs_from_cache must be False in this step
    # to keep the reconnect_from_cache function from setting blobs_are_connected to True
    # when there is no cache! 

    if command == "preprocessing" and (reconnect_blobs_from_cache is None or reconnect_blobs_from_cache is True):
        logger.warning(f"Detected reconnect_blobs_from_cache is set to {reconnect_blobs_from_cache}. This is forbidden for command {command}, only False is allowed. I will set it to False")
        reconnect_blobs_from_cache = False

    push_idtrackerai_job_to_ts(root_dir, store_path, chunk, command, reconnect_blobs_from_cache, skip_saving_identification_images, skip_every_frame=skip_every_frame, wait_for=wait_for, submit=submit)
    
def push_idtrackerai_job_to_ts(root_dir, store_path, chunk, command, reconnect_blobs_from_cache, skip_saving_identification_images, skip_every_frame=1, wait_for=[], submit=True):

    os.chdir(root_dir)

    chunk_pad = str(chunk).zfill(6)
    metadata_path = store_path
    experiment = os.path.basename(os.path.dirname(store_path))
    
    with open(metadata_path, "r") as filehandle:
        extension = yaml.load(filehandle, yaml.SafeLoader)["__store"]["extension"]

    logger.info(f"Making {experiment}/idtrackerai")
    working_dir = os.path.join(experiment, ANALYSIS_FOLDER)
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)
    # "MAX_RATIO_OF_PRETRAINED_IMAGES": 0.80,
    # "THRESHOLD_EARLY_STOP_ACCUMULATION": 0.80,
    
    # lines = [
    #     "SETTINGS_PRIORITY=1",
    #     f"CHUNK={args.chunk}",
    #     "COLOR=False",
    #     "READ_FORMAT=\"imgstore\"",
    #     "MULTI_STORE_ENABLED=False",
    #     "NUMBER_OF_JOBS_FOR_BACKGROUND_SUBTRACTION=12",
    #     "NUMBER_OF_JOBS_FOR_CONNECTING_BLOBS=12",
    #     "NUMBER_OF_JOBS_FOR_SEGMENTATION=12",
    #     "NUMBER_OF_JOBS_FOR_SETTING_ID_IMAGES=12",
    #     "RECONNECT_BLOBS_FROM_CACHE=True",
    #     "MAX_RATIO_OF_PRETRAINED_IMAGES=0.80",
    #     "THRESHOLD_EARLY_STOP_ACCUMULATION=0.80",
    #     "POSTPROCESS_IMPOSSIBLE_JUMPS=False",
    #     "DISABLE_PROTOCOL_3=True",
    # ]

    with open(TS_JSON, "r") as filehandle:
        ts_config = json.load(filehandle)


    ts_config["SKIP_EVERY_FRAME"] = skip_every_frame
        
    if experiment == "lowres":
        # ts_config["SIGMA_GAUSSIAN_BLURRING"] = 2
        ts_config["ADVANCED_SEGMENTATION"] = False
    
    if reconnect_blobs_from_cache is not None:
        ts_config["RECONNECT_BLOBS_FROM_CACHE"] = reconnect_blobs_from_cache

    lines = []
    for entry in ts_config:
        if type(ts_config[entry]) is str:
            lines.append(f'{entry}="{ts_config[entry]}"')
        else:
            lines.append(f"{entry}={ts_config[entry]}")

    lines.append(f"CHUNK={chunk}")
    LOCAL_SETTINGS = os.path.join(f"session_{chunk_pad}-local_settings.py")

    logger.info(f"Saving {LOCAL_SETTINGS}")
    print(f"Saving {LOCAL_SETTINGS}")
    with open(LOCAL_SETTINGS, "w") as filehandle:
        for line in lines:
            filehandle.write(f"{line}\n")

    if os.path.exists(chunk_pad + extension):
        if os.path.islink(chunk_pad + extension):
            os.remove(chunk_pad + extension)
        else:
            raise Exception(
                "An actual video has been found on the analysis folder. Only softlinks should be present"
            )
    os.symlink(os.path.join("..", chunk_pad + extension), chunk_pad + extension)

    if not os.path.exists(f"{experiment}.conf"):
        os.symlink(
            os.path.join("..", f"{experiment}.conf"), f"{experiment}.conf"
        )

    config_file = (
        os.path.join(root_dir, experiment, experiment) + ".conf"
    )
    copy_local_settings_cmd = f"cp {LOCAL_SETTINGS} local_settings.py"
    cwd = os.path.join(root_dir, experiment, ANALYSIS_FOLDER)

    if command is None:
        session = make_session_script(
            config_file=config_file,
            chunk=chunk,
            command="preprocessing",
            before=[copy_local_settings_cmd],
            wait_for=wait_for,
        )
        
        if submit:
            preprocessing_id = submit_to_ts(*session, wd=cwd)

        if preprocessing_id is None:
            return

        session = make_session_script(
            config_file=config_file,
            chunk=chunk,
            command="tracking",
            before=[copy_local_settings_cmd],
            wait_for=[preprocessing_id] + wait_for,
        )
        if submit:
            preprocessing_id = submit_to_ts(*session, wd=cwd)


    else:
        session = make_session_script(
            config_file=config_file,
            chunk=chunk,
            command=command,
            before=[copy_local_settings_cmd],
            wait_for=wait_for,
        )
        if submit:
            preprocessing_id = submit_to_ts(*session, wd=cwd)

    os.chdir(root_dir)


def submit_to_ts(session_script, output_file, label, wait_for, command, wd):

    print(f"Submitting job with label: {label}")

    process_id = ts_sub(
        session_script,
        output_file,
        gpu=True,
        cwd=wd,
        append=command == "tracking",
        label=label,
        wait_for=wait_for,
    )
    return process_id


def make_session_script(
    config_file, chunk, command, before=[], wait_for=[]
):

    chunk_pad = str(chunk).zfill(6)
    session_script = os.path.join(
        ".", f"session_{chunk_pad}", f"session_{chunk_pad}_{command}.sh"
    )
    output_file = os.path.join(
        f"session_{chunk_pad}", f"session_{chunk_pad}_{command}_output.txt"
    )

    cmd = f"idtrackerai terminal_mode --_session {chunk_pad}  --load  {config_file} --exec {command}"
    logger.debug(f"Running {cmd} with ts")

    label = f"{chunk}_{command}"

    path = os.path.join(
        os.path.dirname(os.path.dirname(session_script)),
        os.path.basename(session_script),
    )
    write_shell_script(session_script, [*before, cmd])
    shutil.copyfile(
        session_script,
        path,
    )
    print(f"Saving {path}")
    
    return session_script, output_file, label, wait_for, command
