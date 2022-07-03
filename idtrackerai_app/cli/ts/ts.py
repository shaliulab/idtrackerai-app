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
    return ap


def main():

    ap = get_parser()
    args = ap.parse_args()
    if args.command is not None:
        if not args.command in ["preprocessing", "tracking", "track_video"]:
            raise Exception("Invalid")

    os.chdir(args.root_dir)

    chunk_pad = str(args.chunk).zfill(6)
    metadata_path = os.path.join(args.experiment, STORE_MD_FILENAME)
    with open(metadata_path, "r") as filehandle:
        extension = yaml.load(filehandle, yaml.SafeLoader)["__store"]["extension"]

    logger.info(f"Making {args.experiment}/idtrackerai")
    working_dir = os.path.join(args.experiment, ANALYSIS_FOLDER)
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

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

    lines = []
    for entry in ts_config:
        if type(ts_config[entry]) is str:
            lines.append(f'{entry}="{ts_config[entry]}"')
        else:
            lines.append(f"{entry}={ts_config[entry]}")

    lines.append(f"CHUNK={args.chunk}")
    LOCAL_SETTINGS = os.path.join(f"session_{chunk_pad}-local_settings.py")

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
    if not os.path.exists(f"{args.experiment}.conf"):
        os.symlink(
            os.path.join("..", f"{args.experiment}.conf"), f"{args.experiment}.conf"
        )

    config_file = (
        os.path.join(args.root_dir, args.experiment, args.experiment) + ".conf"
    )
    copy_local_settings_cmd = f"cp {LOCAL_SETTINGS} local_settings.py"
    cwd = os.path.join(args.root_dir, args.experiment, ANALYSIS_FOLDER)

    if args.command is None:
        preprocessing_id = make_session_script_and_run(
            config_file=config_file,
            chunk=args.chunk,
            command="preprocessing",
            cwd=cwd,
            before=[copy_local_settings_cmd],
            wait_for=args.wait_for,
        )

        if preprocessing_id is None:
            return

        tracking_id = make_session_script_and_run(
            config_file=config_file,
            chunk=args.chunk,
            command="tracking",
            cwd=cwd,
            before=[copy_local_settings_cmd],
            wait_for=[preprocessing_id] + args.wait_for,
        )

    else:
        make_session_script_and_run(
            config_file=config_file,
            chunk=args.chunk,
            command=args.command,
            cwd=cwd,
            before=[copy_local_settings_cmd],
            wait_for=args.wait_for,
        )

    os.chdir(args.root_dir)


def make_session_script_and_run(
    config_file, chunk, command, cwd, before=[], wait_for=[]
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

    print(f"Submitting job with label: {label}")

    write_shell_script(session_script, [*before, cmd])
    shutil.copyfile(
        session_script,
        os.path.join(
            os.path.dirname(os.path.dirname(session_script)),
            os.path.basename(session_script),
        ),
    )
    process_id = ts_sub(
        session_script,
        output_file,
        gpu=True,
        cwd=cwd,
        append=command == "tracking",
        label=label,
        wait_for=wait_for,
    )
    return process_id
