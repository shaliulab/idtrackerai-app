import stat
import time
import shlex
import subprocess
import os.path
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile

TS_FILE = tempfile.NamedTemporaryFile(prefix="ts").name

#TS_FILE = os.path.join(os.environ["HOME"], ".config", "ts")
#os.makedirs(os.path.dirname(TS_FILE), exist_ok=True)


def write_shell_script(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as filehandle:
        filehandle.write("#! /bin/bash\n")
        for line in lines:
            filehandle.write(line.rstrip("\n"))
            filehandle.write("\n")
        # filehandle.write("echo 'Hello World'")
        filehandle.write("\n")
    os.chmod(path, stat.S_IRWXU)
    return os.path.exists(path)


def ts_sub(
    script, output, gpu=False, cwd=None, append=False, delay=0, label=None, wait_for=[]
):

    if gpu:
        gpu_requirement = "-g 0"
    else:
        gpu_requirement = "-G 0"

    if append:
        operator = ">>"
    else:
        operator = ">"

    offset = -1
    if delay > 0:
        while offset < delay:
            time.sleep(1)
            offset = time.time() - os.path.getmtime(TS_FILE)

    ts_cmd = f'bash -c "{script} {operator} {output} 2>&1"'
    if label is None:
        label = os.path.basename(script)
    wait_for = [str(w) for w in wait_for]

    if len(wait_for) > 0:
        wait_processes = ",".join(wait_for)
        wait = f"-W {wait_processes}"
        ts_label = "[" + ",".join(wait_for) + "]&&" + f"[{label}]"

    else:
        wait = ""
        ts_label = f"[{label}]"

    ts_prefix = f"ts -n {gpu_requirement} -L {label} {wait}"
    final_cmd = f"{ts_prefix} {ts_cmd}"
    cmd_list = shlex.split(final_cmd)
    Path(TS_FILE).touch()
    process = subprocess.Popen(
        cmd_list, cwd=cwd
    )  # , stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    process2 = subprocess.Popen(["ts"], stdout=subprocess.PIPE)
    data = process2.communicate()

    data = data[0].decode().split("\n")

    data = [shlex.split(line) for line in data]
    lines = [[data[0][0], data[0][1], data[0][6]]]

    for line in data[1:]:
        if len(line) > 6:
            lines.append([line[0], line[1]])
            label_column = [
                "".join([column for column in line if column.startswith("[")])
            ]
            lines[-1].extend(label_column)

    ts_table = pd.DataFrame(lines[1:], columns=lines[0])
    states = ["running", "allocating", "finished"]

    for state in states:
        process_id = find_process(ts_table, state, ts_label)
        if process_id is not None:
            break

    if process_id is None:
        warnings.warn(f"Could not find process with label {ts_label}!")

    return process_id


def find_process(ts_table, state, label):

    processes = ts_table.loc[
        np.bitwise_and(
            ts_table["Command"].str.startswith(label), ts_table["State"] == state
        )
    ]

    processes = processes["ID"].tolist()

    if len(processes) > 0:
        process_id = processes[-1]
        return process_id
    else:
        return None
