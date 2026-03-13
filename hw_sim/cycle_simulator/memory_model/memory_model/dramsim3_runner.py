import glob
import json
import os
import shutil
import subprocess
from typing import Optional

from .types import DRAMSim3Input, DRAMSim3Result
from .utils import ensure_dir


class DRAMSim3Runner:
    def __init__(self, dramsim3_bin: str, work_root: str):
        self.dramsim3_bin = os.path.abspath(dramsim3_bin)
        self.work_root = os.path.abspath(work_root)
        ensure_dir(self.work_root)

        if not os.path.isfile(self.dramsim3_bin):
            raise FileNotFoundError(f"DRAMsim3 binary not found: {self.dramsim3_bin}")

    def _find_json_outputs(self, outdir: str):
        overall = None
        epoch = None

        for p in glob.glob(os.path.join(outdir, "*.json")):
            name = os.path.basename(p).lower()
            if "epoch" in name:
                epoch = p
            else:
                overall = p
        return overall, epoch

    def run(self, cfg: DRAMSim3Input) -> DRAMSim3Result:
        outdir = os.path.abspath(cfg.output_dir or os.path.join(self.work_root, "dramsim3_run"))
        ensure_dir(outdir)

        cmd = [self.dramsim3_bin, cfg.config_path]
        if cfg.num_cycles is not None:
            cmd += ["-c", str(cfg.num_cycles)]
        cmd += ["-t", cfg.trace_path]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=outdir,
            check=False,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"DRAMsim3 failed with code {proc.returncode}\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        overall_json_path, epoch_json_path = self._find_json_outputs(outdir)

        result = DRAMSim3Result(
            stdout=proc.stdout,
            stderr=proc.stderr,
            output_dir=outdir,
        )

        if overall_json_path and os.path.isfile(overall_json_path):
            with open(overall_json_path, "r", encoding="utf-8") as f:
                result.json_stats = json.load(f)

        if epoch_json_path and os.path.isfile(epoch_json_path):
            with open(epoch_json_path, "r", encoding="utf-8") as f:
                result.epoch_stats = json.load(f)

        return result