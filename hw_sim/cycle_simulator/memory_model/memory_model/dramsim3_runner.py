import glob
import json
import os
import shutil
import subprocess
import time
from typing import Any, List, Optional

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

    @staticmethod
    def _parse_json_allow_concatenated(text: str) -> Any:
        if not text.strip():
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Some DRAMSim3 builds append multiple top-level JSON documents
            # into one file. Parse them sequentially for robustness.
            decoder = json.JSONDecoder()
            idx = 0
            docs: List[Any] = []
            n = len(text)
            while idx < n:
                while idx < n and text[idx].isspace():
                    idx += 1
                if idx >= n:
                    break
                try:
                    obj, end = decoder.raw_decode(text, idx)
                except json.JSONDecodeError:
                    trailing = text[idx:].strip()
                    # Some DRAMSim3 epoch dumps end with an unmatched closing
                    # bracket after concatenating multiple top-level documents.
                    if docs and trailing and all(ch in "],}" for ch in trailing):
                        break
                    raise
                docs.append(obj)
                idx = end
            if not docs:
                return None
            if len(docs) == 1:
                return docs[0]
            return docs

    @classmethod
    def _load_json_allow_concatenated(cls, path: str) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(3):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            try:
                return cls._parse_json_allow_concatenated(text)
            except json.JSONDecodeError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.05)
                    continue
                raise
        if last_error is not None:
            raise last_error
        return None

    @staticmethod
    def _normalize_epoch_stats(raw: Any) -> List[Any]:
        if raw is None:
            return []
        if isinstance(raw, list):
            out: List[Any] = []
            for item in raw:
                if isinstance(item, list):
                    out.extend(item)
                else:
                    out.append(item)
            return out
        return [raw]

    def run(self, cfg: DRAMSim3Input) -> DRAMSim3Result:
        outdir = os.path.abspath(cfg.output_dir or os.path.join(self.work_root, "dramsim3_run"))
        ensure_dir(outdir)

        cmd = [self.dramsim3_bin, cfg.config_path]
        if cfg.num_cycles is not None:
            cmd += ["-c", str(cfg.num_cycles)]
        cmd += ["-t", cfg.trace_path]

        # Heartbeat for long-running DRAMSim3 runs (opt-in via env).
        # This keeps numerical behavior unchanged while improving observability.
        hb_s: Optional[float] = None
        for k in ("DRAMSIM3_PROGRESS_INTERVAL_S", "PROGRESS_INTERVAL_S"):
            v = os.getenv(k, "").strip()
            if not v:
                continue
            try:
                hb_s = float(v)
            except Exception:
                hb_s = None
            break
        hb_s = hb_s if (hb_s is not None and hb_s > 0) else None

        # Use file redirection to avoid pipe deadlocks, while still collecting output.
        stdout_path = os.path.join(outdir, "dramsim3.stdout.txt")
        stderr_path = os.path.join(outdir, "dramsim3.stderr.txt")

        start = time.perf_counter()
        last_hb = start
        with open(stdout_path, "w", encoding="utf-8") as f_out, open(stderr_path, "w", encoding="utf-8") as f_err:
            p = subprocess.Popen(
                cmd,
                stdout=f_out,
                stderr=f_err,
                text=True,
                cwd=outdir,
            )
            while True:
                rc = p.poll()
                if rc is not None:
                    break
                if hb_s is not None:
                    now = time.perf_counter()
                    if (now - last_hb) >= hb_s:
                        last_hb = now
                        try:
                            from pointkan_sim.utils.logger import get_logger
                            get_logger().info(
                                f"DRAMSim3 still running... elapsed={now - start:.1f}s | outdir={outdir}"
                            )
                        except Exception:
                            # Fallback: keep silent if simulator logger isn't available.
                            pass
                time.sleep(0.2)

        # Collect output from files (bounded to keep memory sane).
        def _read_tail(pth: str, max_chars: int = 200000) -> str:
            try:
                if not os.path.isfile(pth):
                    return ""
                with open(pth, "r", encoding="utf-8", errors="ignore") as f:
                    s = f.read()
                if len(s) > max_chars:
                    return s[-max_chars:]
                return s
            except Exception:
                return ""

        stdout_s = _read_tail(stdout_path)
        stderr_s = _read_tail(stderr_path)

        # Re-run returncode check with captured output.
        if p.returncode != 0:
            raise RuntimeError(
                f"DRAMsim3 failed with code {p.returncode}\n"
                f"STDOUT(tail):\n{stdout_s}\nSTDERR(tail):\n{stderr_s}"
            )

        overall_json_path, epoch_json_path = self._find_json_outputs(outdir)

        result = DRAMSim3Result(
            stdout=stdout_s,
            stderr=stderr_s,
            output_dir=outdir,
        )

        if overall_json_path and os.path.isfile(overall_json_path):
            result.json_stats = self._load_json_allow_concatenated(overall_json_path)

        if epoch_json_path and os.path.isfile(epoch_json_path):
            epoch_raw = self._load_json_allow_concatenated(epoch_json_path)
            result.epoch_stats = self._normalize_epoch_stats(epoch_raw)

        return result
