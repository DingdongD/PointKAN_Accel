import json
import os
import re
import subprocess
from dataclasses import asdict
from typing import Optional, Dict, List, Tuple

from .types import CactiInput, CactiResult
from .utils import ensure_dir, stable_hash_dict


class CactiRunner:
    """
    CACTI runner with:
      1) full template_cfg preservation
      2) parameter override on top of template
      3) automatic stdout/stderr/json saving
      4) optional result reuse
    """

    def __init__(
        self,
        cacti_bin: str,
        template_cfg: str,
        generated_cfg_dir: str,
        output_dir: Optional[str] = None,
        reuse_if_exists: bool = True,
    ):
        self.cacti_bin = os.path.abspath(cacti_bin)
        self.template_cfg = os.path.abspath(template_cfg)
        self.generated_cfg_dir = os.path.abspath(generated_cfg_dir)
        self.output_dir = os.path.abspath(output_dir) if output_dir else None
        self.reuse_if_exists = reuse_if_exists

        ensure_dir(self.generated_cfg_dir)
        if self.output_dir:
            ensure_dir(self.output_dir)

        if not os.path.isfile(self.cacti_bin):
            raise FileNotFoundError(f"CACTI binary not found: {self.cacti_bin}")
        if not os.path.isfile(self.template_cfg):
            raise FileNotFoundError(f"CACTI template cfg not found: {self.template_cfg}")

    # ------------------------------------------------------------------
    # Key / paths
    # ------------------------------------------------------------------
    def _cfg_key(self, cfg: CactiInput) -> str:
        return stable_hash_dict(cfg.__dict__)

    def _get_output_paths(self, key: str) -> dict:
        if not self.output_dir:
            return {}
        return {
            "stdout_txt": os.path.join(self.output_dir, f"cacti_{key}.stdout.txt"),
            "stderr_txt": os.path.join(self.output_dir, f"cacti_{key}.stderr.txt"),
            "result_json": os.path.join(self.output_dir, f"cacti_{key}.result.json"),
            "meta_json": os.path.join(self.output_dir, f"cacti_{key}.meta.json"),
        }

    # ------------------------------------------------------------------
    # Template override helpers
    # ------------------------------------------------------------------
    def _load_template_lines(self) -> List[str]:
        with open(self.template_cfg, "r", encoding="utf-8") as f:
            return f.readlines()

    def _normalize_param_name(self, line: str) -> Optional[str]:
        """
        Extract CACTI parameter key from an active config line.

        Example:
            '-size (bytes) 131072' -> 'size (bytes)'
            '-cache type "cache"' -> 'cache type'
            '-Print level (DETAILED, CONCISE) - "DETAILED"'
                -> 'Print level (DETAILED, CONCISE)'
        """
        s = line.strip()
        if not s:
            return None
        if s.startswith("#"):
            return None
        if s.startswith("//"):
            return None
        if not s.startswith("-"):
            return None

        # Remove leading '-'
        s = s[1:].strip()

        # Split parameter name from value.
        # We assume the "name" is everything before the first value-like token.
        # Since CACTI parameter names can contain spaces and parentheses,
        # we use heuristics:
        #
        # 1) If ' - ' exists (e.g. Print level ... - "DETAILED"), split at last ' - '
        # 2) Else split before the last whitespace-separated token block.
        #
        # More robust approach: match everything up to the final value expression.
        #
        # For ordinary lines like:
        #   size (bytes) 131072
        #   cache type "cache"
        # the regex below works well.
        m = re.match(r'^(.*?)(\s+(?:"[^"]*"|[^\s]+(?:\s*//.*)?))\s*$', s)
        if not m:
            return s.strip()

        name = m.group(1).strip()
        if name.endswith("-"):
            name = name[:-1].strip()
        return name

    def _format_override_lines(self, cfg: CactiInput) -> Dict[str, str]:
        """
        Build the parameter overrides we want to apply to template_cfg.
        Keys must match the parameter names in template_cfg.
        """
        bus_width = cfg.output_bus_width_bits or (cfg.block_size_bytes * 8)

        overrides = {
            "size (bytes)": f"-size (bytes) {cfg.size_bytes}\n",
            "block size (bytes)": f"-block size (bytes) {cfg.block_size_bytes}\n",
            "associativity": f"-associativity {cfg.associativity}\n",
            "read-write port": f"-read-write port {cfg.read_write_port}\n",
            "exclusive read port": f"-exclusive read port {cfg.exclusive_read_port}\n",
            "exclusive write port": f"-exclusive write port {cfg.exclusive_write_port}\n",
            "single ended read ports": f"-single ended read ports {cfg.single_ended_read_ports}\n",
            "UCA bank count": f"-UCA bank count {cfg.uca_bank_count}\n",
            "technology (u)": f"-technology (u) {cfg.technology_u}\n",
            "output/input bus width": f"-output/input bus width {bus_width}\n",
            "cache type": f'-cache type "{cfg.cache_type}"\n',
        }
        return overrides

    def _build_cfg_text_from_template(self, cfg: CactiInput) -> str:
        """
        Preserve full template cfg and override only selected parameters.
        If a parameter is not found in the template, append it at the end.
        """
        lines = self._load_template_lines()
        overrides = self._format_override_lines(cfg)
        remaining = dict(overrides)

        output_lines: List[str] = []

        for line in lines:
            key = self._normalize_param_name(line)

            # Only replace active lines. Commented-out lines stay untouched.
            if key is not None and key in remaining:
                output_lines.append(remaining[key])
                del remaining[key]
            else:
                output_lines.append(line)

        # Append any not-found parameters at the end
        if remaining:
            output_lines.append("\n# ---- Auto-appended overrides by CactiRunner ----\n")
            for _, v in remaining.items():
                output_lines.append(v)

        return "".join(output_lines)

    def _write_cfg(self, cfg: CactiInput) -> Tuple[str, str]:
        key = self._cfg_key(cfg)
        cfg_path = os.path.join(self.generated_cfg_dir, f"cacti_{key}.cfg")
        text = self._build_cfg_text_from_template(cfg)

        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(text)

        return key, cfg_path

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------
    def _extract_float(self, text: str, label: str) -> Optional[float]:
        label = re.escape(label)
        m = re.search(rf"{label}\s*:\s*([0-9eE\.\+\-]+)", text)
        if m:
            return float(m.group(1))
        return None

    def _parse_output(self, text: str) -> CactiResult:
        result = CactiResult(raw_text=text)

        result.access_time_ns = self._extract_float(text, "Access time (ns)")
        result.cycle_time_ns = self._extract_float(text, "Cycle time (ns)")
        result.read_energy_nj = self._extract_float(text, "Total dynamic read energy per access (nJ)")
        result.write_energy_nj = self._extract_float(text, "Total dynamic write energy per access (nJ)")
        result.leak_power_mw = self._extract_float(text, "Total leakage power of a bank (mW)")
        result.gate_leak_power_mw = self._extract_float(text, "Total gate leakage power of a bank (mW)")

        h = self._extract_float(text, "Cache height (mm)")
        w = self._extract_float(text, "Cache width (mm)")

        if h is None or w is None:
            m = re.search(
                r"Cache height x width \(mm\)\s*:\s*([0-9eE\.\+\-]+)\s*x\s*([0-9eE\.\+\-]+)",
                text,
            )
            if m:
                h = float(m.group(1))
                w = float(m.group(2))

        result.height_mm = h
        result.width_mm = w
        if h is not None and w is not None:
            result.area_mm2 = h * w

        return result

    # ------------------------------------------------------------------
    # Save / load cache
    # ------------------------------------------------------------------
    def _save_outputs(
        self,
        key: str,
        cfg: CactiInput,
        cfg_path: str,
        stdout: str,
        stderr: str,
        result: CactiResult,
        returncode: int,
    ) -> None:
        if not self.output_dir:
            return

        paths = self._get_output_paths(key)

        with open(paths["stdout_txt"], "w", encoding="utf-8") as f:
            f.write(stdout)

        with open(paths["stderr_txt"], "w", encoding="utf-8") as f:
            f.write(stderr)

        with open(paths["result_json"], "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)

        meta = {
            "cfg_key": key,
            "cfg_path": cfg_path,
            "template_cfg": self.template_cfg,
            "cacti_bin": self.cacti_bin,
            "returncode": returncode,
            "input_cfg": asdict(cfg),
            "saved_files": paths,
        }
        with open(paths["meta_json"], "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _try_load_cached_result(self, key: str) -> Optional[CactiResult]:
        if not self.output_dir or not self.reuse_if_exists:
            return None

        result_json = self._get_output_paths(key).get("result_json")
        if not result_json or not os.path.isfile(result_json):
            return None

        with open(result_json, "r", encoding="utf-8") as f:
            d = json.load(f)

        return CactiResult(**d)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, cfg: CactiInput) -> CactiResult:
        key, cfg_path = self._write_cfg(cfg)

        cached = self._try_load_cached_result(key)
        if cached is not None:
            cached.cfg_path = cfg_path
            return cached

        proc = subprocess.run(
            [self.cacti_bin, "-infile", cfg_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(self.cacti_bin),
            check=False,
        )

        if proc.returncode != 0:
            if self.output_dir:
                paths = self._get_output_paths(key)
                with open(paths["stdout_txt"], "w", encoding="utf-8") as f:
                    f.write(proc.stdout or "")
                with open(paths["stderr_txt"], "w", encoding="utf-8") as f:
                    f.write(proc.stderr or "")

            raise RuntimeError(
                f"CACTI failed with code {proc.returncode}\n"
                f"CFG: {cfg_path}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        result = self._parse_output(proc.stdout)
        result.cfg_path = cfg_path

        self._save_outputs(
            key=key,
            cfg=cfg,
            cfg_path=cfg_path,
            stdout=proc.stdout,
            stderr=proc.stderr,
            result=result,
            returncode=proc.returncode,
        )

        return result