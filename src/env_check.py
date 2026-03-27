from __future__ import annotations

import argparse
import importlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List

import torch

from .utils import save_json, save_text


def _module_version(name: str) -> str:
    module = importlib.import_module(name)
    return getattr(module, "__version__", "unknown")


def _gpu_info() -> List[Dict[str, str]]:
    if not torch.cuda.is_available():
        return []
    info = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        info.append(
            {
                "index": str(idx),
                "name": props.name,
                "total_memory_gb": f"{props.total_memory / (1024 ** 3):.2f}",
            }
        )
    return info


def _nvidia_smi_output() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def build_report() -> Dict:
    modules = [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "numpy",
        "tqdm",
        "modelscope",
        "yaml",
        "pytest",
    ]
    versions = {}
    for name in modules:
        try:
            versions[name] = _module_version(name)
        except Exception as exc:  # pragma: no cover
            versions[name] = f"unavailable: {exc}"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_info": _gpu_info(),
        "module_versions": versions,
        "nvidia_smi": _nvidia_smi_output(),
        "cwd": os.getcwd(),
    }


def build_markdown(report: Dict) -> str:
    lines = [
        "# 服务器环境记录",
        "",
        f"更新时间：{report['generated_at_utc']}",
        "",
        "## 基础环境",
        "",
        f"- Python：`{report['python_version']}`",
        f"- 平台：`{report['platform']}`",
        f"- torch：`{report['torch_version']}`",
        f"- CUDA：`{report['cuda_version']}`",
        f"- CUDA 可用：`{report['cuda_available']}`",
        f"- GPU 数量：`{report['gpu_count']}`",
        "",
        "## GPU 信息",
        "",
    ]
    if report["gpu_info"]:
        for item in report["gpu_info"]:
            lines.append(
                f"- GPU {item['index']}：`{item['name']}`，显存约 `"
                f"{item['total_memory_gb']} GB`"
            )
    else:
        lines.append("- 当前未检测到可用 GPU")

    lines.extend(["", "## 关键模块版本", ""])
    for name, version in report["module_versions"].items():
        lines.append(f"- {name}：`{version}`")

    if report["nvidia_smi"]:
        lines.extend(["", "## nvidia-smi", "", "```text", report["nvidia_smi"], "```"])

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default="artifacts/env/env_check.json")
    ap.add_argument("--out_md", default="docs/environment.md")
    args = ap.parse_args()

    report = build_report()
    save_json(report, args.out_json)
    save_text(build_markdown(report), args.out_md)
    print(f"[OK] JSON -> {args.out_json}")
    print(f"[OK] Markdown -> {args.out_md}")


if __name__ == "__main__":
    main()
