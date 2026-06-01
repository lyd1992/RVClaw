from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rvclaw.utils import ensure_dir, utc_now_iso


DEFAULT_REMOTE_WORKDIR = "/data/zl/mnn_rvv_tests"
SUPPORTED_FRAMEWORK = "MNN"


class BenchmarkUnsupportedError(ValueError):
    def __init__(self, message: str, support_hint: str):
        super().__init__(message)
        self.support_hint = support_hint


@dataclass(frozen=True, slots=True)
class MnnRvvTestSpec:
    function_name: str
    test_binary: str
    category: str
    pr: str
    aliases: tuple[str, ...] = ()

    @property
    def search_terms(self) -> tuple[str, ...]:
        return (self.function_name, self.test_binary, *self.aliases)


@dataclass(slots=True)
class MnnRvvRemoteConfig:
    host: str
    user: str
    port: int
    workdir: str

    @classmethod
    def from_env(cls) -> "MnnRvvRemoteConfig":
        return cls(
            host=os.environ.get("RVCLAW_SG2044_HOST", "10.213.6.143"),
            user=os.environ.get("RVCLAW_SG2044_USER", "root"),
            port=int(os.environ.get("RVCLAW_SG2044_PORT", "22")),
            workdir=os.environ.get("RVCLAW_SG2044_WORKDIR", DEFAULT_REMOTE_WORKDIR),
        )

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


def default_manifest() -> list[MnnRvvTestSpec]:
    rows = [
        ("MNNMatrixProd", "test_matrix_prod", "matmul", "#3779"),
        ("MNNPackC4ForMatMul_A", "test_pack_c4_for_mat_mul_a", "matmul", "#3813", ("packc4formatmula",)),
        ("MNNPackC2", "test_pack_c2", "pack", "#4021"),
        ("MNNPackC4", "test_pack_c4", "pack", "#4021"),
        ("MNNUnpackC4", "test_unpack_c4", "pack", "#4021"),
        ("MNNMatrixAdd", "test_matrix_add", "elementwise", "#3913"),
        ("MNNMatrixSub", "test_matrix_sub", "elementwise", "#3913"),
        ("MNNMatrixMax", "test_matrix_max", "elementwise", "#3913"),
        ("MNNAxByClampBroadcastUnit", "test_ax_by_clamp_broadcast_unit", "elementwise", "#4026"),
        ("MNNScaleAndAddBias", "test_scale_and_add_bias", "elementwise", "#4026"),
        ("MNNTranspose16Bit", "test_transpose_16bit", "transpose", "#4023"),
        ("MNNTranspose32Bit", "test_transpose_32bit", "transpose", "#4023"),
        ("MNNConvRunForLineDepthwise", "test_conv_run_for_line_depthwise", "conv", "#4042"),
        ("MNNDeconvRunForUnitDepthWise", "test_deconv_run_for_unit_depth_wise", "conv", "#4042"),
        ("MNNConvRunForLineint8_t", "test_conv_run_for_line_int8", "conv", "#4457", ("MNNConvRunForLineInt8",)),
        ("MNNConvRunForUnitint8_t", "test_conv_run_for_unit_int8", "conv", "#4457", ("MNNConvRunForUnitInt8",)),
        ("MNNSoftmax", "test_softmax", "activation", "#4044", ("softmax",)),
        ("MNNReluWithSlopeChannel", "test_relu_with_slope_channel", "activation", "#4044", ("relu",)),
        ("MNNMaxFloat", "test_max_float", "reduce", "#4036"),
        ("MNNMinFloat", "test_min_float", "reduce", "#4036"),
        ("MNNVectorTop1Float", "test_vector_top1_float", "reduce", "#4050"),
        ("MNNVectorTop1Int32", "test_vector_top1_int32", "reduce", "#4050"),
        ("MNNAbsMaxFP32", "test_absmax_fp32", "reduce", "#4433"),
        ("MNNAccumulateSequenceNumber", "test_accumulate_sequence_number", "reduce", "#4433"),
        ("CPUBilinearLineC4", "test_cpu_bilinear_line_c4", "resize", "#4053"),
        ("CPUBilinearSampleC4", "test_cpu_bilinear_sample_c4", "resize", "#4053"),
        ("MNNBilinearLineC8", "test_bilinear_line_c8", "resize", "#4053"),
        ("MNNBilinearSampleC8", "test_bilinear_sample_c8", "resize", "#4053"),
        ("MNNCubicLineC4", "test_cubic_line_c4", "resize", "#4053"),
        ("MNNCubicSampleC4", "test_cubic_sample_c4", "resize", "#4053"),
        ("MNNCubicLineC16", "test_cubic_line_c16", "resize", "#4053"),
        ("MNNCubicSampleC16", "test_cubic_sample_c16", "resize", "#4053"),
        ("MNNGRAYToC3", "test_gray_to_c3", "blitter", "#4067", ("MNNGrayToC3",)),
        ("MNNGRAYToC4", "test_gray_to_c4", "blitter", "#4067", ("MNNGrayToC4",)),
        ("MNNRGBToGRAY", "test_rgb_to_gray", "blitter", "#4067", ("MNNRGBToGray",)),
        ("MNNRGBAToGRAY", "test_rgba_to_gray", "blitter", "#4067", ("MNNRGBAToGray",)),
        ("MNNBGRToGRAY", "test_bgr_to_gray", "blitter", "#4067", ("MNNBGRToGray",)),
        ("MNNBGRAToGRAY", "test_bgra_to_gray", "blitter", "#4067", ("MNNBGRAToGray",)),
        ("MNNRGBToBGR", "test_rgb_to_bgr", "blitter", "#4067"),
        ("MNNRGBAToBGR", "test_rgba_to_bgr", "blitter", "#4067"),
        ("MNNBGRAToBGR", "test_bgra_to_bgr", "blitter", "#4067"),
        ("MNNRGBAToBGRA", "test_rgba_to_bgra", "blitter", "#4067"),
        ("MNNC3ToC4", "test_c3_to_c4", "blitter", "#4067"),
        ("MNNC3ToYUV", "test_c3_to_yuv", "colorspace", "#4079"),
        ("MNNC3ToHSV", "test_c3_to_hsv", "colorspace", "#4079"),
        ("MNNC3ToXYZ", "test_c3_to_xyz", "colorspace", "#4079"),
        ("MNNC3ToBGR555", "test_c3_to_bgr555", "colorspace", "#4079"),
        ("MNNC3ToBGR565", "test_c3_to_bgr565", "colorspace", "#4079"),
        ("MNNNV21ToRGB", "test_nv21_to_rgb", "colorspace", "#4079"),
        ("MNNNV21ToRGBA", "test_nv21_to_rgba", "colorspace", "#4079"),
        ("MNNNV21ToBGR", "test_nv21_to_bgr", "colorspace", "#4079"),
        ("MNNNV21ToBGRA", "test_nv21_to_bgra", "colorspace", "#4079"),
        ("MNNStrassenMergeCFunction", "test_strassen_merge_c_function", "strassen", "#4042"),
        ("MNNCopyC4WithStride", "test_copy_c4_with_stride", "stride", "#4026"),
        ("MNNAddC4WithStride", "test_add_c4_with_stride", "stride", "#4026"),
        ("MNNAvgPoolInt8", "test_avg_pool_int8", "quant", "#4359"),
        ("MNNMaxPoolInt8", "test_max_pool_int8", "quant", "#4359"),
        ("MNNFloat2Int8", "test_float2int8", "quant", "#4359"),
        ("MNNInt8ScaleToFloat", "test_int8_scale_to_float", "quant", "#4359"),
        ("MNNLineDepthWiseInt8AddBiasScaleUnit", "test_line_depthwise_int8", "quant", "#4359"),
        ("MNNReluWithSlopeChannelInt8", "test_relu_slope_int8", "quant", "#4359"),
        ("MNNDynamicQuantFP32", "test_dynamic_quant_fp32", "quant", "#4433"),
        ("MNNSumWeightInt8", "test_sum_weight_int8", "quant", "#4433"),
        ("MNNAsyQuantFunc", "test_asy_quant_func", "quant", "#4433"),
        ("MNNAsyQuantInfo_FP32", "test_asy_quant_info_fp32", "quant", "#4433", ("MNNAsyQuantInfoFP32",)),
        ("MNNReorderWeightInt4", "test_reorder_weight_int4", "quant", "#4433"),
        ("MNNGemmInt8AddBiasScale_16x4_Unit_RVV", "test_gemm_int8_add_bias_scale", "int8_gemm", "#4331"),
        ("MNNPackedMatMul_int8", "test_packed_matmul_int8", "int8_gemm", "#4433", ("MNNPackedMatMulInt8",)),
        ("MNNPackedMatMulRemainFP32", "test_packed_matmul_remain_fp32", "fp32_gemm", "#4426"),
        ("MNNSumByAxisLForMatmul_A", "test_sumbyl_for_matmul_a", "fp32_gemm", "#4425"),
    ]
    return [
        MnnRvvTestSpec(
            function_name=row[0],
            test_binary=row[1],
            category=row[2],
            pr=row[3],
            aliases=tuple(row[4]) if len(row) > 4 else (),
        )
        for row in rows
    ]


def support_hint() -> str:
    categories = sorted({spec.category for spec in default_manifest()})
    examples = ", ".join(spec.function_name for spec in default_manifest()[:8])
    return (
        "当前仅支持MNNRVV静态函数测试。"
        f"支持类别:{', '.join(categories)}。"
        f"示例函数:{examples}。"
    )


def normalize_name(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", value.lower())


def resolve_test_spec(function: str, manifest: list[MnnRvvTestSpec] | None = None) -> MnnRvvTestSpec:
    manifest = manifest or default_manifest()
    needle = normalize_name(function)
    for spec in manifest:
        if any(needle == normalize_name(term) for term in spec.search_terms):
            return spec
    for spec in manifest:
        if any(needle and needle in normalize_name(term) for term in spec.search_terms):
            return spec
    raise BenchmarkUnsupportedError(f"未覆盖的MNNRVV函数:{function}", support_hint())


def is_mnn_rvv_goal(goal: str) -> bool:
    normalized = goal.lower()
    markers = ("mnn", "rvv", "speedup", "benchmark", "加速比", "优化提升", "优化", "test_")
    if any(marker in normalized for marker in markers):
        return True
    compact = normalize_name(goal)
    return any(normalize_name(spec.function_name) in compact or normalize_name(spec.test_binary) in compact for spec in default_manifest())


def extract_framework(goal: str) -> str:
    normalized = goal.lower()
    for framework in ("pytorch", "torch", "tensorflow", "tensorrt", "onnx", "vllm", "mnn"):
        if framework in normalized:
            return "MNN" if framework == "mnn" else framework.upper()
    return "MNN"


def extract_function_from_goal(goal: str) -> str:
    compact = normalize_name(goal)
    for spec in sorted(default_manifest(), key=lambda item: len(item.function_name), reverse=True):
        if any(normalize_name(term) in compact for term in spec.search_terms):
            return spec.function_name
    match = re.search(r"\b(test_[0-9a-zA-Z_]+)\b", goal)
    if match:
        return match.group(1)
    match = re.search(r"\b(MNN[0-9A-Za-z_]+|CPU[0-9A-Za-z_]+)\b", goal)
    if match:
        return match.group(1)
    words = re.findall(r"[A-Za-z][0-9A-Za-z_]+", goal)
    return words[-1] if words else goal.strip()


def run_benchmark(
    framework: str,
    function: str,
    mode: str,
    refresh: bool,
    artifact_dir: str | Path,
) -> dict[str, Any]:
    if framework.upper() != SUPPORTED_FRAMEWORK:
        raise BenchmarkUnsupportedError(f"不支持的框架:{framework}", support_hint())
    if mode != "single":
        raise BenchmarkUnsupportedError(f"不支持的benchmark模式:{mode}", "第一版仅支持mode=single的单函数测试。")

    spec = resolve_test_spec(function)
    artifact_root = ensure_dir(Path(artifact_dir))
    mnn_dir = ensure_dir(artifact_root / "mnn_rvv")
    local_output = os.environ.get("RVCLAW_MNN_RVV_OUTPUT_DIR")
    remote = MnnRvvRemoteConfig.from_env()
    raw_log = ""
    source = "ssh"

    if local_output:
        source = "local_refresh" if refresh else "local_output"
        if refresh:
            raw_log = run_local_refresh(remote.workdir)
        try:
            copy_existing_output(Path(local_output), mnn_dir)
        except Exception as exc:
            raise BenchmarkUnsupportedError(str(exc), output_location_hint(local=True)) from exc
        raw_log += f"\nImported local MNNRVV output from {local_output}"
    else:
        try:
            raw_log = fetch_remote_output(remote, mnn_dir, refresh=refresh)
        except Exception as exc:
            raise BenchmarkUnsupportedError(str(exc), output_location_hint(local=False)) from exc

    jsonl_path = mnn_dir / "all_results.jsonl"
    try:
        rows = read_jsonl(jsonl_path)
    except Exception as exc:
        raise BenchmarkUnsupportedError(str(exc), output_location_hint(local=bool(local_output))) from exc
    matched = [row for row in rows if str(row.get("test")) == spec.test_binary]
    if not matched:
        raise BenchmarkUnsupportedError(f"output中没有函数{spec.function_name}的结果", support_hint())

    chart_path = artifact_root / "mnn_speedup_bar.svg"
    write_speedup_svg(chart_path, spec, matched)
    summary = summarize_results(spec, matched)
    summary.update(
        {
            "benchmark_framework": SUPPORTED_FRAMEWORK,
            "benchmark_mode": mode,
            "benchmark_source": source,
            "benchmark_chart": str(chart_path),
            "benchmark_results_jsonl": str(jsonl_path),
            "remote_host": remote.host,
            "remote_workdir": remote.workdir,
            "refresh": refresh,
            "raw_log": raw_log,
            "summary": (
                f"{spec.function_name}完成{summary['case_count']}个case，"
                f"平均加速比{summary['speedup_mean']:.2f}x，最大加速比{summary['speedup_max']:.2f}x。"
            ),
        }
    )
    return summary


def copy_existing_output(source_output: Path, target_dir: Path) -> None:
    if not source_output.is_dir():
        raise FileNotFoundError(f"MNNRVV本地output目录不存在:{source_output}")
    jsonl = source_output / "all_results.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"MNNRVV本地output缺少all_results.jsonl:{jsonl}")
    shutil.copy2(jsonl, target_dir / "all_results.jsonl")
    report = source_output / "report.md"
    if report.is_file():
        shutil.copy2(report, target_dir / "source_report.md")


def output_location_hint(local: bool) -> str:
    if local:
        return "请检查RVCLAW_MNN_RVV_OUTPUT_DIR是否指向包含all_results.jsonl的output目录。"
    return "请检查RVCLAW_SG2044_HOST/RVCLAW_SG2044_USER/RVCLAW_SG2044_PORT/RVCLAW_SG2044_WORKDIR以及远端output/all_results.jsonl。"


def fetch_remote_output(remote: MnnRvvRemoteConfig, target_dir: Path, refresh: bool) -> str:
    if refresh:
        command = f"cd {shell_quote(remote.workdir)} && bash run_all_and_report.sh --run-only"
        completed = run_command(["ssh", "-p", str(remote.port), remote.target, command], timeout_s=900)
        raw_log = completed.stdout + completed.stderr
    else:
        raw_log = f"Skipped remote rerun at {utc_now_iso()}"

    remote_output = f"{remote.target}:{remote.workdir.rstrip('/')}/output/"
    completed = run_command(["scp", "-P", str(remote.port), f"{remote_output}all_results.jsonl", str(target_dir / "all_results.jsonl")], timeout_s=120)
    raw_log += "\n" + completed.stdout + completed.stderr
    report_result = subprocess.run(
        ["scp", "-P", str(remote.port), f"{remote_output}report.md", str(target_dir / "source_report.md")],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    raw_log += "\n" + report_result.stdout + report_result.stderr
    return raw_log


def run_local_refresh(workdir: str) -> str:
    script = Path(workdir) / "run_all_and_report.sh"
    if not script.is_file():
        raise FileNotFoundError(f"MNNRVV本地测试脚本不存在:{script}")
    completed = run_command(["bash", str(script), "--run-only"], timeout_s=900, cwd=Path(workdir))
    return completed.stdout + completed.stderr


def run_command(args: list[str], timeout_s: int, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(args, text=True, capture_output=True, check=False, timeout=timeout_s, cwd=cwd)
    except FileNotFoundError as exc:
        raise RuntimeError("缺少ssh/scp命令，请确认本机已安装OpenSSH客户端。") from exc
    if completed.returncode != 0:
        command_text = " ".join(args[:3])
        raise RuntimeError(f"远端MNNRVV命令失败:{command_text};{completed.stderr.strip() or completed.stdout.strip()}")
    return completed


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"MNNRVVoutput缺少all_results.jsonl:{path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"MNNRVVoutput为空:{path}")
    return rows


def summarize_results(spec: MnnRvvTestSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    speedups = [float(row.get("speedup") or 0) for row in rows]
    passed = [row for row in rows if bool(row.get("passed"))]
    return {
        "benchmark_function": spec.function_name,
        "benchmark_test_binary": spec.test_binary,
        "benchmark_category": spec.category,
        "benchmark_pr": spec.pr,
        "case_count": len(rows),
        "passed_count": len(passed),
        "failed_count": len(rows) - len(passed),
        "speedup_mean": round(sum(speedups) / len(speedups), 4) if speedups else 0.0,
        "speedup_max": round(max(speedups), 4) if speedups else 0.0,
        "speedup_min": round(min(speedups), 4) if speedups else 0.0,
        "vlen": rows[0].get("vlen"),
        "isa": rows[0].get("isa"),
        "cases": [
            {
                "config": row.get("config"),
                "passed": bool(row.get("passed")),
                "speedup": float(row.get("speedup") or 0),
                "scalar_s": row.get("scalar_s"),
                "rvv_s": row.get("rvv_s"),
            }
            for row in rows
        ],
    }


def write_speedup_svg(path: Path, spec: MnnRvvTestSpec, rows: list[dict[str, Any]]) -> Path:
    width = 920
    height = 420
    margin_left = 70
    margin_bottom = 92
    chart_width = width - margin_left - 28
    chart_height = height - 86 - margin_bottom
    max_speedup = max([float(row.get("speedup") or 0) for row in rows] or [1.0])
    max_speedup = max(max_speedup, 1.0)
    bar_gap = 8
    bar_width = max(14, int((chart_width - bar_gap * max(len(rows) - 1, 0)) / max(len(rows), 1)))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0d1214"/>',
        f'<text x="{margin_left}" y="32" fill="#eef5f2" font-size="20" font-family="Segoe UI,Arial">{html.escape(spec.function_name)}speedupbyconfig</text>',
        f'<text x="{margin_left}" y="56" fill="#91a29e" font-size="13" font-family="Segoe UI,Arial">{html.escape(spec.category)} {html.escape(spec.pr)} | higherisbetter</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 28}" y2="{height - margin_bottom}" stroke="#2a363c"/>',
        f'<line x1="{margin_left}" y1="76" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#2a363c"/>',
    ]
    for tick in range(0, 5):
        value = max_speedup * tick / 4
        y = height - margin_bottom - (value / max_speedup) * chart_height
        parts.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{width - 28}" y2="{y:.1f}" stroke="#1d292e"/>')
        parts.append(f'<text x="12" y="{y + 4:.1f}" fill="#91a29e" font-size="12" font-family="Segoe UI,Arial">{value:.1f}x</text>')
    for index, row in enumerate(rows):
        speedup = float(row.get("speedup") or 0)
        x = margin_left + index * (bar_width + bar_gap)
        bar_height = (speedup / max_speedup) * chart_height
        y = height - margin_bottom - bar_height
        color = "#35d49d" if row.get("passed") else "#ff6b6b"
        label = str(row.get("config") or f"case{index + 1}")
        short = label if len(label) <= 18 else label[:15] + "..."
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" fill="#eef5f2" font-size="11" font-family="Segoe UI,Arial">{speedup:.2f}x</text>')
        parts.append(f'<text transform="translate({x + bar_width / 2:.1f},{height - margin_bottom + 18}) rotate(45)" fill="#91a29e" font-size="10" font-family="Segoe UI,Arial">{html.escape(short)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
