import argparse
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort

from driver.assets import GRAPH_NAMES, GmfssAssets
from driver.pipeline import GmfssDriver

NUMBER_RE = re.compile(r"\d+")


def load_driver(model_dir: Path, prefer_fp16: bool = False) -> GmfssDriver:
    assets = GmfssAssets.load(model_dir)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]

    sessions = {}
    for name in GRAPH_NAMES:
        graph_path = assets.graph_path(name)

        if prefer_fp16 and name == "fusionnet":
            fp16_path = model_dir / "fusionnet_fp16.onnx"
            if fp16_path.exists():
                graph_path = fp16_path
                print(f"  fusionnet: Using FP16 ({fp16_path})")
            else:
                print(f"  warning: FP16 model not found, falling back to fp32 ({fp16_path})")

        sessions[name] = ort.InferenceSession(
            str(graph_path), sess_options=session_options, providers=providers
        )

    def run_graph(name: str, feeds: dict) -> list:
        return sessions[name].run(None, feeds)

    return GmfssDriver(assets, run_graph)


def allocate_extra_frames(n_pairs: int, target_total: int) -> list[int]:
    if target_total < n_pairs + 1:
        raise ValueError(
            f"target frame count ({target_total}) is smaller than source frame count ({n_pairs + 1})"
        )
    total_extra = target_total - (n_pairs + 1)
    base, remainder = divmod(total_extra, n_pairs)
    return [base + 1 if i < remainder else base for i in range(n_pairs)]


def allocate_extra_frames_fixed(n_pairs: int, multiplier: int) -> list[int]:
    return [multiplier - 1] * n_pairs


def build_timesteps(n_extra: int) -> list[float]:
    return [(i + 1) / (n_extra + 1) for i in range(n_extra)]


def compute_pad_offsets(orig_w: int, orig_h: int, padded_hw: tuple[int, int]) -> tuple[int, int]:
    padded_h, padded_w = padded_hw
    if orig_w > padded_w or orig_h > padded_h:
        raise ValueError(
            f"input resolution {orig_w}x{orig_h} exceeds the model's fixed processing "
            f"resolution {padded_w}x{padded_h}. Downscaling is not supported."
        )
    left = (padded_w - orig_w) // 2
    top = (padded_h - orig_h) // 2
    return left, top


def pad_to_canvas(frame_rgb_hwc: np.ndarray, padded_hw: tuple[int, int]) -> np.ndarray:
    padded_h, padded_w = padded_hw
    orig_h, orig_w = frame_rgb_hwc.shape[:2]
    left, top = compute_pad_offsets(orig_w, orig_h, padded_hw)

    canvas = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
    canvas[top : top + orig_h, left : left + orig_w] = frame_rgb_hwc

    chw = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
    return chw[np.newaxis, ...]


def crop_from_canvas(tensor: np.ndarray, orig_size: tuple[int, int], padded_hw: tuple[int, int]) -> np.ndarray:
    orig_w, orig_h = orig_size
    left, top = compute_pad_offsets(orig_w, orig_h, padded_hw)

    arr = np.clip(tensor[0], 0, 1)
    chw = (arr * 255.0).round().astype(np.uint8)
    hwc = chw.transpose(1, 2, 0)
    return hwc[top : top + orig_h, left : left + orig_w].copy()


def run_video_mode(args, driver: GmfssDriver) -> None:
    import cv2

    input_path = Path(args.input)
    output_path = Path(args.output)
    padded_hw = driver.assets.padded_hw

    def frame_to_padded_tensor(frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return pad_to_canvas(rgb, padded_hw)

    def tensor_to_frame(tensor: np.ndarray, orig_size: tuple[int, int]) -> np.ndarray:
        rgb = crop_from_canvas(tensor, orig_size, padded_hw)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Cannot open input file: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if src_total_frames < 2:
        print("Input video has fewer than 2 frames. Nothing to interpolate.")
        cap.release()
        return

    try:
        compute_pad_offsets(width, height, padded_hw)
    except ValueError as e:
        print(f"Error: {e}")
        cap.release()
        return

    n_pairs = src_total_frames - 1

    if args.multiplier is not None:
        extra_per_pair = allocate_extra_frames_fixed(n_pairs, args.multiplier)
        tail_extra = args.multiplier - 1
        total_frames = src_total_frames + sum(extra_per_pair)
    else:
        extra_per_pair = allocate_extra_frames(n_pairs, args.total_frames)
        tail_extra = extra_per_pair[-1] if extra_per_pair else 0
        total_frames = args.total_frames

    if args.copy_last:
        total_frames += tail_extra

    out_fps = fps * total_frames / src_total_frames

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))

    ok, prev_frame = cap.read()
    if not ok:
        print("Could not read any frame from the input video.")
        cap.release()
        writer.release()
        return

    writer.write(prev_frame)
    written = 1

    last_frame = prev_frame
    for pair_idx in range(n_pairs):
        ok, curr_frame = cap.read()
        if not ok:
            print(f"\nwarning: input ended early at pair {pair_idx + 1}")
            break

        n_extra = extra_per_pair[pair_idx]

        if n_extra > 0:
            img0 = frame_to_padded_tensor(prev_frame)
            img1 = frame_to_padded_tensor(curr_frame)
            timesteps = build_timesteps(n_extra)
            interpolated = driver.interpolate_pair(img0, img1, timesteps=timesteps)
            for t_tensor in interpolated:
                writer.write(tensor_to_frame(t_tensor, (width, height)))
                written += 1

        writer.write(curr_frame)
        written += 1

        prev_frame = curr_frame
        last_frame = curr_frame
        print(f"processing: pair {pair_idx + 1}/{n_pairs} (written {written}/{total_frames})", end="\r")

    if args.copy_last and tail_extra > 0:
        for _ in range(tail_extra):
            writer.write(last_frame)
            written += 1

    cap.release()
    writer.release()
    print(f"\ndone: {output_path} (source {src_total_frames} frames/{fps:.2f}fps -> output {written} frames/{out_fps:.2f}fps)")


def run_sequence_mode(args, driver: GmfssDriver) -> None:
    from PIL import Image

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    padded_hw = driver.assets.padded_hw

    def list_input_frames(directory: Path) -> list[Path]:
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        files = [p for p in directory.iterdir() if p.suffix.lower() in exts]

        def sort_key(p: Path):
            m = NUMBER_RE.search(p.stem)
            return (int(m.group()) if m else 0, p.stem)

        files.sort(key=sort_key)
        return files

    def load_image_rgb(path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.array(img)

    def save_image_rgb(arr_hwc_uint8: np.ndarray, path: Path) -> None:
        img = Image.fromarray(arr_hwc_uint8, mode="RGB")
        img.save(path, format="PNG", method=6)

    def frame_to_padded_tensor(frame_rgb_hwc: np.ndarray) -> np.ndarray:
        return pad_to_canvas(frame_rgb_hwc, padded_hw)

    def tensor_to_frame(tensor: np.ndarray, orig_size: tuple[int, int]) -> np.ndarray:
        return crop_from_canvas(tensor, orig_size, padded_hw)

    if not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = list_input_frames(input_dir)
    src_total_frames = len(frame_paths)

    if src_total_frames < 2:
        print(f"Input folder has fewer than 2 images: {input_dir}")
        return

    print(f"found {src_total_frames} input frames (first: {frame_paths[0].name})")

    n_pairs = src_total_frames - 1

    if args.multiplier is not None:
        extra_per_pair = allocate_extra_frames_fixed(n_pairs, args.multiplier)
        tail_extra = args.multiplier - 1
        total_frames = src_total_frames + sum(extra_per_pair)
    else:
        extra_per_pair = allocate_extra_frames(n_pairs, args.total_frames)
        tail_extra = extra_per_pair[-1] if extra_per_pair else 0
        total_frames = args.total_frames

    if args.copy_last:
        total_frames += tail_extra

    first_frame = load_image_rgb(frame_paths[0])
    orig_h, orig_w = first_frame.shape[:2]
    orig_size = (orig_w, orig_h)

    try:
        compute_pad_offsets(orig_w, orig_h, padded_hw)
    except ValueError as e:
        print(f"Error: {e}")
        return

    digits = args.digits

    def out_path(idx: int) -> Path:
        return output_dir / f"{idx:0{digits}d}.png"

    out_idx = 1
    save_image_rgb(first_frame, out_path(out_idx))
    out_idx += 1
    written = 1

    prev_frame = first_frame
    last_frame = first_frame
    for pair_idx in range(n_pairs):
        curr_frame = load_image_rgb(frame_paths[pair_idx + 1])
        n_extra = extra_per_pair[pair_idx]

        if n_extra > 0:
            img0 = frame_to_padded_tensor(prev_frame)
            img1 = frame_to_padded_tensor(curr_frame)
            timesteps = build_timesteps(n_extra)
            interpolated = driver.interpolate_pair(img0, img1, timesteps=timesteps)
            for t_tensor in interpolated:
                save_image_rgb(tensor_to_frame(t_tensor, orig_size), out_path(out_idx))
                out_idx += 1
                written += 1

        save_image_rgb(curr_frame, out_path(out_idx))
        out_idx += 1
        written += 1

        prev_frame = curr_frame
        last_frame = curr_frame
        print(f"processing: pair {pair_idx + 1}/{n_pairs} (written {written}/{total_frames})", end="\r")

    if args.copy_last and tail_extra > 0:
        for _ in range(tail_extra):
            save_image_rgb(last_frame, out_path(out_idx))
            out_idx += 1
            written += 1

    print(f"\ndone: {output_dir} ({src_total_frames} source frames -> {written} output frames)")


def main():
    parser = argparse.ArgumentParser(description="GMFSS ONNX frame interpolation")
    parser.add_argument("input", help="input video file, or (with -S) an image sequence folder")
    parser.add_argument("output", help="output video file, or (with -S) an image sequence folder")

    rate_group = parser.add_mutually_exclusive_group(required=True)
    rate_group.add_argument(
        "-n",
        "--total-frames",
        type=int,
        help="Target total output frame count.",
    )
    rate_group.add_argument(
        "-x",
        "--multiplier",
        type=int,
        help="Interpolation multiplier applied uniformly to every frame pair.",
    )
    parser.add_argument(
        "-c",
        "--copy-last",
        action="store_true",
        help="Also duplicate the final frame to match the interpolation rate of the last pair.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mode.",
    )
    parser.add_argument(
        "--model-dir", default="models", help="Folder containing the extracted models-v1.0 release. (default: models)"
    )
    parser.add_argument(
        "-S",
        "--seq",
        action="store_true",
        help="Use image sequence mode. (ex: gmfss-onnx-ml input_dir output_dir [argument...])",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=8,
        help="Sequence mode only, output filename digit count. (default: 8)",
    )
    args = parser.parse_args()

    if args.multiplier is not None and args.multiplier < 1:
        parser.error("--multiplier must be >= 1")

    print("Loading...")
    driver = load_driver(Path(args.model_dir), prefer_fp16=args.fp16)

    if args.seq:
        run_sequence_mode(args, driver)
    else:
        run_video_mode(args, driver)


if __name__ == "__main__":
    main()
