from __future__ import annotations
import io
import argparse
import subprocess as sp
import multiprocessing as mp
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import h5py
from tqdm import tqdm


img = Image.Image
array = np.ndarray


def get_flow_frames(
    rgb_frames: list[img],
    bound: int,
) -> list[img]:

    gray_frames = [
        np.array(frame.copy().convert("L"))
        for frame in rgb_frames
    ]

    h, w = gray_frames[0].shape[:2]
    dummy_channel = np.zeros(shape=(h, w, 1), dtype=np.uint8)

    func = cv2.cuda_OpticalFlowDual_TVL1().create().calc

    prev_gpu = cv2.cuda_GpuMat()
    next_gpu = cv2.cuda_GpuMat()

    flow_frames = []
    for prev_frame, next_frame in zip(gray_frames[:-1], gray_frames[1:]):
        prev_gpu.upload(prev_frame)
        next_gpu.upload(next_frame)

        flow = func(prev_gpu, next_gpu, None).download()

        # clip by bound value.
        flow = np.clip(flow, -bound, bound)
        flow += bound

        # convert to image.
        flow = (flow * 255) / (2 * bound)
        flow = np.clip(flow, 0, 255).astype(np.uint8)

        flow = np.concatenate([flow, dummy_channel], axis=-1)  # add dummy channel to make n_channels=3.
        flow = Image.fromarray(flow)

        flow_frames.append(flow)

    return flow_frames


def get_rgb_frames(
    video_path: Path,
    tmp_dir: Path,
    size: int = 256,
    fps: int = -1,
) -> list[img]:
    video_id = video_path.stem

    # clean the previous frames.
    for frame_path in tmp_dir.glob(f"{video_id}_*.jpg"):
        frame_path.unlink()

    vf_param = f"scale=trunc(iw*max({size}/iw\,{size}/ih)/2)*2:trunc(ih*max({size}/iw\,{size}/ih)/2)*2"
    if fps > 0:
        vf_param += f",minterpolate={fps}"

    sp.run([
        "ffmpeg", "-i", str(video_path), "-loglevel", "quiet", "-threads", "1",
        "-vf", vf_param, f"{tmp_dir}/{video_id}_%05d.jpg"
    ])

    frames = []
    for frame_path in sorted(tmp_dir.glob(f"{video_id}_*.jpg")):
        frame = Image.open(frame_path)
        frames.append(frame)
        frame_path.unlink()

    return frames


def save_as_hdf5(
    hdf5_path: Path,
    rgb_frames: list[img],
    flow_frames: list[img],
    quality: int = 75
) -> None:

    def image_to_buffer(image: img) -> array:
        image_bin = io.BytesIO()
        image.save(image_bin, "JPEG", quality=quality)
        buffer = np.frombuffer(image_bin.getbuffer(), "uint8")
        return buffer

    with h5py.File(hdf5_path, "w") as f:
        for i, frame in enumerate(rgb_frames):
            f.create_dataset(name=f"RGB/{i:05d}", data=image_to_buffer(frame))
        for i, frame in enumerate(flow_frames):
            f.create_dataset(name=f"Flow/{i:05d}", data=image_to_buffer(frame))


def video_to_hdf5(
    video_path: Path,
    out_dir: Path,
    tmp_dir: Path,
    size: int = 256,
    bound: int = 20,
    fps: int = -1,
    quality: int = 75,
    recompute: bool = False,
) -> None:
    completed_path = out_dir / (video_path.stem + ".hdf5.done")
    hdf5_path = out_dir / (video_path.stem + ".hdf5")

    if recompute:
        completed_path.unlink(missing_ok=True)

    if completed_path.exists():
        return

    # remove hdf5.
    hdf5_path.unlink(missing_ok=True)

    # extract rgb frames.
    rgb_frames = get_rgb_frames(video_path, tmp_dir, size, fps)

    # extract flow frames.
    flow_frames = get_flow_frames(rgb_frames, bound)

    # save them as hdf5.
    save_as_hdf5(hdf5_path, rgb_frames, flow_frames, quality)

    # create empty file to skip this process in the next run.
    completed_path.touch()


def worker(params):
    # wrapper for multiprocessing.
    video_to_hdf5(**params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video files to hdf5 files.")

    parser.add_argument("--video_dir", "-i", type=Path, help="Video directory", required=True)
    parser.add_argument("--out_dir", "-o", type=Path, help="Output directory.", required=True)
    parser.add_argument("--ext", "-e", type=str, default=".mp4", help="Video extension.")
    parser.add_argument("--size", "-s", type=int, default=256, help="size.")
    parser.add_argument("--fps", type=int, default=-1)
    parser.add_argument("--recompute", action="store_true",
                        help="If specify --recompute, recompute all video files again.")
    parser.add_argument("--bound", "-b", type=int, default=20, help="Bi-bound parameter to convert optical flow to image.")
    parser.add_argument("--quality", "-q", type=int, default=75, help="Quality of output images.")
    parser.add_argument("--n_jobs", "-j", type=int, default=-1, help="Number of jobs.")

    args = parser.parse_args()

    print("Args:", vars(args))

    # define tmp_dir
    tmp_dir: Path = args.out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)  # out_dir is also created if not exists.

    # search target videos.
    kwargs_list = []
    for video_path in args.video_dir.glob(f"**/*{args.ext}"):
        kwargs = {
            "video_path": video_path,
            "out_dir": args.out_dir,
            "tmp_dir": tmp_dir,
            "size": args.size,
            "bound": args.bound,
            "fps": args.fps,
            "quality": args.quality,
            "recompute": args.recompute,
        }
        kwargs_list.append(kwargs)

    print(f"Total videos: {len(kwargs_list)}")

    # run in parallel.
    n_jobs = args.n_jobs
    if n_jobs < 0:
        n_jobs = max(1, mp.cpu_count() + n_jobs + 1)

    if n_jobs == 0:
        for kwargs in tqdm(kwargs_list):
            worker(kwargs)
    else:
        print(f"Number of jobs: {n_jobs}")
        with mp.Pool(n_jobs) as pool, \
             tqdm(total=len(kwargs_list)) as pbar:
            for _ in pool.imap_unordered(worker, kwargs_list):
                pbar.update(1)

    # remove tmp_dir.
    print("Clean up unnecessary files...")
    for tmp_file in tmp_dir.iterdir():
        tmp_file.unlink()
    tmp_dir.rmdir()
