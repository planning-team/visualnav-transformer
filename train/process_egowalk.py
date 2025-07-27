import pickle
import fire

from typing import List, Any, List, Callable
from functools import partial
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map
from egowalk_dataset.datasets.trajectory.trajectory import EgoWalkTrajectory
from egowalk_dataset.datasets.gnm.cutters import (AbstractTrajectoryCutter,
                                                  SpikesCutter,
                                                  StuckCutter,
                                                  BackwardCutter,
                                                  apply_cutter)


def do_parallel(task_fn: Callable[[Any], Any], 
                arguments: List[Any], 
                n_workers: int, 
                use_tqdm: bool,
                mode: str) -> List[Any]:
    assert isinstance(n_workers, int) and n_workers >= 0, f"n_workers must be int >=0, got {n_workers}"
    assert mode in ["process", "thread"], f"mode must be 'process' or 'thread', got {mode}"
    if n_workers == 0:
        result = []
        if use_tqdm:
            for arg in tqdm(arguments):
                result.append(task_fn(arg))
        else:
            for arg in arguments:
                result.append(task_fn(arg))
        return result
    else:
        if use_tqdm:
            if mode == "process":
                return process_map(task_fn, arguments, max_workers=n_workers)
            elif mode == "thread":
                return thread_map(task_fn, arguments, max_workers=n_workers)
            else:
                raise ValueError(f"Invalid mode: {mode}")
        else:
            if mode == "process":
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    result = executor.map(task_fn, arguments)
                    return [e for e in result]
            elif mode == "thread":
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    result = executor.map(task_fn, arguments)
                    return [e for e in result]
            else:
                raise ValueError(f"Invalid mode: {mode}")


def process_traj(traj_name: str,
                 dataset_root: Path,
                 output_root: Path,
                 cutters: List[AbstractTrajectoryCutter]):
    traj = EgoWalkTrajectory.from_dataset(name=traj_name,
                                          data_path=dataset_root)
    
    timestamps = traj.odometry.valid_timestamps
    traj_bev = traj.odometry.get_bev(filter_valid=True)
    segments = apply_cutter(trajectory=traj_bev,
                            cutter=cutters)
    segments = [e for e in segments if (e[1] - e[0]) > 12]

    for segment_idx, segment in enumerate(segments):
        segment_output_dir = output_root / f"{traj_name}__{segment_idx}"
        segment_output_dir.mkdir(parents=True, exist_ok=True)

        segment_timestamps = timestamps[segment[0]:segment[1]]
        segment_traj_bev = traj_bev[segment[0]:segment[1]]
        position_data = segment_traj_bev[:, :2]
        yaw_data = segment_traj_bev[:, 2]
        
        for i in range(0, len(segment_timestamps)):
            img = traj.rgb.at(segment_timestamps[i])
            img = Image.fromarray(img)
            img.save(segment_output_dir / f"{i}.jpg")
        
        with open(segment_output_dir / "traj_data.pkl", "wb") as f:
            pickle.dump({"position": position_data, "yaw": yaw_data}, f)


def main(dataset_root: str = "/mnt/vol0/hf_cache/egowalk",
         output_root: str = "/mnt/vol0/datasets/vint_datasets/egowalk"):
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    traj_names = sorted([e.stem for e in (dataset_root / "data").glob("*.parquet")])
    
    cutters=[StuckCutter(eps=1e-2),
            BackwardCutter(backwards_eps=1e-2,
                            stuck_eps=1e-2,
                            ignore_stuck=True),
            SpikesCutter(spike_threshold=2.)]

    task_fn = partial(process_traj,
                      dataset_root=dataset_root,
                      output_root=output_root,
                      cutters=cutters)

    do_parallel(task_fn=task_fn,
                arguments=traj_names,
                n_workers=3,
                use_tqdm=True,
                mode="process")

    # for traj_name in tqdm(traj_names):
    #     process_traj(traj_name=traj_name,
    #                  dataset_root=dataset_root,
    #                  output_root=output_root,
    #                  cutters=cutters)
    #     break
    # print("Done")


if __name__ == "__main__":
    fire.Fire(main)
