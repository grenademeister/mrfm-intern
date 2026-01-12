import os
import random
import re
from pathlib import Path
import json

import scipy.io as sio
import nibabel as nib
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from multiprocessing import Pool

DATA_DIR = "/fast_storage/intern/data/data_curation"
OASIS3_DIR = f"{DATA_DIR}/oasis3/image"
OUTPUT_DIR = "/fast_storage/intern/data/instruction_tuning/oasis3_longitudinal_mat"
SEED = 42
WORKERS = max(1, (os.cpu_count() or 1) - 10)

INSTRUCTION_TEMPLATES = [
    "Predict the MRI slice {time} days later.",
    "Generate the follow-up MRI slice after {time} days.",
    "Synthesize the future MRI slice at {time} days from now.",
    "Create the next-timepoint MRI slice ({time} days later).",
    "Forecast how this MRI slice will look after {time} days.",
    "Estimate the follow-up MRI appearance at {time} days.",
    "Produce the MRI slice corresponding to {time} days into the future.",
    "Given this MRI slice, generate the future slice {time} days later.",
]

_WORKER_TASKS = []
_WORKER_OUTPUT_DIR = None
_WORKER_SEED = 0

def _init_worker(tasks, output_dir, seed):
    global _WORKER_TASKS, _WORKER_OUTPUT_DIR, _WORKER_SEED
    _WORKER_TASKS = tasks
    _WORKER_OUTPUT_DIR = Path(output_dir)
    _WORKER_SEED = seed


def extract_days(path):
    match = re.search(r"sess?-d(\d+)", path.name)
    return int(match.group(1)) if match else 0


def choose_slice_index(data, num_slices=4, rng=random):
    z_dim = data.shape[2]
    slice_means = np.mean(data, axis=(0, 1))
    low = z_dim * 3 // 10
    high = z_dim * 7 // 10

    middle = slice_means[low:high]
    peak_pos = int(np.argmax(middle)) + low
    peak_intensity = slice_means[peak_pos]

    left_threshold_intensity = peak_intensity * 0.8
    right_threshold_intensity = peak_intensity * 0.7

    valid_indices = [i for i in range(low, peak_pos) if slice_means[i] >= left_threshold_intensity
                     ] + [i for i in range(peak_pos, high) if slice_means[i] >= right_threshold_intensity]

    if len(valid_indices) < num_slices:
        print(f"Warning: Not enough valid slices found. Needed {num_slices}, found {len(valid_indices)}.")
        return []
    
    indices = rng.sample(valid_indices, num_slices)
    indices.sort()
    return indices


def load_grouped_data(root_path):
    root = Path(root_path)
    source_nii_list = sorted(root.glob("*.nii.gz"))
    print(f"Found {len(source_nii_list)} nii.gz files")
    
    grouped_data = defaultdict(lambda: defaultdict(list))
    
    for file_path in source_nii_list:
        filename = file_path.name
        if "echo-" in filename:
            continue
        json_path = file_path.parent / (file_path.name.replace(".nii.gz", ".json"))

        if not json_path.exists():
            print(f"Skip: JSON not found for {filename}")
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                if len(metadata) < 15:
                    continue
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            continue

        match = re.search(r"sub-(?P<sub_id>[^_]+)_sess?-d\d+_(?P<proto_part>.+)\.nii.gz", filename)
        
        if match:
            subject_id = match.group("sub_id")
            protocol_raw = match.group("proto_part")
            protocol_cleaned = re.sub(r"run-\d+_", "", protocol_raw)
            grouped_data[subject_id][protocol_cleaned].append(file_path)
    
    return grouped_data


def volume_intensity_difference(src_volume, tgt_volume):
    difference = np.abs(src_volume - tgt_volume)
    difference_mean = np.mean(difference)
    return difference_mean


def slice_intensity_difference(src_slice, tgt_slice):
    difference = np.abs(src_slice - tgt_slice)
    difference_mean = np.mean(difference)
    return difference_mean


def make_instruction(time_diff, rng=random):
    template = rng.choice(INSTRUCTION_TEMPLATES)
    instruction = template.format(time=time_diff)
    return instruction


def process_protocol_unit(idx: int):
    """Process a single subject-protocol unit."""
    if _WORKER_OUTPUT_DIR is None:
        raise RuntimeError("Worker not initialized")
    
    sub_id, proto, file_paths = _WORKER_TASKS[idx]
    rng = random.Random(_WORKER_SEED + idx)
    local_saved_slices = 0

    sorted_files = sorted(file_paths, key=extract_days)
    if len(sorted_files) < 2:
        return 0

    for input_file, label_file in combinations(sorted_files, 2):
        try:
            json_path = Path(str(input_file).replace(".nii.gz", ".json"))
            if not json_path.exists():
                continue

            json_text = json_path.read_text()

            src_ses = re.search(r"d\d+", input_file.name).group()
            tgt_ses = re.search(r"d\d+", label_file.name).group()

            src_data = nib.load(str(input_file))
            tgt_data = nib.load(str(label_file))

            src_shape = src_data.header.get_data_shape()
            tgt_shape = tgt_data.header.get_data_shape()

            src_volume = src_data.get_fdata()
            tgt_volume = tgt_data.get_fdata()

                
            if src_shape == tgt_shape and src_ses != tgt_ses and volume_intensity_difference(src_volume, tgt_volume) < 45:
                selected_slices = choose_slice_index(src_volume, num_slices=4, rng=rng)
                
                if not selected_slices:
                    continue

                for s in selected_slices:
                    src_slice = src_volume[:, :, s].astype(np.float32)
                    tgt_slice = tgt_volume[:, :, s].astype(np.float32)

                    if slice_intensity_difference(src_slice, tgt_slice) < 45:
                        time_diff = extract_days(label_file) - extract_days(input_file)
                        instruction = make_instruction(time_diff, rng=rng)
                        save_filename = f"{sub_id}_{proto}_{src_ses}_to_{tgt_ses}_slice{s:03d}.mat"
                        if (_WORKER_OUTPUT_DIR / save_filename).exists():
                            continue
                        mat_content = {
                            'image': src_slice,
                            'label': tgt_slice,
                            'instruction': np.array(instruction, dtype=object),
                            'text': np.array(json_text, dtype=object)
                        }
                        sio.savemat(_WORKER_OUTPUT_DIR / save_filename, mat_content)
                        local_saved_slices += 1
        
        except Exception as e:
            print(f"Error in {sub_id}/{proto}: {e}")
            
    return local_saved_slices


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    grouped_data = load_grouped_data(OASIS3_DIR)
    if not grouped_data:
        raise RuntimeError(f"No OASIS3 cases found under {OASIS3_DIR}")
    
    tasks = []
    for sub_id, protocols in grouped_data.items():
        for proto, file_paths in protocols.items():
            tasks.append((sub_id, proto, file_paths))
    
    total_tasks = len(tasks)
    print(f"Total tasks to process: {total_tasks}")
    
    total_saved_slices = 0
    with Pool(processes=WORKERS, initializer=_init_worker, initargs=(tasks, str(out_dir), SEED)) as pool:
        for done, result in enumerate(pool.imap_unordered(process_protocol_unit, range(total_tasks)), start=1):
            total_saved_slices += result
            print(f"\rProgress: [{done}/{total_tasks}] tasks completed. Total pairs: {total_saved_slices}", end="", flush=True)
        print("\n")
    print(f"Saved {total_saved_slices} samples to {out_dir.resolve()}")


if __name__ == "__main__":
    main()