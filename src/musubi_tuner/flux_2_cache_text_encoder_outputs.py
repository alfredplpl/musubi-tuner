from __future__ import annotations

import argparse
import json
import os
import tempfile

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer

from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_text_encoder_output_cache_flux_2

from musubi_tuner.flux_2 import flux2_utils
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(text_embedder: torch.nn.Module, batch: list[ItemInfo], device: torch.device, arch_full: str):
    prompts = [item.caption for item in batch]
    autocast_dtype = torch.bfloat16 if text_embedder.dtype.itemsize == 1 else text_embedder.dtype  # use bfloat16 for fp8 models
    with torch.autocast(device_type=device.type, dtype=autocast_dtype), torch.no_grad():
        ctx_vec = text_embedder(prompts)
        ctx_vec = ctx_vec.cpu()  # [1, 512, 15360]

    # save prompt cache
    for item, _ctx_vec in zip(batch, ctx_vec):
        save_text_encoder_output_cache_flux_2(item, _ctx_vec, arch_full=arch_full)


def add_multi_gpu_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--multi_gpu_devices",
        type=str,
        default=None,
        help=(
            "comma-separated GPU devices for parallel text encoder output caching, e.g. '0,1' or 'cuda:0,cuda:1'. "
            "If set, one worker process is launched per device."
        ),
    )
    return parser


def parse_multi_gpu_devices(devices: str) -> list[str]:
    parsed_devices = []
    for device in devices.split(","):
        device = device.strip()
        if not device:
            continue
        if device.isdigit():
            device = f"cuda:{device}"
        parsed_devices.append(device)
    if len(parsed_devices) == 0:
        raise ValueError("--multi_gpu_devices must contain at least one device")
    return parsed_devices


def load_datasets(args: argparse.Namespace, architecture: str):
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=architecture)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    return train_dataset_group.datasets


def process_text_encoder_batches_shard(
    num_workers: int | None,
    skip_existing: bool,
    batch_size: int | None,
    datasets,
    all_cache_files_for_dataset: list[set[str]],
    all_cache_paths_file_path: str | None,
    encode: callable,
    rank: int,
    world_size: int,
):
    num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() - 1)
    all_cache_paths_for_dataset = [set() for _ in datasets]

    for dataset_index, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{dataset_index}] on shard {rank}/{world_size}")
        all_cache_files = all_cache_files_for_dataset[dataset_index]
        all_cache_paths = all_cache_paths_for_dataset[dataset_index]
        batch_index = 0
        batches = dataset.retrieve_text_encoder_output_cache_batches(num_workers)
        if rank == 0:
            batches = tqdm(batches)

        for batch in batches:
            batch = list(batch)
            all_cache_paths.update([os.path.normpath(item.text_encoder_output_cache_path) for item in batch])

            bs = batch_size if batch_size is not None else len(batch)
            for start in range(0, len(batch), bs):
                mini_batch = batch[start : start + bs]
                if batch_index % world_size != rank:
                    batch_index += 1
                    continue

                if skip_existing:
                    mini_batch = [
                        item
                        for item in mini_batch
                        if os.path.normpath(item.text_encoder_output_cache_path) not in all_cache_files
                    ]
                    if len(mini_batch) == 0:
                        batch_index += 1
                        continue

                encode(mini_batch)
                batch_index += 1

    if rank == 0 and all_cache_paths_file_path is not None:
        serializable_paths = [list(paths) for paths in all_cache_paths_for_dataset]
        with open(all_cache_paths_file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_paths, f)


def flux_2_cache_text_encoder_outputs_worker(
    rank: int,
    world_size: int,
    devices: list[str],
    args: argparse.Namespace,
    all_cache_paths_file_path: str,
):
    device = torch.device(devices[rank])
    model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]
    datasets = load_datasets(args, model_version_info.architecture)
    all_cache_files_for_dataset, _ = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    logger.info(f"Loading text encoder from {args.text_encoder} on {device}")
    text_encoder_dtype = torch.float8_e4m3fn if args.fp8_text_encoder else torch.bfloat16
    text_embedder = flux2_utils.load_text_embedder(
        model_version_info, args.text_encoder, dtype=text_encoder_dtype, device=device, disable_mmap=True
    )

    def encode_for_text_encoder(batch: list[ItemInfo]):
        encode_and_save_batch(text_embedder, batch, device, model_version_info.architecture_full)

    process_text_encoder_batches_shard(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_file_path,
        encode_for_text_encoder,
        rank,
        world_size,
    )
    del text_embedder


def process_text_encoder_outputs_multi_gpu(args: argparse.Namespace, devices: list[str]):
    world_size = len(devices)
    logger.info(f"Using {world_size} GPUs for FLUX.2 text encoder output caching: {', '.join(devices)}")
    ctx = mp.get_context("spawn")
    all_cache_paths_file = tempfile.NamedTemporaryFile(prefix="flux_2_te_cache_paths_", suffix=".json", delete=False)
    all_cache_paths_file_path = all_cache_paths_file.name
    all_cache_paths_file.close()
    processes = []

    for rank in range(world_size):
        process = ctx.Process(
            target=flux_2_cache_text_encoder_outputs_worker,
            args=(rank, world_size, devices, args, all_cache_paths_file_path),
        )
        process.start()
        processes.append(process)

    exit_codes = []
    for process in processes:
        process.join()
        exit_codes.append(process.exitcode)

    if any(exit_code != 0 for exit_code in exit_codes):
        if os.path.exists(all_cache_paths_file_path):
            os.remove(all_cache_paths_file_path)
        raise RuntimeError(f"One or more FLUX.2 text encoder cache workers failed: exit codes={exit_codes}")

    try:
        model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]
        datasets = load_datasets(args, model_version_info.architecture)
        all_cache_files_for_dataset, _ = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

        with open(all_cache_paths_file_path, "r", encoding="utf-8") as f:
            all_cache_paths_for_dataset = [set(paths) for paths in json.load(f)]

        cache_text_encoder_outputs.post_process_cache_files(
            datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
        )
    finally:
        if os.path.exists(all_cache_paths_file_path):
            os.remove(all_cache_paths_file_path)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = flux_2_setup_parser(parser)
    add_multi_gpu_args(parser)

    args = parser.parse_args()
    model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]

    if args.multi_gpu_devices is not None:
        devices = parse_multi_gpu_devices(args.multi_gpu_devices)
        process_text_encoder_outputs_multi_gpu(args, devices)
        return

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    datasets = load_datasets(args, model_version_info.architecture)

    # Prepare existing cache files and expected cache paths for cleanup.
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    # Load Mistral 3 or Qwen-3 text encoder
    m3_dtype = torch.float8_e4m3fn if args.fp8_text_encoder else torch.bfloat16
    text_embedder = flux2_utils.load_text_embedder(
        model_version_info, args.text_encoder, dtype=m3_dtype, device=device, disable_mmap=True
    )

    # Encode with Mistral 3 or Qwen-3 text encoder
    logger.info("Encoding with text encoder")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal text_embedder
        encode_and_save_batch(text_embedder, batch, device, model_version_info.architecture_full)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )
    del text_embedder

    # remove cache files not in dataset
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


def flux_2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, default=None, required=True, help="text encoder (mistral 3) checkpoint path")
    parser.add_argument("--fp8_text_encoder", action="store_true", help="use fp8 for Text Encoder model")
    flux2_utils.add_model_version_args(parser)
    return parser


if __name__ == "__main__":
    main()
