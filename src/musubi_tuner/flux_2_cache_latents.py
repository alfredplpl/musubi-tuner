from __future__ import annotations

import logging
import json
import os
import tempfile
from typing import List

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_latent_cache_flux_2
from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.flux_2 import flux2_models
import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents_flux_2(batch: List[ItemInfo]) -> tuple[torch.Tensor, List[List[np.ndarray]]]:
    # item.content: target image (H, W, C)
    # item.control_content: list of images (H, W, C), optional

    # Stack batch into target tensor (B,H,W,C) in RGB order and control images list of tensors (H, W, C)
    contents = []
    for item in batch:
        content = item.content
        content = content[0] if isinstance(content, list) else content  # (H, W, C)
        contents.append(torch.from_numpy(content))  # target image

    contents = torch.stack(contents, dim=0)  # B, H, W, C
    contents = contents.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    controls = []
    for item in batch:
        if item.control_content is not None and len(item.control_content) > 0:
            controls.append([torch.from_numpy(cc[..., :3]) for cc in item.control_content])  # ensure RGB, remove alpha if present

    if len(controls) > 0:  # controls is list of list of (H, W, C), where H, W can vary
        controls = [[c.permute(2, 0, 1) for c in cl] for cl in controls]  # list of list of (H, W, C) -> list of list of (C, H, W)
        controls = [[c / 127.5 - 1.0 for c in cl] for cl in controls]  # normalize to [-1, 1]
    else:
        controls = None

    return contents, controls


def encode_and_save_batch(ae: flux2_models.AutoEncoder, batch: List[ItemInfo], arch_full: str):
    # item.content: target image (H, W, C)
    # item.control_content: list of images (H, W, C)

    contents, controls = preprocess_contents_flux_2(batch)

    with torch.no_grad():
        latents = ae.encode(contents.to(ae.device, dtype=ae.dtype))  # B, C, H, W
        if controls is not None:
            control_latents = [[ae.encode(c.to(ae.device, dtype=ae.dtype).unsqueeze(0))[0] for c in cl] for cl in controls]
            # now control_latents is list of list of (C, H, W) tensors
        else:
            control_latents = None

    # save cache for each item in the batch
    for b, item in enumerate(batch):
        target_latent = latents[b]  # C, H, W. Target latents for this image (ground truth)
        control_latent = control_latents[b] if control_latents is not None else None  # list of (C, H, W) tensors or None

        print(
            f"Saving cache for item {item.item_key} at {item.latent_cache_path}, target latents shape: {target_latent.shape}, "
            f"control latents shape: {[cl.shape for cl in control_latent] if control_latent is not None else None}"
        )

        # save cache (file path is inside item.latent_cache_path pattern)
        save_latent_cache_flux_2(
            item_info=item,
            latent=target_latent,  # Ground truth for this image
            control_latent=control_latent,  # Control latent for this image
            arch_full=arch_full,
        )


def add_multi_gpu_args(parser):
    parser.add_argument(
        "--multi_gpu_devices",
        type=str,
        default=None,
        help=(
            "comma-separated GPU devices for parallel latent caching, e.g. '0,1' or 'cuda:0,cuda:1'. "
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


def load_datasets(args, architecture: str):
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=architecture)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    return train_dataset_group.datasets


def encode_datasets_shard(
    datasets,
    encode,
    args,
    rank: int,
    world_size: int,
    all_cache_paths_file_path: str | None = None,
):
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)
    all_latent_cache_paths = []

    for dataset_index, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{dataset_index}] on shard {rank}/{world_size}")
        batch_index = 0
        iterator = dataset.retrieve_latent_cache_batches(num_workers)
        if rank == 0:
            iterator = tqdm(iterator)

        for _, batch in iterator:
            batch = list(batch)
            for item in batch:
                if isinstance(item.content, np.ndarray):
                    if item.content.shape[-1] == 4:
                        item.content = item.content[..., :3]
                else:
                    item.content = [img[..., :3] if img.shape[-1] == 4 else img for img in item.content]

            all_latent_cache_paths.extend([item.latent_cache_path for item in batch])

            bs = args.batch_size if args.batch_size is not None else len(batch)
            for start in range(0, len(batch), bs):
                mini_batch = batch[start : start + bs]
                if batch_index % world_size != rank:
                    batch_index += 1
                    continue

                if args.skip_existing:
                    mini_batch = [item for item in mini_batch if not os.path.exists(item.latent_cache_path)]
                    if len(mini_batch) == 0:
                        batch_index += 1
                        continue

                encode(mini_batch)
                batch_index += 1

    if rank == 0 and all_cache_paths_file_path is not None:
        with open(all_cache_paths_file_path, "w", encoding="utf-8") as f:
            json.dump(all_latent_cache_paths, f)


def cleanup_old_latent_caches(datasets, args, all_latent_cache_paths: list[str]):
    all_latent_cache_paths = [os.path.normpath(p) for p in all_latent_cache_paths]
    all_latent_cache_paths = set(all_latent_cache_paths)

    for dataset in datasets:
        all_cache_files = dataset.get_all_latent_cache_files()
        for cache_file in all_cache_files:
            if os.path.normpath(cache_file) not in all_latent_cache_paths:
                if args.keep_cache:
                    logger.info(f"Keep cache file not in the dataset: {cache_file}")
                else:
                    os.remove(cache_file)
                    logger.info(f"Removed old cache file: {cache_file}")


def flux_2_cache_latents_worker(rank: int, world_size: int, devices: list[str], args, all_cache_paths_file_path: str):
    if args.disable_cudnn_backend:
        torch.backends.cudnn.enabled = False

    device = torch.device(devices[rank])
    model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]
    datasets = load_datasets(args, model_version_info.architecture)

    logger.info(f"Loading AE model from {args.vae} on {device}")
    vae_dtype = torch.float32 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    ae = flux2_utils.load_ae(args.vae, dtype=vae_dtype, device=device, disable_mmap=True)
    ae.to(device)

    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(ae, batch, model_version_info.architecture_full)

    encode_datasets_shard(datasets, encode, args, rank, world_size, all_cache_paths_file_path)


def encode_datasets_multi_gpu(args, devices: list[str]):
    assert args.vae is not None, "ae checkpoint is required"

    world_size = len(devices)
    logger.info(f"Using {world_size} GPUs for FLUX.2 latent caching: {', '.join(devices)}")
    ctx = mp.get_context("spawn")
    all_cache_paths_file = tempfile.NamedTemporaryFile(prefix="flux_2_cache_paths_", suffix=".json", delete=False)
    all_cache_paths_file_path = all_cache_paths_file.name
    all_cache_paths_file.close()
    processes = []

    for rank in range(world_size):
        process = ctx.Process(
            target=flux_2_cache_latents_worker,
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
        raise RuntimeError(f"One or more FLUX.2 latent cache workers failed: exit codes={exit_codes}")

    try:
        with open(all_cache_paths_file_path, "r", encoding="utf-8") as f:
            all_latent_cache_paths = json.load(f)

        if not args.keep_cache:
            model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]
            datasets = load_datasets(args, model_version_info.architecture)
            cleanup_old_latent_caches(datasets, args, all_latent_cache_paths)
    finally:
        if os.path.exists(all_cache_paths_file_path):
            os.remove(all_cache_paths_file_path)


def main():
    parser = cache_latents.setup_parser_common()
    flux2_utils.add_model_version_args(parser)
    add_multi_gpu_args(parser)

    args = parser.parse_args()
    model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    if args.debug_mode is not None:
        datasets = load_datasets(args, model_version_info.architecture)
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    if args.multi_gpu_devices is not None:
        devices = parse_multi_gpu_devices(args.multi_gpu_devices)
        encode_datasets_multi_gpu(args, devices)
        return

    # Load dataset config
    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    datasets = load_datasets(args, model_version_info.architecture)

    assert args.vae is not None, "ae checkpoint is required"

    logger.info(f"Loading AE model from {args.vae}")
    vae_dtype = torch.float32 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    ae = flux2_utils.load_ae(args.vae, dtype=vae_dtype, device=device, disable_mmap=True)
    ae.to(device)

    # encoding closure
    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(ae, batch, model_version_info.architecture_full)

    # reuse core loop from cache_latents with no change
    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
