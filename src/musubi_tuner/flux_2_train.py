import argparse
import logging

from musubi_tuner import flux_2_train_network
from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer
from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.qwen_image_train import QwenImageTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Flux2Trainer(Flux2NetworkTrainer):
    def train(self, args):
        QwenImageTrainer.train(self, args)


def flux2_finetune_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--full_bf16", action="store_true", help="Enable full bfloat16 training for FLUX.2")
    parser.add_argument("--fused_backward_pass", action="store_true", help="Use fused backward pass for Adafactor optimizer")
    parser.add_argument(
        "--mem_eff_save",
        action="store_true",
        help=(
            "Enable memory efficient saving (saving states requires normal saving, "
            "so it takes the same amount of memory even with this option enabled)"
        ),
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = flux_2_train_network.flux2_setup_parser(parser)
    parser = flux2_finetune_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    if args.vae_dtype is None:
        args.vae_dtype = "float32"

    if args.model_version != "klein-base-4b":
        logger.warning(
            "FLUX.2 full fine-tuning has only been added for klein-base-4b. "
            "Other FLUX.2 variants may work but are not the primary supported target."
        )

    if args.fp8_base or args.fp8_scaled:
        logger.warning("FP8 training is not supported for FLUX.2 full fine-tuning. Set --fp8_base or --fp8_scaled to False.")
        args.fp8_base = False
        args.fp8_scaled = False

    args.dit_dtype = None

    trainer = Flux2Trainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
