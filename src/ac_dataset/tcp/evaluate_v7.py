"""Offline evaluation for V7 delay-compensated control supervision."""

from pathlib import Path

from .data_v7 import TCPDatasetV7
from .evaluate_v6 import main as evaluate_main


def main() -> None:
    evaluate_main(
        expected_recipe_version=7,
        default_prepared=Path("training/tcp_lagunaseca_v7"),
        default_checkpoint=Path("training/tcp_lagunaseca_v7/checkpoints/best.pt"),
        default_output=Path("training/tcp_lagunaseca_v7/evaluation"),
        metrics_filename="metrics_v7.json",
        dataset_class=TCPDatasetV7,
    )


if __name__ == "__main__":
    main()
