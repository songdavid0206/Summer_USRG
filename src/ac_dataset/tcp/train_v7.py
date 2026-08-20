"""V7 training entry point: V6 policy with measured t+50ms control targets."""

from pathlib import Path

from .data_v7 import TCPDatasetV7
from .train_v6 import main as train_main


def main() -> None:
    train_main(
        recipe_version=7,
        default_prepared=Path("training/tcp_lagunaseca_v7"),
        default_output=Path("training/tcp_lagunaseca_v7/checkpoints"),
        description="Train TCP-AC v7 with t+50ms delay-compensated control labels",
        dataset_class=TCPDatasetV7,
    )


if __name__ == "__main__":
    main()
