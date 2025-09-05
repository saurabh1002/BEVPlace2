import argparse
from pathlib import Path

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Evaluation
eval_arg = add_argument_group("Eval")
eval_arg.add_argument(
    "--checkpoint", type=str, default="runs/Aug08_10-17-29/model_best.pth.tar"
)
eval_arg.add_argument(
    "--thresh_min", default=0.0, type=float, help="Thresholds on euclidean-distance."
)
eval_arg.add_argument(
    "--thresh_max", default=1.0, type=float, help="Thresholds on euclidean-distance."
)
eval_arg.add_argument(
    "--num_thresholds",
    default=20,
    type=int,
    help="Number of thresholds. Number of points on PR curve.",
)


# Dataset specific configurations
data_arg = add_argument_group("Data")

data_arg.add_argument(
    "--eval_feature_distance", type=str, default="euclidean"
)  # cosine#euclidean

data_arg.add_argument("--dataloader", type=str, required=True, help="Dataloader name")
data_arg.add_argument(
    "--data_dir", type=Path, required=True, help="Path to the dataset"
)
data_arg.add_argument("--sequence", type=str, default=None, help="Sequence Name")
data_arg.add_argument(
    "--results_dir", type=Path, required=True, help="Path to the results directory"
)


def get_config_eval():
    args = parser.parse_args()
    return args
