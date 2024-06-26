# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import sys
import uuid
from pathlib import Path

sys.path.append("/p/home/ritwik/dev/xLT/")

import submitit
from omegaconf import OmegaConf

import main as runner
from xt.config import MainConfig, create_config


def parse_args():
    parser = argparse.ArgumentParser("Submitit for Segmentor Train+Val")
    parser.add_argument(
        "--nodes", default=1, type=int, help="Number of nodes to request"
    )
    parser.add_argument(
        "--timeout", default=10080, type=int, help="Duration of the job (min)"
    )
    parser.add_argument(
        "--job_dir",
        default="",
        type=str,
        help="Job dir. Leave empty for automatic.",
    )

    parser.add_argument(
        "--qos",
        default="qos",
        type=str,
        choices=("frontier", "standard"),
        help="Queue to use",
    )
    parser.add_argument(
        "--comment", default="", type=str, help="Comment to pass to scheduler"
    )
    parser.add_argument(
        "--account", default="", type=str, help="The Account string to use"
    )
    parser.add_argument(
        "--constraint",
        default="mla",
        type=str,
        help="Which Nautilus constraint to use",
    )
    parser.add_argument(
        "--config",
        default="config/base.yaml",
        type=str,
        help="Path to config file",
    )

    return parser.parse_args()


def get_shared_folder() -> Path:
    if Path("/p/app/projects/nga-frontier/xlt-runs/jobs/").is_dir():
        p = Path("/p/app/projects/nga-frontier/xlt-runs/jobs/")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, config):
        self.config = config
        print(self.config)

    def __call__(self):
        import sys

        sys.path.append("/p/home/ritwik/dev/xLT/")
        import main as runner

        self._setup_gpu_args()
        runner.main(self.config)

    def checkpoint(self):
        import os

        import submitit

        self.config.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.config)
        empty_trainer = type(self)(self.config)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import os
        from pathlib import Path

        import submitit

        job_env = submitit.JobEnvironment()
        dist_env = submitit.helpers.TorchDistributedEnvironment().export(
            set_cuda_visible_devices=False
        )
        self.config.output_dir = Path(
            str(self.config.output_dir).replace(
                "%j", f"{job_env.job_id}_{self.config.name}"
            )
        )
        self.config.model.resume = Path(
            str(self.config.model.resume).replace(
                "%j", f"{job_env.job_id}_{self.config.name}"
            )
        )

        # These are needed because submitit errors out otherwise.
        # I thought these were deprecated? Who knows.
        # os.environ["RANK"] = str(self.args.rank)
        # os.environ["WORLD_SIZE"] = str(self.args.world_size)

        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(
            f"World size: {dist_env.world_size}, Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}"
        )


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    if args.constraint == "mla":
        num_gpus_per_node = 4
    elif args.constraint == "viz":
        num_gpus_per_node = 1
    else:
        num_gpus_per_node = 0

    print(f"Num GPUs per node: {num_gpus_per_node}")

    nodes = args.nodes
    timeout_min = args.timeout
    qos = args.qos
    account = args.account
    constraint = args.constraint

    kwargs = {}
    if args.comment:
        kwargs["slurm_comment"] = args.comment

    executor.update_parameters(
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_qos=qos,
        slurm_signal_delay_s=120,
        slurm_account=account,
        slurm_constraint=constraint,
        **kwargs,
    )

    executor.update_parameters(name=os.environ["EXP_NAME"])

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir
    config = create_config(
        schema=OmegaConf.structured(MainConfig), cfg_path=args.config
    )
    print(OmegaConf.to_yaml(config))

    trainer = Trainer(config)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
