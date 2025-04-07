import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
# libero_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# lifelong_dir = os.path.dirname(os.path.abspath(__file__))
# if libero_dir not in sys.path:
#     sys.path.insert(0, libero_dir)
# if lifelong_dir not in sys.path:
#     sys.path.insert(0, lifelong_dir)
import re
import json
import pprint
import time
from pathlib import Path

import hydra
import numpy as np
import wandb
import yaml
import torch
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import glob
import h5py
import init_path

from hil.libero import get_libero_path
from hil.libero.benchmark import get_benchmark
# from hil.lifelong.algos import get_algo_class, get_algo_list
from hil.lifelong.algos import *
from hil.lifelong.models import get_policy_list
from hil.lifelong.datasets import GroupedTaskDataset, SequenceVLDataset, get_dataset, SkillLearningDataset, MetaPolicyDataset, MetaPolicySequenceDataset
from hil.lifelong.metric import evaluate_loss, evaluate_success
from hil.lifelong.utils import (
    NpEncoder,
    compute_flops,
    control_seed,
    safe_device,
    torch_load_model,
    create_experiment_dir,
    get_task_embs,
)


from hil.lifelong.models.model_utils import safe_cuda
from hil.lifelong.models.conf_utils import *
# from hil.lifelong.models.torch_utils import *


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(hydra_cfg):
    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(cfg)
    #
    # pp.pprint("Available algorithms:")
    # pp.pprint(get_algo_list())
    #
    # pp.pprint("Available policies:")
    # pp.pprint(get_policy_list())

    # control seed
    control_seed(cfg.seed)

    # prepare lifelong learning
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_tasks = benchmark.n_tasks
    new_task_name = benchmark.new_task_name

    # prepare datasets from the benchmark
    manip_datasets = []
    descriptions = []
    task_ids = []
    shape_meta = None

    for i in range(n_tasks):
        # currently we assume tasks from same benchmark have the same shape_meta
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(i)),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
            )
            # shape_meta: {'ac_dim': 7, 'all_shapes': OrderedDict([('agentview_rgb', [3, 128, 128]), ('eye_in_hand_rgb', [3, 128, 128]), ('gripper_states', [2]), ('joint_states', [7])]),
            # 'all_obs_keys': ['agentview_rgb', 'eye_in_hand_rgb', 'gripper_states', 'joint_states'], 'use_images': True}
        except Exception as e:
            print(f"[error] failed to load task {i} name {benchmark.get_task_names()[i]}")
            print(f"[error] {e}")
        print(os.path.join(cfg.folder, benchmark.get_task_demonstration(i)))
        # add language to the vision dataset, hence we call vl_dataset
        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)
        task_ids.append(i)

    # save task_embs to file instead of computing it every time
    task_embs_dir = os.path.join('bert', benchmark.name)
    os.makedirs(task_embs_dir, exist_ok=True)  # 确保目录存在
    task_embs_file = os.path.join(task_embs_dir, 'task_embs.pt')

    if os.path.exists(task_embs_file):
        print(f"[info] Loading task embeddings from {task_embs_file}")
        task_embs = torch.load(task_embs_file)
    else:
        task_embs = get_task_embs(cfg, descriptions)  # (n_tasks, emb_dim)
        torch.save(task_embs, task_embs_file)

    benchmark.set_task_embs(task_embs)
    task_names = benchmark.get_task_names()

    datasets = [SequenceVLDataset(ds, emb, id) for (ds, emb, id) in zip(manip_datasets, task_embs, task_ids)]  # 把每个任务的数据集和语言嵌入打包成一个数据集
    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]

    # from torch.utils.data import DataLoader, ConcatDataset
    # concat_dataset = ConcatDataset(datasets)
    # dataloader = DataLoader(concat_dataset, batch_size=4, shuffle=True)
    # for (idx, data) in enumerate(dataloader):
    #     print(data["obs"]["agentview_rgb"].shape)
    #     print(data["obs"]["eye_in_hand_rgb"].shape)
    #     print(data["obs"]["gripper_states"].shape)
    #     print(data["obs"]["joint_states"].shape)
    #     break

    print("\n=================== Lifelong Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_tasks}")
    for i in range(n_tasks):
        print(f"    - Task {i+1}:")
        print(f"        {benchmark.get_task(i).language}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")

    create_experiment_dir(cfg, extra=cfg.exp+"_")
    cfg.shape_meta = shape_meta

    # result_summary = {
    #     "L_conf_mat": np.zeros((n_tasks, n_tasks)),  # loss confusion matrix
    #     "S_conf_mat": np.zeros((n_tasks, n_tasks)),  # success confusion matrix
    #     "L_fwd": np.zeros((n_tasks,)),  # loss AUC, how fast the agent learns
    #     "S_fwd": np.zeros((n_tasks,)),  # success AUC, how fast the agent succeeds
    # }

    ##TODO: moe policy training
    cfg.skill_learning.num_subtasks = n_tasks
    with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)

    # moe_policy = safe_device(MoeController(n_tasks, cfg), cfg.device)
    moe_policy = safe_device(Moetask(n_tasks, cfg), cfg.device)
    moe_policy.train()

    if cfg.pretrain_model_path != "":
        moe_policy.load_model(task_id=-1, experiment_dir=cfg.pretrain_model_path)

    s_fwd, l_fwd = moe_policy.learn_multi_task(datasets, benchmark, cfg.use_wandb)
    # result_summary["L_fwd"][-1] = l_fwd
    # result_summary["S_fwd"][-1] = s_fwd

    print("[info] finished learning\n")


if __name__ == '__main__':
    main()