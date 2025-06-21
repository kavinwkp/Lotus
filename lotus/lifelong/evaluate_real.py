import argparse
import sys
import os

import numpy
from sympy.physics.units import action

# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
import json
import numpy as np
import pprint
import time
import torch
import wandb
import yaml
from easydict import EasyDict
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModel, pipeline, AutoTokenizer, logging
from pathlib import Path

from lotus.libero import get_libero_path
from lotus.libero.benchmark import get_benchmark
from lotus.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from lotus.libero.utils.time_utils import Timer
from lotus.libero.utils.video_utils import VideoWriter
from lotus.lifelong.algos import *
from lotus.lifelong.datasets import get_dataset, SequenceVLDataset, GroupedTaskDataset
from lotus.lifelong.metric import (
    evaluate_loss,
    evaluate_success,
    raw_obs_to_tensor_obs,
)
from lotus.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    NpEncoder,
    compute_flops,
)

from lotus.lifelong.main import get_task_embs

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

import time
import h5py

# benchmark_map = {
#     "libero_10": "LIBERO_10",
#     "libero_spatial": "LIBERO_SPATIAL",
#     "libero_object": "LIBERO_OBJECT",
#     "libero_goal": "LIBERO_GOAL",
#     "libero_100": "LIBERO_100",
# }

algo_map = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
    "moe": "Moetask",
    "diffusion": "DiffusionTask"
}

policy_map = {
    "bc_rnn_policy": "BCRNNPolicy",
    "bc_transformer_policy": "BCTransformerPolicy",
    "bc_transformer_moe_policy": "BCTransformerMoEPolicy",
    "bc_vilt_policy": "BCViLTPolicy",
    "diffusion_policy": "BCDiffusionPolicy"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    # for which task suite
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        # choices=["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_100"],
    )
    parser.add_argument("--task_id", type=int, required=False)
    # method detail
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        # choices=["base", "er", "ewc", "packnet", "multitask", "moe"],
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        # choices=["bc_rnn_policy", "bc_transformer_policy", "bc_transformer_moe_policy", "bc_vilt_policy"],
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ep", type=int, default=50, help="epoch number of which .pth")
    parser.add_argument("--load_task", type=int, help="for single task")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--save-videos", action="store_true")
    # parser.add_argument('--save_dir',  type=str, required=True)
    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)
    args.save_dir = f"{args.experiment_dir}_saved"

    return args


def eval_sim(args):
    control_seed(args.seed)

    file_path = 'lotus/datasets/real_10/pickup_cube_demo.hdf5'

    with h5py.File(file_path, 'r') as file:

        # for i in range(len(file['data'].keys())):
        i = 1
        # print(demo)
        actions_gt = file[f"data/demo_{i}/actions"][()]
        # print(actions.shape)
        agentview_rgb = file[f"data/demo_{i}/obs/agentview_rgb"][()]
        # print(agentview_rgb.shape)
        eye_in_hand_rgb = file[f"data/demo_{i}/obs/eye_in_hand_rgb"][()]
        # print(eye_in_hand_rgb.shape)
        joint_states = file[f"data/demo_{i}/obs/joint_states"][()]
        # print(joint_states.shape)
        # break

    # e.g., experiments/LIBERO_SPATIAL/Multitask/BCRNNPolicy_seed100/

    # experiment_dir = os.path.join(
    #     args.experiment_dir,
    #     f"{benchmark_map[args.benchmark]}/"
    #     + f"{algo_map[args.algo]}/"
    #     + f"{policy_map[args.policy]}_seed{args.seed}",
    # )
    # experiment_dir = os.path.join(
    #     args.experiment_dir,
    #     f"{args.benchmark}/"
    #     + f"{algo_map[args.algo]}/"
    #     + f"{policy_map[args.policy]}_seed{args.seed}",
    # )
    # experiment_dir = args.experiment_dir

    # find the checkpoint
    # experiment_id = 0
    # import ipdb; ipdb.set_trace()
    # for path in Path(experiment_dir).glob("run_*"):
    #     if not path.is_dir():
    #         continue
    #     try:
    #         folder_id = int(str(path).split("run_")[-1])
    #         if folder_id > experiment_id:
    #             experiment_id = folder_id
    #     except BaseException:
    #         pass
    # if experiment_id == 0:
    #     print(f"[error] cannot find the checkpoint under {experiment_dir}")
    #     sys.exit(0)

    # run_folder = os.path.join(experiment_dir, f"run_{experiment_id:03d}")
    run_folder = args.experiment_dir
    train_task = "multitask"
    # train_task = "singletask"
    try:
        if train_task == "multitask":
            model_path = os.path.join(run_folder, f"multitask_model_ep{args.ep}.pth")
        else:
            model_path = os.path.join(run_folder, f"task{args.task_id}_model_ep{args.ep}.pth")
        sd, cfg, previous_mask = torch_load_model(model_path, map_location=args.device_id)
    except:
        print(f"[error] cannot find the checkpoint at {str(model_path)}")
        sys.exit(0)

    cfg.folder = get_libero_path("datasets")
    # cfg.bddl_folder = get_libero_path("bddl_files")
    # cfg.init_states_folder = get_libero_path("init_states")

    cfg.device = args.device_id
    cfg.temporal_agg = True     # TODO: temporal_agg
    algo = safe_device(eval(algo_map[args.algo])(10, cfg), cfg.device)
    algo.policy.previous_mask = previous_mask

    algo.policy.load_state_dict(sd)

    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0

    # get the benchmark the task belongs to
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]

    # task_embs = get_task_embs(cfg, descriptions)
    task_embs_dir = os.path.join('bert', benchmark.name)
    os.makedirs(task_embs_dir, exist_ok=True)
    task_embs_file = os.path.join(task_embs_dir, 'task_embs.pt')

    if os.path.exists(task_embs_file):
        print(f"[info] Loading task embeddings from {task_embs_file}")
        task_embs = torch.load(task_embs_file)
    else:
        task_embs = get_task_embs(cfg, descriptions)  # (n_tasks, emb_dim)
        torch.save(task_embs, task_embs_file)
    benchmark.set_task_embs(task_embs)

    task = benchmark.get_task(args.task_id)

    ### ======================= start evaluation ============================

    # 1. evaluate dataset loss
    try:
        dataset, shape_meta = get_dataset(
            dataset_path=os.path.join(
                cfg.folder, benchmark.get_task_demonstration(args.task_id)
            ),
            obs_modality=cfg.data.obs.modality,
            initialize_obs_utils=True,
            seq_len=cfg.data.seq_len,
        )
        dataset = GroupedTaskDataset(
            [dataset], task_embs[args.task_id : args.task_id + 1]
        )
    except:
        print(f"[error] failed to load task {args.task_id} name {benchmark.get_task_names()[args.task_id]}")
        sys.exit(0)

    algo.eval()

    test_loss = 0.0

    # 2. evaluate success rate
    video_folder = os.path.join(
        args.save_dir,
        f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_on_task{args.task_id}_videos",
    )

    save_folder = os.path.join(
        video_folder,
        f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_ep{args.ep}_on{args.task_id}.stats",
    )

    algo.reset()
    steps = 0
    task_emb = benchmark.get_task_emb(args.task_id)

    num_success = 0

    actions_output = []

    obs = [{}]

    with torch.no_grad():
        # while steps < cfg.eval.max_steps:  # 600
        while steps < 240:  # 600

            if steps % 10 == 0:
                print(steps)

            obs[0]["agentview_image"] = agentview_rgb[steps]
            obs[0]["robot0_eye_in_hand_image"] = eye_in_hand_rgb[steps]
            obs[0]["robot0_joint_pos"] = joint_states[steps]

            data = raw_obs_to_tensor_obs(obs, task_emb, cfg, args.task_id)
            actions = algo.policy.get_action(data)
            # print(actions.shape)  # (1,7)

            actions_output.append(actions[0])
            # obs, reward, done, info = env.step(actions)
            steps += 1

    actions_output = numpy.array(actions_output)

    import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    rows = 2
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))

    for i in range(actions_output.shape[1]):
        row = i // cols
        col = i % cols
        if row == 1 and col == 3:
            break
        ax = axes[row, col]
        ax.plot(actions_output[:steps, i], label=f'{i}')
        ax.plot(actions_gt[:steps, i], label=f'label {i}')
        ax.legend()
        ax.grid(True)

    plt.show()

    # with Timer() as t, VideoWriter(video_folder, args.save_videos) as video_writer:
    #     env_args = {
    #         "bddl_file_name": os.path.join(
    #             cfg.bddl_folder, task.problem_folder, task.bddl_file
    #         ),
    #         "camera_heights": cfg.data.img_h,
    #         "camera_widths": cfg.data.img_w,
    #     }
    #
    #     env_num = 20
    #     env = SubprocVectorEnv(
    #         [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
    #     )
    #     env.reset()
    #     env.seed(cfg.seed)
    #     algo.reset()
    #
    #     init_states_path = os.path.join(
    #         cfg.init_states_folder, task.problem_folder, task.init_states_file
    #     )
    #     init_states = torch.load(init_states_path)
    #     indices = np.arange(env_num) % init_states.shape[0]
    #     init_states_ = init_states[indices]
    #
    #     dones = [False] * env_num
    #     steps = 0
    #     obs = env.set_init_state(init_states_)
    #     task_emb = benchmark.get_task_emb(args.task_id)
    #
    #     num_success = 0
    #
    #     experts_cnt = [0, 0, 0, 0]
    #
    #     for _ in range(5):  # simulate the physics without any actions
    #         env.step(np.zeros((env_num, 7)))
    #     with torch.no_grad():
    #         while steps < cfg.eval.max_steps:   # 600
    #             steps += 1
    #
    #             data = raw_obs_to_tensor_obs(obs, task_emb, cfg, args.task_id)
    #             actions = algo.policy.get_action(data)
    #             obs, reward, done, info = env.step(actions)
    #
    #             video_writer.append_vector_obs(
    #                 obs, dones, camera_name="agentview_image"
    #             )
    #
    #             # check whether succeed
    #             for k in range(env_num):
    #                 dones[k] = dones[k] or done[k]
    #             if all(dones):
    #                 break
    #
    #         for k in range(env_num):
    #             num_success += int(dones[k])
    #
    #     success_rate = num_success / env_num
    #     env.close()
    #
    #     eval_stats = {
    #         "loss": test_loss,
    #         "success_rate": success_rate,
    #     }
    #
    #     os.system(f"mkdir -p {args.save_dir}")
    #     torch.save(eval_stats, save_folder)
    #
    #     os.makedirs(video_folder, exist_ok=True)
    #     with open(save_folder, "w") as f:
    #         json.dump(eval_stats, f, cls=NpEncoder, indent=4)

    # print(f"[info] finish for ckpt at {run_folder} in {t.get_elapsed_time()} sec for rollouts")
    # print(f"Results are saved at {save_folder}")
    # print("success_rate: ", success_rate)
    # return success_rate


if __name__ == "__main__":
    args = parse_args()
    eval_sim(args)




    # all_success = []
    # all_avg_success = []
    # eps = []
    #
    # for ep in range(40, 160, 10):
    #     success_ep = []
    #     for i in range(10):
    #         args.task_id = i
    #         args.ep = ep
    #         success_ep.append(main(args))
    #     eps.append(ep)
    #     avg_success_ep = sum(success_ep) / len(success_ep)
    #     print(f"[info] success_rate: {success_ep}")
    #     print(f"[info] average success_rate: {avg_success_ep}")
    #     all_success.append(success_ep)
    #     all_avg_success.append(avg_success_ep)
    # print(f"[info] Epoch: {eps}")
    # print(f"[info] All epoch success_rate: {np.array(all_success)}")
    # print(f"[info] All epoch average success_rate: {all_avg_success}")

    # success_ep = []
    # for i in range(0, 10):
    #     args.task_id = i
    #     main(args)
    #
    # avg_success_ep = sum(success_ep) / len(success_ep)
    # print(f"[info] success_rate: {success_ep}")
    # print(f"[info] average success_rate: {avg_success_ep}")

