import os
import wandb

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lotus.lifelong.metric import evaluate_multitask_training_success
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from lotus.lifelong.algos.base import Task
from lotus.lifelong.metric import *
from lotus.lifelong.models import *
from lotus.lifelong.utils import *

from lotus.lifelong.models.act.policy import ACTPolicy

from collections import deque


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.
    Adapted for backward compatibility from the torchvision library:
    https://pytorch.org/vision/0.14/generated/torchvision.ops.MLP.html

    LICENSE:

    From PyTorch:

    Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
    Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
    Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
    Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
    Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
    Copyright (c) 2011-2013 NYU                      (Clement Farabet)
    Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
    Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
    Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

    From Caffe2:

    Copyright (c) 2016-present, Facebook Inc. All rights reserved.

    All contributions by Facebook:
    Copyright (c) 2016 Facebook Inc.

    All contributions by Google:
    Copyright (c) 2015 Google Inc.
    All rights reserved.

    All contributions by Yangqing Jia:
    Copyright (c) 2015 Yangqing Jia
    All rights reserved.

    All contributions by Kakao Brain:
    Copyright 2019-2020 Kakao Brain

    All contributions by Cruise LLC:
    Copyright (c) 2022 Cruise LLC.
    All rights reserved.

    All contributions from Caffe:
    Copyright(c) 2013, 2014, 2015, the respective contributors
    All rights reserved.

    All other contributions:
    Copyright(c) 2015, 2016 the respective contributors
    All rights reserved.

    Caffe2 uses a copyright model similar to Caffe: each contributor holds
    copyright over their contributions to Caffe2. The project versioning records
    all such contribution and copyright details. If a contributor wants to further
    mark their specific copyright on a particular contribution, they should
    indicate their copyright solely in the commit message of the change when it is
    committed.

    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
    and IDIAP Research Institute nor the names of its contributors may be
    used to endorse or promote products derived from this software without
    specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.


    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels,
        activation_layer=torch.nn.ReLU,
        inplace=None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)


class MTACTtask(Task):
    def __init__(self, n_tasks, cfg):
        super().__init__(n_tasks, cfg)
        self.cfg = cfg
        self.experiment_dir = cfg.experiment_dir

        self.device = cfg.device
        self.use_tb = cfg.use_tb
        self.lr = cfg.lr
        self.num_queries = cfg.num_queries
        self.use_proprio = cfg.use_proprio
        self.pixel_keys = cfg.pixel_keys    # agentview_rgb
        self.proprio_key = cfg.proprio_key  # proprioceptive
        self.feature_key = cfg.feature_key

        action_shape = [7]
        episode_len = cfg.max_episode_len
        obs_shape = dict()
        # obs_shape['proprioceptive'] = [9]
        # obs_shape['agentview_rgb'] = [3, 128, 128]
        # obs_shape['eye_in_hand_rgb'] = [3, 128, 128]
        obs_shape['proprioceptive'] = [7]
        obs_shape['agentview_rgb'] = [3, 480, 640]
        obs_shape['eye_in_hand_rgb'] = [3, 480, 640]
        self.proprioceptive_dim = obs_shape[self.proprio_key][0] if cfg.use_proprio else 1  # 9
        self.multitask = cfg.multitask  # true
        self.obs_type = cfg.obs_type    # pixels

        self.language_dim = 768

        # Query frequency for evaluation
        # self.query_freq = 1 if self.temporal_agg else self.num_queries

        # policy config
        policy_config = {
            "lr": cfg.lr,
            "num_queries": cfg.num_queries, # 10
            "kl_weight": cfg.kl_weight,     # 10
            "hidden_dim": cfg.hidden_dim,   # 512
            "dim_feedforward": cfg.dim_feedforward,     # 3200
            "lr_backbone": cfg.lr_backbone,     # 1e-4
            "backbone": cfg.backbone,   # resnet18
            "enc_layers": cfg.enc_layers,   # 4
            "dec_layers": cfg.dec_layers,   # 1
            "nheads": cfg.nheads,           # 8
            "camera_names": cfg.pixel_keys,     # agentview_rgb, eye_in_hand_rgb
            "state_dim": self.proprioceptive_dim,
            "action_dim": action_shape[0],
            "multitask": self.multitask,
            "obs_type": self.obs_type,
            "temporal_agg": cfg.temporal_agg,
        }

        # actor
        self.policy = ACTPolicy(policy_config, self.device)

        # task_env projector
        # if self.multitask:
        #     self.language_projector = MLP(
        #         self.language_dim, hidden_channels=[cfg.hidden_dim, cfg.hidden_dim]
        #     ).to(self.device)
        #     self.language_projector.apply(utils.weight_init)

        # optimizers
        # self.optimizer = self.policy.configure_optimizers()
        # if self.multitask:
        #     self.optimizer.add_param_group(
        #         {"params": self.language_projector.parameters()}
        #     )

        self.reset()

    def __repr__(self):
        return "mtact"

    def reset(self):
        self.observations_buffer = {} if self.obs_type == "pixels" else deque(maxlen=1)

    def clear_buffers(self):
        del self.observations_buffer

    def observe(self, data):

        data = self.map_tensor_to_device(data)

        action = data["actions"].float()    # (bs, 10, 7)
        # if len(data["actions"].shape) == 4:
        #     action = action[:, 0]
        is_pad = torch.zeros(action.shape[0], action.shape[1], dtype=torch.bool).to(self.device)

        # lang projection
        # task_emb = data["task_emb"].float()     # (bs, 768)
        # task_emb = self.language_projector(task_emb)    # (bs, )

        observation = []
        for key in self.pixel_keys:
            observation.append(data["obs"][key].float())
        observation = torch.cat(observation, dim=1)

        joint_states = data["obs"]["joint_states"].float()  # (bs, 1, 7)
        # gripper_states = data["obs"]["gripper_states"].float()  # (bs, 1, 2)
        # proprio = torch.cat([joint_states, gripper_states], dim=-1)  # (bs, 1, 9) T=0
        proprio = joint_states  # (bs, 1, 7) T=0

        proprioceptive = (
            proprio
            if self.use_proprio
            else torch.zeros(observation.shape[0], 1, self.proprioceptive_dim)
            .to(self.device)
            .float()
        )

        # if self.obs_type == "pixels":
        #     observation = observation / 255.0
        proprioceptive = proprioceptive[:, 0]

        # forward pass
        output = self.policy(
            proprioceptive, observation, action, is_pad, task_emb=data["task_emb"]
            # proprioceptive, observation, action, is_pad, task_emb=None
        )

        # optimize
        self.optimizer.zero_grad(set_to_none=True)
        output["loss"].backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()

        metrics = dict()
        if self.use_tb:
            metrics["actor_loss"] = output["loss"].item()
            metrics["actor_l1"] = output["l1"].item()
            metrics["actor_kl"] = output["kl"].item()

        return metrics

    def learn_multi_task(self, datasets, benchmark):
        self.start_task(-1)

        concat_dataset = ConcatDataset(datasets)

        # learn on all tasks, only used in multitask learning
        model_checkpoint_name = os.path.join(self.experiment_dir, f"multitask_model.pth")
        all_tasks = list(range(benchmark.n_tasks))

        train_dataloader = DataLoader(
            concat_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=RandomSampler(concat_dataset),
            persistent_workers=True,
        )

        prev_success_rate = -1.0
        # best_state_dict = self.policy.state_dict()  # currently save the best model
        # for evaluate how fast the agent learns on current task, this corresponds
        # to the area under success rate curve on the new task.
        # cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        losses = []
        all_eval_successes = []

        for epoch in range(1, self.cfg.train.n_epochs + 1):
            t0 = time.time()
            self.policy.train()
            training_loss = 0.0
            for (idx, data) in enumerate(train_dataloader):

                data["obs"]["agentview_rgb"] = data["obs"]["agentview_rgb"][:, 0:1]
                # data["obs"]["eye_in_hand_rgb"] = data["obs"]["eye_in_hand_rgb"][:, 0:1]
                data["obs"]["joint_states"] = data["obs"]["joint_states"][:, 0:1]
                # data["obs"]["gripper_states"] = data["obs"]["gripper_states"][:, 0:1]

                loss = self.observe(data)["actor_loss"]
                training_loss += loss

            training_loss /= len(train_dataloader)

            t1 = time.time()

            print(
                f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.4f} | time: {(t1-t0)/60:4.2f}"
            )

            if epoch > 30 and epoch % self.cfg.eval.eval_every == 0:  # evaluate BC loss
                self.policy.eval()

                model_checkpoint_name_ep = os.path.join(self.experiment_dir, f"multitask_model_ep{epoch}.pth")
                print(f"[info] Save Model in {model_checkpoint_name_ep}")
                torch_save_model(self.policy, model_checkpoint_name_ep, cfg=self.cfg)
                losses.append(training_loss)

                # for multitask learning, we provide an option whether to evaluate
                # the agent once every eval_every epoch on all tasks, note that
                # this can be quite computationally expensive. Nevertheless, we
                # save the checkpoints, so users can always evaluate afterwards.
                if self.cfg.lifelong.eval_in_train:
                    success_rates = evaluate_multitask_training_success(
                        self.cfg, self, benchmark, all_tasks
                    )   # 每个任务的成功率
                    success_rate = np.mean(success_rates)   # 所有任务的平均成功率
                    print("success_rates: ", success_rates)
                    print("average success_rates: ", success_rate)
                else:
                    success_rates = np.zeros(len(all_tasks))
                    success_rate = 0.0
                successes.append(success_rate)
                all_eval_successes.append(success_rates)

                if prev_success_rate < success_rate:
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                    prev_success_rate = success_rate
                    idx_at_best_succ = len(losses) - 1

                # t1 = time.time()
                # cumulated_counter += 1.0
                # ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                # tmp_successes = np.array(successes)
                # tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]


        # load the best policy if there is any
        # if self.cfg.lifelong.eval_in_train:
        #     self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])
        # self.end_task(concat_dataset, -1, benchmark)

        # return the metrics regarding forward transfer
        losses = np.array(losses)   # 每次评估的所有任务的平均损失
        successes = np.array(successes)  # 每次评估的所有任务的平均成功率 [0.   , 0.725, 0.81 , 0.77 , 0.775, 0.785, 0.75 , 0.81 , 0.745, 0.78 , 0.745]
        all_eval_successes = np.array(all_eval_successes)   # 每次评估的每个任务的成功率 是一个矩阵
        auc_checkpoint_name = os.path.join(self.experiment_dir, f"multitask_auc.log")
        torch.save(
            {
                "success": successes,
                "loss": losses,
                "all_eval_successes": all_eval_successes,
            },
            auc_checkpoint_name,
        )

        print("best success index: ", idx_at_best_succ)
        #
        # if self.cfg.lifelong.eval_in_train:
        #     loss_at_best_succ = losses[idx_at_best_succ]
        #     success_at_best_succ = successes[idx_at_best_succ]
        #     losses[idx_at_best_succ:] = loss_at_best_succ
        #     successes[idx_at_best_succ:] = success_at_best_succ
        # return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter

    def learn_one_task(self, dataset, task_id, benchmark):
        print(f"[info] start train task {task_id}: {benchmark.get_task(task_id).language}")
        self.start_task(task_id)

        # recover the corresponding manipulation task ids
        gsz = self.cfg.data.task_group_size  # 1
        manip_task_ids = list(range(task_id * gsz, (task_id + 1) * gsz))

        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_model.pth"
        )

        train_dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=RandomSampler(dataset),
            persistent_workers=True,
        )

        prev_success_rate = -1.0
        best_state_dict = self.policy.state_dict()  # currently save the best model

        # for evaluate how fast the agent learns on current task, this corresponds
        # to the area under success rate curve on the new task.
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        losses = []
        T = 1

        task = benchmark.get_task(task_id)
        task_emb = benchmark.get_task_emb(task_id)

        # start training
        for epoch in range(1, self.cfg.train.n_epochs + 1):

            t0 = time.time()
            self.policy.train()
            training_loss = 0.0
            for (idx, data) in enumerate(train_dataloader):
                # data["obs"]["agentview_rgb"] = data["obs"]["agentview_rgb"][:, 0:T]  # (bs, T, 3, 480, 640)
                # data["obs"]["eye_in_hand_rgb"] = data["obs"]["eye_in_hand_rgb"][:, 0:T]
                # data["obs"]["joint_states"] = data["obs"]["joint_states"][:, 0:T]  # (bs, T, 7)
                # data["obs"]["gripper_states"] = data["obs"]["gripper_states"][:, 0:T]  # (bs, T, 2)

                loss = self.observe(data)["actor_loss"]
                training_loss += loss

            training_loss /= len(train_dataloader)

            t1 = time.time()

            print(
                f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1 - t0) / 60:4.2f}"
            )

            # if use_wandb:
            #     wandb.log({
            #         f"Training/task_{task_id}_training_loss": training_loss,
            #         f"Training/task_{task_id}_training_time": (t1 - t0) / 60,
            #         "Training/step": epoch,
            #     })

            if epoch > 30 and (epoch % self.cfg.eval.eval_every == 0):  # evaluate BC loss
                # every eval_every epoch, we evaluate the agent on the current task,
                # then we pick the best performant agent on the current task as
                # if it stops learning after that specific epoch. So the stopping
                # criterion for learning a new task is achieving the peak performance
                # on the new task. Future work can explore how to decide this stopping
                # epoch by also considering the agent's performance on old tasks.
                t0 = time.time()
                self.policy.eval()

                model_checkpoint_name_ep = os.path.join(self.experiment_dir, f"task{task_id}_model_ep{epoch}.pth")
                torch_save_model(self.policy, model_checkpoint_name_ep, cfg=self.cfg)
                losses.append(training_loss)

                if self.cfg.lifelong.eval_in_train:
                    task_str = f"k{task_id}_e{epoch // self.cfg.eval.eval_every}"
                    sim_states = (
                        result_summary[task_str] if self.cfg.eval.save_sim_states else None
                    )
                    success_rate = evaluate_one_task_success(
                        cfg=self.cfg,
                        algo=self,
                        task=task,
                        task_emb=task_emb,
                        task_id=task_id,
                        sim_states=sim_states,
                        task_str="",
                    )
                    successes.append(success_rate)
                    print(f"success_rate for task {task_id}: ", success_rate)
                else:
                    success_rate = 0.0

                if prev_success_rate < success_rate:
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                    prev_success_rate = success_rate
                    idx_at_best_succ = len(losses) - 1

                # t1 = time.time()
                #
                # cumulated_counter += 1.0
                # ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                # tmp_successes = np.array(successes)
                # tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]
                # print(
                #     f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} "
                #     + f"| succ. AoC {tmp_successes.sum() / cumulated_counter:4.2f} | time: {(t1 - t0) / 60:4.2f}",
                #     flush=True,
                # )
                # if use_wandb:
                #     wandb.log({
                #         f"Training/task_{task_id}_success_rate": success_rate,
                #         f"Training/task_{task_id}_best_success_rate": prev_success_rate,
                #         f"Training/task_{task_id}_AoC": tmp_successes.sum() / cumulated_counter,
                #         f"Training/task_{task_id}_eval_time": (t1 - t0) / 60,
                #         "Training/step": epoch,
                #     })

            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()

        # load the best performance agent on the current task
        # self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])

        # end learning the current task, some algorithms need post-processing
        # self.end_task(dataset, task_id, benchmark)

        # return the metrics regarding forward transfer
        losses = np.array(losses)
        successes = np.array(successes)
        auc_checkpoint_name = os.path.join(self.experiment_dir, f"task{task_id}_auc.log")
        torch.save(
            {
                "success": successes,
                "loss": losses,
            },
            auc_checkpoint_name,
        )

        # pretend that the agent stops learning once it reaches the peak performance
        # losses[idx_at_best_succ:] = losses[idx_at_best_succ]
        # successes[idx_at_best_succ:] = successes[idx_at_best_succ]
        # return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter