# -*- coding: utf-8 -*-
"""
Implementation of the ACIL [1] and the G-ACIL [2].
The G-ACIL is a generalization of the ACIL in the generalized setting.
For the popular setting, the G-ACIL is equivalent to the ACIL.

References:
[1] Zhuang, Huiping, et al.
    "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
    Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
[2] Zhuang, Huiping, et al.
    "G-ACIL: Analytic Learning for Exemplar-Free Generalized Class Incremental Learning"
    arXiv preprint arXiv:2403.15706 (2024).
"""
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from os import path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence
from abc import ABCMeta, abstractmethod


from lotus.lifelong.models import *
from lotus.lifelong.utils import *

import robomimic.utils.tensor_utils as TensorUtils
import time

# from utils import set_weight_decay
# from torch._prims_common import DeviceLikeType
# from .Buffer import RandomBuffer
# from torch.nn import DataParallel
# from .Learner import Learner, loader_t
# from .AnalyticLinear import AnalyticLinear, RecursiveLinear



# class Learner(metaclass=ABCMeta):
#     def __init__(
#         self,
#         args: Dict[str, Any],
#         backbone_output_size,
#         device=None,
#         # all_devices: Optional[Sequence[DeviceLikeType]] = None,
#     ):
#         self.args = args
#         self.backbone_output_size = backbone_output_size
#         self.device = device
#         self.model: torch.nn.Module
#
#     @abstractmethod
#     def base_training(
#         self,
#         train_loader,
#         val_loader,
#         baseset_size: int,
#     ):
#         raise NotImplementedError()
#
#     @abstractmethod
#     def learn(self, data_loader, phase,):
#         raise NotImplementedError()
#
#     @abstractmethod
#     def before_validation(self):
#         raise NotImplementedError()
#
#     @abstractmethod
#     def inference(self, X):
#         raise NotImplementedError()
#
#     def save_object(self, model, file_name):
#         torch.save(model, path.join(self.args["saving_root"], file_name))
#
#     def __call__(self, X):
#         return self.inference(X)

class ACILLearner:
    """
    This implementation is for the G-ACIL [2], a general version of the ACIL [1] that
    supports mini-batch learning and the general CIL setting.
    In the traditional CIL settings, the G-ACIL is equivalent to the ACIL.
    """

    def __init__(
        self,
        cfg,
        backbone_output_size,
        skill_policies=None,
        device=None,
    ):
        # super().__init__(args, backbone_output_size, device)
        self.backbone_output_size = backbone_output_size
        self.device = device
        self.cfg = cfg
        self.buffer_size = 8192
        self.gamma = 0.1
        # self.learning_rate = args["learning_rate"]  # 0.5
        # self.base_epochs = args["base_epochs"]
        # self.warmup_epochs = args["warmup_epochs"]

        self.loss_scale = cfg.train.loss_scale
        if not hasattr(cfg, "experiment_dir"):
            create_experiment_dir(cfg)
            print(
                f"[info] Experiment directory not specified. Creating a default one: {cfg.experiment_dir}"
            )
        self.experiment_dir = cfg.experiment_dir

        self.backbone = ACILTransformerPolicy(cfg=cfg,
                                         num_subtasks=cfg.skill_learning.num_subtasks,
                                         subgoal_embedding_dim=None,
                                         id_layer_dims=cfg.skill_learning.meta.id_layer_dims,
                                         embedding_layer_dims=cfg.skill_learning.meta.embedding_layer_dims,
                                         use_eye_in_hand=cfg.skill_learning.meta.use_eye_in_hand,
                                         activation=cfg.skill_learning.meta.activation,
                                         subgoal_type=cfg.skill_learning.skill_subgoal_cfg.subgoal_type,
                                         latent_dim=cfg.skill_learning.meta_cvae_cfg.latent_dim,
                                         policy_type=cfg.skill_learning.skill_training.policy_type,
                                         use_spatial_softmax=cfg.skill_learning.meta.use_spatial_softmax,
                                         num_kp=cfg.skill_learning.meta.num_kp,
                                         visual_feature_dimension=cfg.skill_learning.meta.visual_feature_dimension,
                                         kl_coeff=cfg.skill_learning.meta_cvae_cfg.kl_coeff,
                                         skill_policies=skill_policies).to(cfg.device)
        # self.make_model()

    def base_training(
        self,
        train_loader,
        val_loader,
        baseset_size,
    ):
        pass

    def make_model(self, out_features):
        self.policy = ACIL(
            self.backbone_output_size,
            self.backbone,
            self.buffer_size,
            out_features,
            self.gamma,
            device=self.device,
            dtype=torch.double,
        )

    @torch.no_grad()
    def learn(self, dataset):
        # self.model.eval()
        # for X, y in tqdm(data_loader, desc=desc):
        #     X = X.to(self.device, non_blocking=True)  # (4096, 64)
        #     y = y.to(self.device, non_blocking=True)  # (4096,)
        #     self.model.fit(X, y, increase_size=incremental_size)

        train_dataloader = DataLoader(
            dataset,
            # batch_size=self.cfg.train.batch_size,
            batch_size=128,
            num_workers=0, #self.cfg.train.num_workers,
            sampler=RandomSampler(dataset),
            # persistent_workers=True,
        )

        t0 = time.time()

        for (idx, data) in enumerate(train_dataloader):
            data = self.map_tensor_to_device(data)
            # print(data["obs"]["agentview_rgb"].shape)
            # print(data["obs"]["joint_states"].shape)
            # print(data["obs"]["gripper_states"].shape)
            # print(data["obs"]["id_vector"].shape)
            # print(data["obs"]["id"])
            self.policy.fit(data)

            t1 = time.time()
            print(f"[info] {idx} | Time: {(t1 - t0) / 60:4.2f}")
            # break

        model_checkpoint_name = os.path.join(self.experiment_dir, f"meta_controller_model.pth")
        torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)


    def before_validation(self):
        self.policy.update()

    def inference(self, X):
        return self.policy(X)


    #####################################
    ####         Sequential
    #####################################

    def start_task(self, task):
        """
        What the algorithm does at the beginning of learning each lifelong task.
        """
        self.current_task = task

        # initialize the optimizer and scheduler
        self.optimizer = eval(self.cfg.train.optimizer.name)(
            self.backbone.parameters(), **self.cfg.train.optimizer.kwargs
        )

        self.scheduler = None
        if self.cfg.train.scheduler is not None:
            self.scheduler = eval(self.cfg.train.scheduler.name)(
                self.optimizer,
                T_max=self.cfg.train.n_epochs,
                **self.cfg.train.scheduler.kwargs,
            )

    def map_tensor_to_device(self, data):
        """Move data to the device specified by self.cfg.device."""
        return TensorUtils.map_tensor(
            data, lambda x: safe_device(x, device=self.cfg.device)
        )

    def observe(self, data):
        """
        How the algorithm learns on each data point.
        """
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        loss = self.backbone.compute_loss(data)
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.backbone.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item()

    def learn_multi_task(self, dataset, benchmark, use_wandb):
        self.start_task(-1)

        # model_checkpoint_name = os.path.join(self.experiment_dir, f"meta_controller_model.pth")

        # TODO: using the first 5 tasks to train backbone
        all_tasks = list(range(benchmark.n_tasks))[:5]

        train_dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=0, #self.cfg.train.num_workers,
            sampler=RandomSampler(dataset),
            # persistent_workers=True,
        )

        prev_success_rate = -1.0
        best_state_dict = self.backbone.state_dict()  # currently save the best model

        # start training
        print(f"[info] start training backbone")
        prev_training_loss = None
        losses = []
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        all_eval_successes = []
        for epoch in range(1, self.cfg.train.n_epochs + 1):

            t0 = time.time()
            # if epoch > 0 or (self.cfg.pretrain):  # update
            self.backbone.train()
            training_loss = 0.0
            for (idx, data) in enumerate(train_dataloader):
                # print(data["obs"]["id"])
                # print(data["task_id"])
                loss = self.observe(data)
                training_loss += loss
                # break

            t1 = time.time()

            print(
                f"[info] Epoch: {epoch:3d} | Train loss: {training_loss:5.2f} | Time: {(t1-t0)/60:4.2f}"
            )


            if epoch > 0 and (epoch % self.cfg.eval.eval_every == 0):  # evaluate BC loss
                t0 = time.time()
                self.backbone.eval()
                model_checkpoint_name_ep = os.path.join(
                    self.experiment_dir, f"backbone_ep{epoch}.pth"
                )
                torch_save_model(self.backbone, model_checkpoint_name_ep, cfg=self.cfg)

                losses.append(training_loss)

            #     # for multitask learning, we provide an option whether to evaluate
            #     # the agent once every eval_every epochs on all tasks, note that
            #     # this can be quite computationally expensive. Nevertheless, we
            #     # save the checkpoints, so users can always evaluate afterwards.
            #     if self.cfg.lifelong.eval_in_train:
            #         success_rates = evaluate_multitask_training_success(self.cfg, self, benchmark, all_tasks)
            #         print("success_rate:", success_rates)
            #         success_rate = np.mean(success_rates)
            #         print("average success_rates: ", success_rate)
            #     else:
            #         success_rates = np.zeros(len(all_tasks))
            #         success_rate = 0.0
            #
            #     successes.append(success_rate)
            #     all_eval_successes.append(success_rates)
            #
            #     if prev_success_rate < success_rate and (not self.cfg.pretrain):
            #         torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
            #         prev_success_rate = success_rate
            #         idx_at_best_succ = len(losses) - 1
            #
            #     t1 = time.time()
            #
            #     cumulated_counter += 1.0
            #     ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
            #     tmp_successes = np.array(successes)
            #     tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]

            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()

        # load the best policy if there is any
        # if self.cfg.lifelong.eval_in_train:
        #     self.backbone.load_state_dict(torch_load_model(model_checkpoint_name)[0])

        # return the metrics regarding skill_training
        losses = np.array(losses)
        # successes = np.array(successes)
        # all_eval_successes = np.array(all_eval_successes)
        auc_checkpoint_name = os.path.join(
            self.experiment_dir, f"multitask_auc.log"
        )
        torch.save(
            {
                # "success": successes,
                # "all_eval_successes": all_eval_successes,
                "loss": losses,
            },
            auc_checkpoint_name,
        )

        # if self.cfg.lifelong.eval_in_train:
        #     loss_at_best_succ = losses[idx_at_best_succ]
        #     success_at_best_succ = successes[idx_at_best_succ]
        #     losses[idx_at_best_succ:] = loss_at_best_succ
        #     successes[idx_at_best_succ:] = success_at_best_succ
        # return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def get_feature(self, dataset):
        train_dataloader = DataLoader(
            dataset,
            # batch_size=self.cfg.train.batch_size,
            batch_size=1024,
            num_workers=0,  # self.cfg.train.num_workers,
            sampler=RandomSampler(dataset),
            # persistent_workers=True,
        )

        X = []
        Y = []
        cnt = 0
        for (idx, data) in enumerate(train_dataloader):
            data = self.map_tensor_to_device(data)
            # print(data["obs"]["agentview_rgb"].shape)
            # print(data["obs"]["joint_states"].shape)
            # print(data["obs"]["gripper_states"].shape)
            # print(data["obs"]["id_vector"].shape)
            # print(data["obs"]["id"])
            # self.policy.fit(data)
            Y.append(data["obs"]["id"].squeeze())  # (bs,)
            X.append(self.policy.feature_expansion(data))  # (bs, 8192)
            if cnt > 5:
                break
            else:
                cnt += 1


        # features = torch.cat(X, dim=0)
        # labels = torch.cat(Y, dim=0)
        features = torch.cat(X, dim=0).cpu().numpy()
        labels = torch.cat(Y, dim=0).cpu().numpy()

        print(features.shape)
        print(labels.shape)

        # torch.save(features, f'{self.experiment_dir}/features.pt')
        # torch.save(labels, f'{self.experiment_dir}/labels.pt')

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))

        colors = ['r', 'b', 'g', 'y', 'k', 'C0', 'C1', 'C2', 'magenta', 'lightpink', 'deepskyblue', 'lawngreen'] + list(mcolors.XKCD_COLORS.values())

        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label < len(colors):  # Ensure the label has a corresponding color
                indices = np.where(labels == label)
                plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=colors[label], alpha=0.8, label=f'Skill {label}')

        handles, legend_labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, legend_labels)  # Show custom legend
        plt.savefig(f"{self.experiment_dir}/tsne.png")

    # @torch.no_grad()
    # def wrap_data_parallel(self, model: torch.nn.Module) -> torch.nn.Module:
    #     if self.all_devices is not None and len(self.all_devices) > 1:
    #         return DataParallel(model, self.all_devices, output_device=self.device) # type: ignore
    #     return model


if __name__ == '__main__':
    backbone = nn.Sequential(*[nn.Linear(10, 10)])
    print(backbone)

    learner = ACILLearner(backbone, backbone_output=8192, device="cpu")
    print(learner.model)

    # learner.base_training()
    # for phase in range(0, args["phases"] + 1):
    #     if phase == 0:
    #         learner.learn(train_loader, dataset_train.base_size, "Re-align")
    #     else:
    #         learner.learn(train_loader, dataset_train.phase_size)
