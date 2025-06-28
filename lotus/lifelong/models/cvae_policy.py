import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torchvision
import numpy as np
from lotus.skill_learning.models.model_utils import safe_cuda
from lotus.skill_learning.models.torch_utils import to_onehot
from lotus.skill_learning.models.resnet_model_utils import resnet18 as no_pool_resnet18
import torch.nn.functional as F
import torch.distributions as D
from enum import Enum
from lotus.lifelong.models.modules.rgb_modules import *
from lotus.lifelong.models.modules.language_modules import *
from lotus.lifelong.models.modules.transformer_modules import *
from lotus.lifelong.models.base_policy import BasePolicy
from lotus.lifelong.models.policy_head import *
from lotus.lifelong.models.modules.data_augmentation import (
    IdentityAug,
    TranslationAug,
    ImgColorJitterAug,
    ImgColorJitterGroupAug,
    BatchWiseImgColorJitterAug,
    DataAugGroup,
)



class PerceptionEmbedding(torch.nn.Module):
    def __init__(self,
                 use_eye_in_hand=True,
                 pretrained=False,
                 no_training=False,
                 activation='relu',
                 remove_layer_num=2,
                 img_c=3):

        super().__init__()
        if use_eye_in_hand:
            print("Using Eye in Hand !!!!!")            
            img_c = img_c + 3

        # For training policy
        layers = list(torchvision.models.resnet18(pretrained=pretrained).children())[:-remove_layer_num]
        if img_c != 3:
            # If use eye_in_hand, we need to increase the channel size
            conv0 = torch.nn.Conv2d(img_c, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            layers[0] = conv0

        self.resnet18_embeddings = torch.nn.Sequential(*layers)

        if no_training:
            for param in self.resnet18_embeddings.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.resnet18_embeddings(x)
        return h

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, in_c, in_h, in_w, num_kp=None):
        super().__init__()
        self._spatial_conv = torch.nn.Conv2d(in_c, num_kp, kernel_size=1)

        # pos_x, pos_y = torch.meshgrid(torch.from_numpy(np.linspace(-1, 1, in_w)).float(),
        #                               torch.from_numpy(np.linspace(-1, 1, in_h)).float())

        pos_x, pos_y = torch.meshgrid(torch.from_numpy(np.linspace(-1, 1, in_w)).float(),
                                      torch.from_numpy(np.linspace(-1, 1, in_h)).float(),
                                      indexing='xy')

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h
        
    def forward(self, x):
        # assert(x.shape[1] == self._in_c)
        # assert(x.shape[2] == self._in_h)
        # assert(x.shape[3] == self._in_w)

        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.view(-1, self._in_h * self._in_w)
        attention = F.softmax(h, dim=-1)

        keypoint_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True).view(-1, self._num_kp)
        keypoint_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True).view(-1, self._num_kp)

        keypoints = torch.cat([keypoint_x, keypoint_y], dim=1)
        return keypoints

class MaskingLayer(torch.nn.Module):
    def __init__(self, max_subtasks_num, num_subtasks):
        super(MaskingLayer, self).__init__()
        self.max_subtasks_num = max_subtasks_num
        self.num_subtasks = num_subtasks
        self.register_buffer('mask', torch.cat([torch.ones(num_subtasks), -1e10*torch.ones(max_subtasks_num - num_subtasks)]))

    def forward(self, x):
        return x + self.mask


class MetaIdLayer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_subtasks,
                 max_subtasks_num,
                 id_layer_dims,
                 policy_type,
                 subgoal_type="sigmoid",
                 activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.num_subtasks = num_subtasks
        self.max_subtasks_num = max_subtasks_num
        self.policy_type = policy_type

        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU

        self._id_layers = [torch.nn.Linear(input_dim, id_layer_dims[0]),
                        activate_fn()]
        for i in range(1, len(id_layer_dims)):
            self._id_layers += [torch.nn.Linear(id_layer_dims[i-1], id_layer_dims[i]),
                            activate_fn()]

        self._id_layers += [torch.nn.Linear(id_layer_dims[-1], max_subtasks_num),
                            MaskingLayer(max_subtasks_num, num_subtasks),
                            torch.nn.Softmax(dim=1)]
                            # output logits

        # self._id_layers += [torch.nn.Linear(id_layer_dims[-1], num_subtasks),
        #                  torch.nn.Softmax(dim=1)]
        self.id_layers = torch.nn.Sequential(*self._id_layers)

    def forward(self, x):
        return self.id_layers(x)

class MetaEmbeddingLayer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_subtasks,
                 subgoal_embedding_dim,
                 embedding_layer_dims,
                 policy_type,
                 subgoal_type="sigmoid",
                 activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.num_subtasks = num_subtasks
        self.subgoal_embedding_dim = subgoal_embedding_dim
        self.policy_type = policy_type

        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU

        self._embedding_layers = [torch.nn.Linear(input_dim, embedding_layer_dims[0]),
                        activate_fn()]
        for i in range(1, len(embedding_layer_dims)):
            self._embedding_layers += [torch.nn.Linear(embedding_layer_dims[i-1], embedding_layer_dims[i]),
                            activate_fn()]

        self._embedding_layers += [torch.nn.Linear(embedding_layer_dims[-1], subgoal_embedding_dim)]
        if self.policy_type == "normal_subgoal":
            if subgoal_type == "sigmoid":
                print("Using sigmoid")
                self._embedding_layers.append(torch.nn.Sigmoid())
        self.embedding_layers = torch.nn.Sequential(*self._embedding_layers)

    def forward(self, x):
        return self.embedding_layers(x)

class MetaCVAEPolicy(torch.nn.Module):
    def __init__(self,
                 cfg,
                 num_subtasks,
                 subgoal_embedding_dim,
                 id_layer_dims,
                 embedding_layer_dims,
                 use_eye_in_hand,
                 policy_type,
                 activation='relu',
                 subgoal_type="sigmoid",
                 latent_dim=32,
                 use_spatial_softmax=False,
                 num_kp=64,
                 visual_feature_dimension=64,
                 separate_id_prediction=False,
                 kl_coeff=1.0,
                 skill_policies=None,
    ):
        super().__init__()

        # add data augmentation for rgb inputs
        self.cfg = cfg
        meta_cfg = cfg.meta
        shape_meta = cfg.shape_meta
        color_aug = eval(meta_cfg.color_aug.network)(
            **meta_cfg.color_aug.network_kwargs
        )

        meta_cfg.translation_aug.network_kwargs["input_shape"] = shape_meta[
            "all_shapes"
        ][cfg.data.obs.modality.rgb[0]]
        translation_aug = eval(meta_cfg.translation_aug.network)(
            **meta_cfg.translation_aug.network_kwargs
        )
        self.img_aug = DataAugGroup((color_aug, translation_aug))

        self.skill_policies = skill_policies

        self.num_subtasks = num_subtasks
        self.subgoal_embedding_dim = subgoal_embedding_dim
        self.kl_coeff = kl_coeff
        self.freq = cfg.skill_learning.eval.meta_freq

        self.use_spatial_softmax = use_spatial_softmax
        if use_spatial_softmax:
            remove_layer_num = 2
        else:
            remove_layer_num = 1
        
        self.perception_encoder_layer = PerceptionEmbedding(use_eye_in_hand=use_eye_in_hand,
                                                              activation=activation,
                                                              remove_layer_num=remove_layer_num)

        if use_spatial_softmax:
            in_c = 512
            in_h = 4
            in_w = 4
            self._spatial_softmax_layers = torch.nn.Sequential(*[SpatialSoftmax(in_c=in_c,
                                                                                in_h=in_h,
                                                                                in_w=in_w,
                                                                                num_kp=num_kp),
                                                                 torch.nn.Linear(num_kp * 2, visual_feature_dimension)])

        if use_spatial_softmax:
            print("Using Spatial softmax for meta policy")
            input_dim = visual_feature_dimension
        else:
            input_dim = 512

        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU
        
        self.latent_dim = latent_dim
        id_dim = 0

        intermediate_state_dim = 256

        ### encode language
        self.task_embedding_dim = 32 #32
        meta_cfg.language_encoder.network_kwargs.output_size = self.task_embedding_dim
        self.language_encoder = eval(meta_cfg.language_encoder.network)(
            **meta_cfg.language_encoder.network_kwargs
        )
        
        self.mlp_encoder_layer = torch.nn.Sequential(*[torch.nn.Linear(input_dim + subgoal_embedding_dim + id_dim + self.task_embedding_dim, intermediate_state_dim),
                                                              activate_fn(),
                                                              torch.nn.Linear(intermediate_state_dim, intermediate_state_dim),
                                                       activate_fn(),
                                                       torch.nn.Linear(intermediate_state_dim, self.latent_dim * 2)])
        id_input_dim = input_dim + self.task_embedding_dim
        embedding_input_dim = input_dim + self.latent_dim + id_dim + self.task_embedding_dim

        self.meta_id_layer = MetaIdLayer(id_input_dim,
                                         num_subtasks,
                                         max_subtasks_num=20, #10,
                                         id_layer_dims=id_layer_dims,
                                         policy_type=policy_type,
                                         subgoal_type=subgoal_type,
                                         activation=activation)
        self.meta_embedding_layer = MetaEmbeddingLayer(embedding_input_dim,
                                                       num_subtasks,
                                                       subgoal_embedding_dim,
                                                       embedding_layer_dims=embedding_layer_dims,
                                                       policy_type=policy_type,
                                                       subgoal_type=subgoal_type,
                                                       activation=activation)

    def forward(self, x):
        # state_image:(100, 3, 128, 128); embedding:(100, 32); id_vector:(100, 6)
        z_state = self.perception_encoder_layer(x["agentview_rgb"])
        if self.use_spatial_softmax:
            z_state = self._spatial_softmax_layers(z_state)
        else:
            z_state = torch.flatten(z_state, start_dim=1)
        # if use task_embed
        task_emb = self.language_encoder(x) # 32
        z_state = torch.cat([z_state, task_emb], dim=1) # (Batch, 512 + 32)

        h = torch.cat([z_state, x["embedding"]], dim=1) # (Batch, 512(512+32) + 32)
        h = self.mlp_encoder_layer(h) # (Batch, 128)

        mu, logvar = torch.split(h, self.latent_dim, dim=1)
        z = self.sampling(mu, logvar) # (Batch, 64)
        z_concat = torch.cat([z_state, z], dim=1) # (Batch, 512(512+32) + 64)
        # z_concat = torch.cat([z_concat, x["id_vector"]], dim=1) # (Batch, 512(512+32) + 64 + 6)
        skill_id = self.meta_id_layer(z_state) # (Batch, 6)
        embedding = self.meta_embedding_layer(z_concat)
        return skill_id, embedding, mu, logvar


    def predict(self, x):
        z_state = self.perception_encoder_layer(x["agentview_rgb"])
        if self.use_spatial_softmax:
            z_state = self._spatial_softmax_layers(z_state)
        else:
            z_state = torch.flatten(z_state, start_dim=1)
        
        # if use task_embed
        task_emb = self.language_encoder(x) # 32
        z_state = torch.cat([z_state, task_emb], dim=1) # (Batch, 512 + 32)
        
        z = safe_cuda(torch.randn(x["agentview_rgb"].shape[0], self.latent_dim))
        # subtask_vector, subgoal_embedding = self.meta_decision_layer(torch.cat([z_state, z], dim=1))
        subtask_vector = self.meta_id_layer(z_state)
        subtask_id = torch.argmax(subtask_vector, dim=1)
        z_concat = [z_state, z]

        subgoal_embedding = self.meta_embedding_layer(torch.cat(z_concat, dim=1))

        subtask_id = subtask_id.cpu().detach().numpy()
        return {"subtask_id": subtask_id,
                "subtask_vector": subtask_vector,
                "embedding": subgoal_embedding}

    def sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps * std + mu
    
    def compute_loss(self, data, reduction="mean"):
        data = self.preprocess_input(data, train_mode=True)
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction=reduction)
        mse_loss = torch.nn.MSELoss(reduction=reduction)
        subtask_id_prediction, subgoal_embedding, mu, logvar = self.forward(data["obs"])

        ce_loss = cross_entropy_loss(subtask_id_prediction.view(-1, self.num_subtasks), data["obs"]["id"].view(-1))
        embedding_loss = mse_loss(data["obs"]["embedding"], subgoal_embedding)
        kl_loss = - self.kl_coeff * 400 * torch.mean(0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2), dim=1), dim=0)
        # run1:40; run2:400; run3:4; run4:4000

        loss = ce_loss + embedding_loss + kl_loss

        return loss, ce_loss, embedding_loss, kl_loss

    def _get_img_tuple(self, data):
        # img_tuple = tuple(
        #     [data["obs"][img_name] for img_name in self.image_encoders.keys()]
        # )
        img_tuple = tuple(
            [data["obs"]["agentview_rgb"].unsqueeze(1)]
        )
        return img_tuple

    def _get_aug_output_dict(self, out):
        # img_dict = {
        #     img_name: out[idx]
        #     for idx, img_name in enumerate(self.image_encoders.keys())
        # }
        img_dict = {
            "agentview_rgb": out[0].squeeze(1)
        }
        return img_dict

    def preprocess_input(self, data, train_mode=True):
        if train_mode:  # apply augmentation
            if self.cfg.train.use_augmentation:
                img_tuple = self._get_img_tuple(data)
                aug_out = self._get_aug_output_dict(self.img_aug(img_tuple))
                # for img_name in self.image_encoders.keys():
                #     data["obs"][img_name] = aug_out[img_name]
                data["obs"]["agentview_rgb"] = aug_out["agentview_rgb"]
            return data
        else:
            data = TensorUtils.recursive_dict_list_tuple_apply(
                data, {torch.Tensor: lambda x: x.unsqueeze(dim=1)}  # add time dimension
            )
            # data["task_emb"] = data["task_emb"].squeeze(1)
        return data
    
    def reset(self):
        """
        Clear all "history" of the policy if there exists any.
        """
        self.current_subtask_id = 0
        self.counter = 0
        self.prev_subgoal_embedding = None
        for policy in self.skill_policies.values():
            policy.reset()
    
    def get_action(self, data):
        self.eval()
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            # meta_state = {"agentview_rgb": data["obs"]["agentview_rgb"].squeeze(1)}
            meta_state = {"agentview_rgb": data["obs"]["agentview_rgb"].squeeze(1), "task_emb": data["task_emb"].squeeze(1)}
            if self.counter % self.freq == 0:
                prediction = self.predict(meta_state)
                subtask_id, subgoal_embedding = prediction["subtask_id"][0], prediction["embedding"]
                self.prev_subgoal_embedding = subgoal_embedding
                if self.current_subtask_id != subtask_id:
                    self.current_subtask_id = subtask_id
                    self.skill_policies[self.current_subtask_id].reset() # reset the subskill Transformer policy
            self.counter += 1
            data["obs"]["subgoal_embedding"] = self.prev_subgoal_embedding
            action = self.skill_policies[self.current_subtask_id].get_action(data)

        return action


# class ExtraModalityTokens(nn.Module):
#     def __init__(
#         self,
#         use_goal=True,
#         extra_num_layers=0,
#         extra_hidden_size=64,
#         extra_embedding_size=32,
#     ):
#         """
#         This is a class that maps all extra modality inputs into tokens of the same size
#         """
#         super().__init__()
#         self.use_goal = use_goal
#         self.extra_embedding_size = extra_embedding_size
#
#         goal_dim = 7
#
#         self.num_extra = int(use_goal)
#
#         extra_low_level_feature_dim = (
#             int(use_goal) * goal_dim
#         )
#
#         assert extra_low_level_feature_dim > 0, "[error] no extra information"
#
#         self.extra_encoders = {}
#
#         def generate_mlp_fn(modality_name, extra_low_level_feature_dim):
#             assert extra_low_level_feature_dim > 0  # we indeed have extra information
#             if extra_num_layers > 0:
#                 layers = [nn.Linear(extra_low_level_feature_dim, extra_hidden_size)]
#                 for i in range(1, extra_num_layers):
#                     layers += [
#                         nn.Linear(extra_hidden_size, extra_hidden_size),
#                         nn.ReLU(inplace=True),
#                     ]
#                 layers += [nn.Linear(extra_hidden_size, extra_embedding_size)]
#             else:
#                 layers = [nn.Linear(extra_low_level_feature_dim, extra_embedding_size)]
#
#             self.mlp = nn.Sequential(*layers)
#             self.extra_encoders[modality_name] = {"encoder": self.mlp}
#
#         for (feature_dim, use_modality, modality_name) in [
#             (goal_dim, self.use_goal, "goal_states"),
#         ]:
#
#             if use_modality:
#                 generate_mlp_fn(modality_name, feature_dim)
#
#         self.encoders = nn.ModuleList(
#             [x["encoder"] for x in self.extra_encoders.values()]
#         )

# class MetaCVAETransformerPolicy(torch.nn.Module):
#     def __init__(self,
#                  cfg,
#                  num_subtasks,
#                  subgoal_embedding_dim,
#                  id_layer_dims,
#                  embedding_layer_dims,
#                  use_eye_in_hand,
#                  policy_type,
#                  activation='relu',
#                  subgoal_type="sigmoid",
#                  latent_dim=32,
#                  use_spatial_softmax=False,
#                  num_kp=64,
#                  visual_feature_dimension=64,
#                  separate_id_prediction=False,
#                  kl_coeff=1.0,
#                  skill_policies=None,
#     ):
#         super().__init__()
#
#         # add data augmentation for rgb inputs
#         self.cfg = cfg
#         meta_cfg = cfg.meta
#         shape_meta = cfg.shape_meta
#
#         self.skill_policies = skill_policies
#         self.num_subtasks = num_subtasks
#         self.subgoal_embedding_dim = subgoal_embedding_dim
#         self.kl_coeff = kl_coeff
#         self.freq = cfg.skill_learning.eval.meta_freq
#
#
#         # add data augmentation for rgb inputs
#         color_aug = eval(meta_cfg.color_aug.network)(
#             **meta_cfg.color_aug.network_kwargs
#         )
#
#         meta_cfg.translation_aug.network_kwargs["input_shape"] = shape_meta[
#             "all_shapes"
#         ][cfg.data.obs.modality.rgb[0]]
#         translation_aug = eval(meta_cfg.translation_aug.network)(
#             **meta_cfg.translation_aug.network_kwargs
#         )
#         self.img_aug = DataAugGroup((color_aug, translation_aug))
#
#
#         ### 1. encode image
#         embed_size = meta_cfg.embed_size
#         transformer_input_sizes = []
#         self.image_encoders = {}
#         for name in shape_meta["all_shapes"].keys():
#             if "rgb" in name or "depth" in name:
#                 kwargs = meta_cfg.image_encoder.network_kwargs
#                 kwargs.input_shape = shape_meta["all_shapes"][name]
#                 kwargs.output_size = embed_size
#                 kwargs.language_dim = (
#                     meta_cfg.language_encoder.network_kwargs.input_size
#                 )
#                 self.image_encoders[name] = {
#                     "input_shape": shape_meta["all_shapes"][name],
#                     "encoder": eval(meta_cfg.image_encoder.network)(**kwargs),
#                 }
#
#         self.encoders = nn.ModuleList(
#             [x["encoder"] for x in self.image_encoders.values()]
#         )
#
#         ### 2. encode language
#         meta_cfg.language_encoder.network_kwargs.output_size = embed_size
#         self.language_encoder = eval(meta_cfg.language_encoder.network)(
#             **meta_cfg.language_encoder.network_kwargs
#         )
#
#         # ### 3. encode extra information (e.g. gripper, joint_state)
#         # self.extra_encoder = ExtraModalityTokens(
#         #     use_goal=True,
#         #     extra_num_layers=meta_cfg.extra_num_layers,
#         #     extra_hidden_size=meta_cfg.extra_hidden_size,
#         #     extra_embedding_size=embed_size,
#         # )
#
#         ### 4. define temporal transformer
#         meta_cfg.temporal_position_encoding.network_kwargs.input_size = embed_size
#         self.temporal_position_encoding_fn = eval(
#             meta_cfg.temporal_position_encoding.network
#         )(**meta_cfg.temporal_position_encoding.network_kwargs)
#
#         self.temporal_transformer = TransformerDecoder(
#             input_size=embed_size,
#             num_layers=meta_cfg.transformer_num_layers,
#             num_heads=meta_cfg.transformer_num_heads,
#             head_output_size=meta_cfg.transformer_head_output_size,
#             mlp_hidden_size=meta_cfg.transformer_mlp_hidden_size,
#             dropout=meta_cfg.transformer_dropout,
#         )
#
#         # policy_head_kwargs = meta_cfg.policy_head.network_kwargs
#         # policy_head_kwargs.input_size = embed_size
#         # policy_head_kwargs.output_size = shape_meta["ac_dim"]
#
#         # self.policy_head = eval(meta_cfg.policy_head.network)(
#         #     **meta_cfg.policy_head.loss_kwargs,
#         #     **meta_cfg.policy_head.network_kwargs
#         # )
#
#         self.latent_queue = []
#         self.max_seq_len = meta_cfg.transformer_max_seq_len
#
#
#
#         transformer_output_dim = embed_size # embed_size
#         if activation == 'relu':
#             activate_fn = torch.nn.ReLU
#         elif activation == 'leaky-relu':
#             activate_fn = torch.nn.LeakyReLU
#
#         self.latent_dim = latent_dim
#         id_input_dim = transformer_output_dim
#         embedding_input_dim = transformer_output_dim + self.latent_dim
#         intermediate_state_dim = 256
#
#
#
#         self.mlp_encoder_layer = torch.nn.Sequential(*[torch.nn.Linear(transformer_output_dim + subgoal_embedding_dim, intermediate_state_dim),
#                                                        activate_fn(),
#                                                        torch.nn.Linear(intermediate_state_dim, intermediate_state_dim),
#                                                        activate_fn(),
#                                                        torch.nn.Linear(intermediate_state_dim, self.latent_dim * 2)])
#
#         self.meta_id_layer = MetaIdLayer(id_input_dim,
#                                          num_subtasks,
#                                          max_subtasks_num= 20,# 10,
#                                          id_layer_dims=id_layer_dims,
#                                          policy_type=policy_type,
#                                          subgoal_type=subgoal_type,
#                                          activation=activation)
#         self.meta_embedding_layer = MetaEmbeddingLayer(embedding_input_dim,
#                                                        num_subtasks,
#                                                        subgoal_embedding_dim,
#                                                        embedding_layer_dims=embedding_layer_dims,
#                                                        policy_type=policy_type,
#                                                        subgoal_type=subgoal_type,
#                                                        activation=activation)
#     def temporal_encode(self, x):
#         pos_emb = self.temporal_position_encoding_fn(x)
#         x = x + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
#         sh = x.shape
#         self.temporal_transformer.compute_mask(x.shape)
#
#         x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality, E)
#         x = self.temporal_transformer(x)
#         x = x.reshape(*sh)
#         return x[:, :, 0]  # (B, T, E)
#
#     def spatial_encode(self, data):
#         encoded = []
#         # # 1. encode extra
#         # extra = self.extra_encoder(data["obs"])  # (B, T, num_extra, E)
#
#         # 3. encode image
#         for img_name in self.image_encoders.keys():
#             x = data["obs"][img_name]
#             B, T, C, H, W = x.shape
#             img_encoded = self.image_encoders[img_name]["encoder"](
#                 x.reshape(B * T, C, H, W),
#                 langs=data["task_emb"]
#                 .reshape(B, 1, -1)
#                 .repeat(1, T, 1)
#                 .reshape(B * T, -1),
#             ).view(B, T, 1, -1)
#             encoded.append(img_encoded)
#
#         # 2. encode language, treat it as action token
#         B, T = img_encoded.shape[:2]
#         text_encoded = self.language_encoder(data)  # (B, E)
#         text_encoded = text_encoded.view(B, 1, 1, -1).expand(
#             -1, T, -1, -1
#         )  # (B, T, 1, E)
#         encoded.append(text_encoded)
#
#         encoded = torch.cat(encoded, -2)  # (B, T, num_modalities, E)
#         return encoded
#
#
#     def forward(self, data):
#         # (B, T, ...)
#         x = self.spatial_encode(data)
#         z_state = self.temporal_encode(x) # (B, T, 64)
#         z_state = z_state.reshape(-1, z_state.shape[-1]) # (Batch, 64)
#         embedding = data['obs']["embedding"].reshape(-1, data['obs']["embedding"].shape[-1]) # (Batch, 64)
#
#
#         h = torch.cat([z_state, embedding], dim=-1) # (Batch, 64 + 64)
#         h = self.mlp_encoder_layer(h) # (Batch, 128)
#
#         mu, logvar = torch.split(h, self.latent_dim, dim=1)
#         z = self.sampling(mu, logvar) # (Batch, 64)
#         z_concat = torch.cat([z_state, z], dim=-1) # (Batch, 64 + 64)
#         skill_id = self.meta_id_layer(z_state) # (Batch, 6)
#         embedding = self.meta_embedding_layer(z_concat)
#         return skill_id, embedding, mu, logvar
#
#
#     def predict(self, data):
#         # (B, T, ...)
#         x = self.spatial_encode(data)
#         self.latent_queue.append(x)
#         if len(self.latent_queue) > self.max_seq_len:
#             self.latent_queue.pop(0)
#         x = torch.cat(self.latent_queue, dim=1)  # (B, T, H_all)
#         B, T = x.shape[:2]
#         z_state = self.temporal_encode(x)[:,-1]
#         z = safe_cuda(torch.randn(B, self.latent_dim))
#
#         subtask_vector = self.meta_id_layer(z_state)
#         subtask_id = torch.argmax(subtask_vector, dim=1)
#         z_concat = [z_state, z]
#
#         subgoal_embedding = self.meta_embedding_layer(torch.cat(z_concat, dim=-1))
#
#         subtask_id = subtask_id.cpu().detach().numpy()
#         return {"subtask_id": subtask_id,
#                 "subtask_vector": subtask_vector,
#                 "embedding": subgoal_embedding}
#
#     def sampling(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(mu)
#         return eps * std + mu
#
#     def compute_loss(self, data, reduction="mean"):
#         use_final_state_goal = False #TODO: update this
#         data = self.preprocess_input(data, train_mode=True)
#         B, T = data["obs"]["id"].shape[:2]
#         cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction=reduction)
#         mse_loss = torch.nn.MSELoss(reduction=reduction)
#         subtask_id_prediction, subgoal_embedding, mu, logvar = self.forward(data)
#         subtask_id_prediction = subtask_id_prediction[:, :self.num_subtasks] # masking
#         # if use_final_state_goal:
#         #     data_obs_id_final = data["obs"]["id"][:, -1, ...].unsqueeze(1).expand(-1, data["obs"]["id"].shape[1], *data["obs"]["id"].shape[2:])
#         #     data_obs_embedding_final = data["obs"]["embedding"][:, -1, ...].unsqueeze(1).expand(-1, data["obs"]["embedding"].shape[1], *data["obs"]["embedding"].shape[2:])
#         #     data_obs_embedding_final = data_obs_embedding_final.reshape(-1, data_obs_embedding_final.shape[-1])
#         #
#         #     ce_loss = cross_entropy_loss(subtask_id_prediction.view(-1, self.num_subtasks), data_obs_id_final.contiguous().view(-1))
#         #     embedding_loss = mse_loss(data_obs_embedding_final, subgoal_embedding)
#         #     kl_loss = - self.kl_coeff * 400 * torch.mean(0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2), dim=1), dim=0)
#         # else:
#         ce_loss = cross_entropy_loss(subtask_id_prediction.view(-1, self.num_subtasks), data["obs"]["id"].view(-1))
#         data_embedding = data["obs"]["embedding"].reshape(-1, data["obs"]["embedding"].shape[-1])
#         embedding_loss = mse_loss(data_embedding, subgoal_embedding)
#         kl_loss = - self.kl_coeff * 400 * torch.mean(0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2), dim=1), dim=0)
#
#         loss = ce_loss + embedding_loss + kl_loss
#
#         return loss, ce_loss, embedding_loss, kl_loss
#
#     def _get_img_tuple(self, data):
#         img_tuple = tuple(
#             [data["obs"][img_name] for img_name in self.image_encoders.keys()]
#         )
#         return img_tuple
#
#     def _get_aug_output_dict(self, out):
#         img_dict = {
#             img_name: out[idx]
#             for idx, img_name in enumerate(self.image_encoders.keys())
#         }
#         return img_dict
#
#     def preprocess_input(self, data, train_mode=True):
#         if train_mode:  # apply augmentation
#             if self.cfg.train.use_augmentation:
#                 img_tuple = self._get_img_tuple(data)
#                 aug_out = self._get_aug_output_dict(self.img_aug(img_tuple))
#                 for img_name in self.image_encoders.keys():
#                     data["obs"][img_name] = aug_out[img_name]
#             return data
#         else:
#             data = TensorUtils.recursive_dict_list_tuple_apply(
#                 data, {torch.Tensor: lambda x: x.unsqueeze(dim=1)}  # add time dimension
#             )
#             data["task_emb"] = data["task_emb"].squeeze(1)
#         return data
#
#     def reset(self):
#         """
#         Clear all "history" of the policy if there exists any.
#         """
#         self.latent_queue = []
#         self.current_subtask_id = 0
#         self.counter = 0
#         self.prev_subgoal_embedding = None
#         for policy in self.skill_policies.values():
#             policy.reset()
#
#     def get_action(self, data):
#         self.eval()
#         with torch.no_grad():
#             data = self.preprocess_input(data, train_mode=False)
#             # meta_state = {"agentview_rgb": data["obs"]["agentview_rgb"], "task_emb": data["task_emb"]}
#             if self.counter % self.freq == 0:
#                 prediction = self.predict(data)
#                 subtask_id, subgoal_embedding = prediction["subtask_id"][0], prediction["embedding"]
#                 self.prev_subgoal_embedding = subgoal_embedding
#                 if self.current_subtask_id != subtask_id:
#                     self.current_subtask_id = subtask_id
#                     self.skill_policies[self.current_subtask_id].reset() # reset the subskill Transformer policy
#             self.counter += 1
#             data["obs"]["subgoal_embedding"] = self.prev_subgoal_embedding
#             action = self.skill_policies[self.current_subtask_id].get_action(data)
#
#         return action

class ExtraModalityTokens(nn.Module):
    def __init__(
        self,
        use_joint=False,
        use_gripper=False,
        use_ee=False,
        goal_modality=None,
        extra_num_layers=0,
        extra_hidden_size=64,
        extra_embedding_size=32,
    ):
        """
        This is a class that maps all extra modality inputs into tokens of the same size
        """
        super().__init__()
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_ee = use_ee
        self.extra_embedding_size = extra_embedding_size
        self.goal_modality = goal_modality
        self.use_goal_state = True if goal_modality in ["ee_states", "joint_states", "dinov2"] else False

        joint_states_dim = 7
        gripper_states_dim = 2
        ee_dim = 3
        if goal_modality == "ee_states":
            goal_dim = 8
        elif goal_modality == "joint_states":
            goal_dim = 9
        elif goal_modality == "dinov2":
            goal_dim = 768 #1536 # TODO
        else:
            goal_dim = 0

        self.num_extra = int(use_joint) + int(use_gripper) + int(use_ee) + int(self.use_goal_state)

        extra_low_level_feature_dim = (
            int(use_joint) * joint_states_dim
            + int(use_gripper) * gripper_states_dim
            + int(use_ee) * ee_dim
            + int(self.use_goal_state) * goal_dim
        )

        assert extra_low_level_feature_dim > 0, "[error] no extra information"

        self.extra_encoders = {}

        def generate_proprio_mlp_fn(modality_name, extra_low_level_feature_dim):
            assert extra_low_level_feature_dim > 0  # we indeed have extra information
            if extra_num_layers > 0:
                layers = [nn.Linear(extra_low_level_feature_dim, extra_hidden_size)]
                if self.goal_modality == "dinov2": # linear projection for dinov2
                    for i in range(1, extra_num_layers):
                        layers += [
                            nn.Linear(extra_hidden_size, extra_hidden_size),
                        ]
                else:
                    for i in range(1, extra_num_layers):
                        layers += [
                            nn.Linear(extra_hidden_size, extra_hidden_size),
                            nn.ReLU(inplace=True),
                        ]
                layers += [nn.Linear(extra_hidden_size, extra_embedding_size)]
            else:
                layers = [nn.Linear(extra_low_level_feature_dim, extra_embedding_size)]

            self.proprio_mlp = nn.Sequential(*layers)
            self.extra_encoders[modality_name] = {"encoder": self.proprio_mlp}

        for (proprio_dim, use_modality, modality_name) in [
            (joint_states_dim, self.use_joint, "joint_states"),
            (gripper_states_dim, self.use_gripper, "gripper_states"),
            (ee_dim, self.use_ee, "ee_states"),
            (goal_dim, self.use_goal_state, "goal_states"),
        ]:

            if use_modality:
                generate_proprio_mlp_fn(modality_name, proprio_dim)

        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.extra_encoders.values()]
        )

    def forward(self, obs_dict):
        """
        obs_dict: {
            (optional) joint_stats: (B, T, 7),
            (optional) gripper_states: (B, T, 2),
            (optional) ee: (B, T, 3)
        }
        map above to a latent vector of shape (B, T, H)
        """
        tensor_list = []

        for (use_modality, modality_name) in [
            (self.use_joint, "joint_states"),
            (self.use_gripper, "gripper_states"),
            (self.use_ee, "ee_states"),
        ]:

            if use_modality:
                tensor_list.append(
                    self.extra_encoders[modality_name]["encoder"](
                        obs_dict[modality_name]
                    )
                )

        x = torch.stack(tensor_list, dim=-2)
        return x

class MetaCVAETransformerPolicy(torch.nn.Module):
    def __init__(self,
                 cfg,
                 num_subtasks,
                 subgoal_embedding_dim,
                 id_layer_dims,
                 embedding_layer_dims,
                 use_eye_in_hand,
                 policy_type,
                 activation='relu',
                 subgoal_type="sigmoid",
                 latent_dim=32,
                 use_spatial_softmax=False,
                 num_kp=64,
                 visual_feature_dimension=64,
                 separate_id_prediction=False,
                 kl_coeff=1.0,
                 skill_policies=None,
    ):
        super().__init__()

        # add data augmentation for rgb inputs
        self.cfg = cfg
        meta_cfg = cfg.meta
        shape_meta = cfg.shape_meta

        self.skill_policies = skill_policies
        self.num_subtasks = num_subtasks    # 2
        # self.subgoal_embedding_dim = subgoal_embedding_dim
        self.kl_coeff = kl_coeff    # 0.01
        self.freq = cfg.skill_learning.eval.meta_freq   # 80

        # add data augmentation for rgb inputs
        color_aug = eval(meta_cfg.color_aug.network)(
            **meta_cfg.color_aug.network_kwargs
        )

        meta_cfg.translation_aug.network_kwargs["input_shape"] = shape_meta[
            "all_shapes"
        ][cfg.data.obs.modality.rgb[0]]
        translation_aug = eval(meta_cfg.translation_aug.network)(
            **meta_cfg.translation_aug.network_kwargs
        )
        self.img_aug = DataAugGroup((color_aug, translation_aug))


        ### 1. encode image
        embed_size = meta_cfg.embed_size
        transformer_input_sizes = []
        self.image_encoders = {}
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                kwargs = meta_cfg.image_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                kwargs.output_size = embed_size
                kwargs.language_dim = (
                    meta_cfg.language_encoder.network_kwargs.input_size
                )
                self.image_encoders[name] = {
                    "input_shape": shape_meta["all_shapes"][name],
                    "encoder": eval(meta_cfg.image_encoder.network)(**kwargs),
                }

        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.image_encoders.values()]
        )

        ### 2. encode language
        meta_cfg.language_encoder.network_kwargs.output_size = embed_size
        self.language_encoder = eval(meta_cfg.language_encoder.network)(
            **meta_cfg.language_encoder.network_kwargs
        )

        # ### 3. encode extra information (e.g. gripper, joint_state)
        # self.extra_encoder = ExtraModalityTokens(
        #     use_goal=True,
        #     extra_num_layers=meta_cfg.extra_num_layers,
        #     extra_hidden_size=meta_cfg.extra_hidden_size,
        #     extra_embedding_size=embed_size,
        # )
        self.extra_encoder = ExtraModalityTokens(
            use_joint=True,
            use_gripper=True,
            use_ee=False,
            extra_num_layers=meta_cfg.extra_num_layers,
            extra_hidden_size=meta_cfg.extra_hidden_size,
            extra_embedding_size=embed_size,
        )

        ### 4. define temporal transformer
        meta_cfg.temporal_position_encoding.network_kwargs.input_size = embed_size
        self.temporal_position_encoding_fn = eval(
            meta_cfg.temporal_position_encoding.network
        )(**meta_cfg.temporal_position_encoding.network_kwargs)


        self.trans = "moe"
        # self.trans = "transformer"
        if self.trans == "moe":
            self.temporal_transformer = MoeTransformerDecoder(
                input_size=embed_size,
                num_layers=meta_cfg.transformer_num_layers,
                num_heads=meta_cfg.transformer_num_heads,
                head_output_size=meta_cfg.transformer_head_output_size,
                num_experts=16,  # 12
                top_k=4,
                dropout=meta_cfg.transformer_dropout,
                is_multigate=True
            )
        else:
            self.temporal_transformer = TransformerDecoder(
                input_size=embed_size,
                num_layers=meta_cfg.transformer_num_layers,
                num_heads=meta_cfg.transformer_num_heads,
                head_output_size=meta_cfg.transformer_head_output_size,
                mlp_hidden_size=meta_cfg.transformer_mlp_hidden_size,
                dropout=meta_cfg.transformer_dropout,
            )

        # policy_head_kwargs = meta_cfg.policy_head.network_kwargs
        # policy_head_kwargs.input_size = embed_size
        # policy_head_kwargs.output_size = shape_meta["ac_dim"]

        # self.policy_head = eval(meta_cfg.policy_head.network)(
        #     **meta_cfg.policy_head.loss_kwargs,
        #     **meta_cfg.policy_head.network_kwargs
        # )

        self.latent_queue = []
        # self.max_seq_len = meta_cfg.transformer_max_seq_len
        self.max_seq_len = cfg.data.meta_seq_len

        transformer_output_dim = embed_size  # embed_size
        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU

        self.latent_dim = latent_dim
        id_input_dim = transformer_output_dim
        intermediate_state_dim = 256

        # self.mlp_encoder_layer = torch.nn.Sequential(*[torch.nn.Linear(transformer_output_dim, intermediate_state_dim),
        #                                                activate_fn(),
        #                                                torch.nn.Linear(intermediate_state_dim, intermediate_state_dim),
        #                                                activate_fn(),
        #                                                torch.nn.Linear(intermediate_state_dim, self.latent_dim * 2)])

        self.meta_id_layer = MetaIdLayer(id_input_dim,
                                         num_subtasks,
                                         max_subtasks_num=num_subtasks,# 10,
                                         id_layer_dims=id_layer_dims,   # [300, 400]
                                         policy_type=policy_type,
                                         subgoal_type=subgoal_type,
                                         activation=activation)


    def temporal_encode(self, x, task_id=None):
        pos_emb = self.temporal_position_encoding_fn(x)
        x = x + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality, E)
        if self.trans == "moe":
            x = self.temporal_transformer(x, task_id)
        else:
            x = self.temporal_transformer(x)
        x = x.reshape(*sh)
        return x[:, :, 0]  # (B, T, E)

    def spatial_encode(self, data):
        encoded = []
        # # 1. encode extra
        extra = self.extra_encoder(data["obs"])  # (B, T, num_extra, E)
        encoded.append(extra)
        # 3. encode image
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape
            img_encoded = self.image_encoders[img_name]["encoder"](
                x.reshape(B * T, C, H, W),
                langs=data["task_emb"]
                .reshape(B, 1, -1)
                .repeat(1, T, 1)
                .reshape(B * T, -1),
            ).view(B, T, 1, -1)
            encoded.append(img_encoded)
        
        # 2. encode language, treat it as action token
        B, T = img_encoded.shape[:2]
        text_encoded = self.language_encoder(data)  # (B, E)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(
            -1, T, -1, -1
        )  # (B, T, 1, E)
        encoded.append(text_encoded)

        encoded = torch.cat(encoded, -2)  # (B, T, num_modalities, E)
        return encoded

    def forward(self, data):
        # (B, T, ...)
        x = self.spatial_encode(data)
        if self.trans == "moe":
            z_state = self.temporal_encode(x, data["task_id"])  # (B, T, 64)
        else:
            z_state = self.temporal_encode(x)  # (B, T, 64)
        z_state = z_state.reshape(-1, z_state.shape[-1])  # (Batch, 64)
        skill_id = self.meta_id_layer(z_state)  # (Batch, 20)
        return skill_id

    def predict(self, data):
        # (B, T, ...)
        x = self.spatial_encode(data)
        self.latent_queue.append(x)
        if len(self.latent_queue) > self.max_seq_len:
            self.latent_queue.pop(0)
        x = torch.cat(self.latent_queue, dim=1)  # (B, T, H_all)
        B, T = x.shape[:2]
        if self.trans == "moe":
            z_state = self.temporal_encode(x, data["task_id"])[:, -1]
        else:
            z_state = self.temporal_encode(x)[:, -1]


        subtask_vector = self.meta_id_layer(z_state)
        # print(subtask_vector.shape)   # (bs, 10)
        subtask_id = torch.argmax(subtask_vector, dim=1)
        # print(subtask_id.shape)   # (bs,)
        subtask_id = subtask_id.cpu().detach().numpy()

        return {"subtask_id": subtask_id}


    def compute_loss(self, data, reduction="mean"):
        data = self.preprocess_input(data, train_mode=True)
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction=reduction)
        subtask_id_prediction = self.forward(data)
        subtask_id_prediction = subtask_id_prediction[:, :self.num_subtasks]  # masking (32, 2)
        # print(subtask_id_prediction.shape)

        loss = cross_entropy_loss(subtask_id_prediction.view(-1, self.num_subtasks), data["obs"]["id"].view(-1))    # (32,)

        return loss

    def _get_img_tuple(self, data):
        img_tuple = tuple(
            [data["obs"][img_name] for img_name in self.image_encoders.keys()]
        )
        return img_tuple

    def _get_aug_output_dict(self, out):
        img_dict = {
            img_name: out[idx]
            for idx, img_name in enumerate(self.image_encoders.keys())
        }
        return img_dict

    def preprocess_input(self, data, train_mode=True):
        if train_mode:  # apply augmentation
            if self.cfg.train.use_augmentation:
                img_tuple = self._get_img_tuple(data)
                aug_out = self._get_aug_output_dict(self.img_aug(img_tuple))
                for img_name in self.image_encoders.keys():
                    data["obs"][img_name] = aug_out[img_name]
            return data
        else:
            data = TensorUtils.recursive_dict_list_tuple_apply(
                data, {torch.Tensor: lambda x: x.unsqueeze(dim=1)}  # add time dimension
            )
            data["task_emb"] = data["task_emb"].squeeze(1)
            if "task_id" in data:
                data["task_id"] = data["task_id"].squeeze(1)
        return data

    def reset(self):
        """
        Clear all "history" of the policy if there exists any.
        """
        self.latent_queue = []
        self.current_subtask_id = 0
        self.counter = 0
        self.prev_subgoal_embedding = None
        for policy in self.skill_policies.values():
            policy.reset()

    def get_action(self, data):
        self.eval()
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            # if self.counter % self.freq == 0:
            if self.counter % 10 == 0:
                predict = self.predict(data)["subtask_id"]
                # print(predict)
                subtask_id = predict[0]

                # if self.counter < 250:
                #     subtask_id_set = 0
                # else:
                #     subtask_id_set = 0
                #
                # print(subtask_id_set)
                #
                # if self.current_subtask_id != subtask_id_set:
                #     self.current_subtask_id = subtask_id_set
                if self.current_subtask_id != subtask_id:
                    self.current_subtask_id = subtask_id
                    self.skill_policies[self.current_subtask_id].reset()  # reset the subskill Transformer policy
            self.counter += 1
            data["task_id"] = self.current_subtask_id    # (bs,)
            action = self.skill_policies[self.current_subtask_id].get_action(data)

        return action


class ACILTransformerPolicy(torch.nn.Module):
    def __init__(self,
                 cfg,
                 num_subtasks,
                 subgoal_embedding_dim,
                 id_layer_dims,
                 embedding_layer_dims,
                 use_eye_in_hand,
                 policy_type,
                 activation='relu',
                 subgoal_type="sigmoid",
                 latent_dim=32,
                 use_spatial_softmax=False,
                 num_kp=64,
                 visual_feature_dimension=64,
                 separate_id_prediction=False,
                 kl_coeff=1.0,
                 skill_policies=None,
                 ):
        super().__init__()

        # add data augmentation for rgb inputs
        self.cfg = cfg
        meta_cfg = cfg.meta
        shape_meta = cfg.shape_meta

        shape_meta["all_shapes"] = {}
        # shape_meta["all_shapes"]["agentview_rgb"] = [3, 128, 128]
        shape_meta["all_shapes"]["agentview_rgb"] = [3, 480, 640]   # for real

        self.skill_policies = skill_policies
        self.num_subtasks = num_subtasks  # 2
        # self.subgoal_embedding_dim = subgoal_embedding_dim
        self.kl_coeff = kl_coeff  # 0.01
        self.freq = cfg.skill_learning.eval.meta_freq  # 80

        # add data augmentation for rgb inputs
        color_aug = eval(meta_cfg.color_aug.network)(
            **meta_cfg.color_aug.network_kwargs
        )

        meta_cfg.translation_aug.network_kwargs["input_shape"] = shape_meta[
            "all_shapes"
        ][cfg.data.obs.modality.rgb[0]]
        translation_aug = eval(meta_cfg.translation_aug.network)(
            **meta_cfg.translation_aug.network_kwargs
        )
        self.img_aug = DataAugGroup((color_aug, translation_aug))

        ### 1. encode image
        embed_size = meta_cfg.embed_size
        transformer_input_sizes = []
        self.image_encoders = {}
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                kwargs = meta_cfg.image_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                kwargs.output_size = embed_size
                kwargs.language_dim = (
                    meta_cfg.language_encoder.network_kwargs.input_size
                )
                self.image_encoders[name] = {
                    "input_shape": shape_meta["all_shapes"][name],
                    "encoder": eval(meta_cfg.image_encoder.network)(**kwargs),
                }

        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.image_encoders.values()]
        )

        ### 2. encode language
        meta_cfg.language_encoder.network_kwargs.output_size = embed_size
        self.language_encoder = eval(meta_cfg.language_encoder.network)(
            **meta_cfg.language_encoder.network_kwargs
        )

        # ### 3. encode extra information (e.g. gripper, joint_state)
        # self.extra_encoder = ExtraModalityTokens(
        #     use_goal=True,
        #     extra_num_layers=meta_cfg.extra_num_layers,
        #     extra_hidden_size=meta_cfg.extra_hidden_size,
        #     extra_embedding_size=embed_size,
        # )
        self.extra_encoder = ExtraModalityTokens(
            use_joint=True,
            use_gripper=True,
            # use_gripper=False,   # for real robot
            use_ee=False,
            extra_num_layers=meta_cfg.extra_num_layers,
            extra_hidden_size=meta_cfg.extra_hidden_size,
            extra_embedding_size=embed_size,
        )

        ### 4. define temporal transformer
        meta_cfg.temporal_position_encoding.network_kwargs.input_size = embed_size
        self.temporal_position_encoding_fn = eval(
            meta_cfg.temporal_position_encoding.network
        )(**meta_cfg.temporal_position_encoding.network_kwargs)


        self.temporal_transformer = TransformerDecoder(
            input_size=embed_size,
            num_layers=meta_cfg.transformer_num_layers,
            num_heads=meta_cfg.transformer_num_heads,
            head_output_size=meta_cfg.transformer_head_output_size,
            mlp_hidden_size=meta_cfg.transformer_mlp_hidden_size,
            dropout=meta_cfg.transformer_dropout,
        )

        self.latent_queue = []
        # self.max_seq_len = meta_cfg.transformer_max_seq_len
        self.max_seq_len = cfg.data.meta_seq_len

        transformer_output_dim = embed_size  # embed_size
        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU

        self.latent_dim = latent_dim
        id_input_dim = transformer_output_dim
        intermediate_state_dim = 256

        # self.mlp_encoder_layer = torch.nn.Sequential(*[torch.nn.Linear(transformer_output_dim, intermediate_state_dim),
        #                                                activate_fn(),
        #                                                torch.nn.Linear(intermediate_state_dim, intermediate_state_dim),
        #                                                activate_fn(),
        #                                                torch.nn.Linear(intermediate_state_dim, self.latent_dim * 2)])

        # self.meta_id_layer = MetaIdLayer(id_input_dim,
        #                                  num_subtasks,
        #                                  max_subtasks_num=50,  # 10,
        #                                  id_layer_dims=id_layer_dims,  # [300, 400]
        #                                  policy_type=policy_type,
        #                                  subgoal_type=subgoal_type,
        #                                  activation=activation)     # (64, 300), (300, 400), (400, 17)

    def temporal_encode(self, x, task_id=None):
        pos_emb = self.temporal_position_encoding_fn(x)
        x = x + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality, E)
        if self.trans == "moe":
            x = self.temporal_transformer(x, task_id)
        else:
            x = self.temporal_transformer(x)
        x = x.reshape(*sh)
        return x[:, :, 0]  # (B, T, E)

    def spatial_encode(self, data):
        encoded = []
        # # 1. encode extra
        extra = self.extra_encoder(data["obs"])  # (B, T, num_extra, E)
        encoded.append(extra)
        # 3. encode image
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape
            img_encoded = self.image_encoders[img_name]["encoder"](
                x.reshape(B * T, C, H, W),
                langs=data["task_emb"]
                .reshape(B, 1, -1)
                .repeat(1, T, 1)
                .reshape(B * T, -1),
            ).view(B, T, 1, -1)
            encoded.append(img_encoded)

        # 2. encode language, treat it as action token
        B, T = img_encoded.shape[:2]
        text_encoded = self.language_encoder(data)  # (B, E)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(
            -1, T, -1, -1
        )  # (B, T, 1, E)
        encoded.append(text_encoded)

        encoded = torch.cat(encoded, -2)  # (B, T, num_modalities, E)
        return encoded

    def forward(self, data):
        # (B, T, ...)
        x = self.spatial_encode(data)
        if self.trans == "moe":
            z_state = self.temporal_encode(x, data["task_id"])  # (B, T, 64)
        else:
            z_state = self.temporal_encode(x)  # (B, T, 64)
        z_state = z_state.reshape(-1, z_state.shape[-1])  # (Batch, 64)
        # skill_id = self.meta_id_layer(z_state)  # (Batch, 20)
        # return skill_id
        return z_state

    # def predict(self, data):
    #     # (B, T, ...)
    #     x = self.spatial_encode(data)
    #     self.latent_queue.append(x)
    #     if len(self.latent_queue) > self.max_seq_len:
    #         self.latent_queue.pop(0)
    #     x = torch.cat(self.latent_queue, dim=1)  # (B, T, H_all)
    #     B, T = x.shape[:2]
    #     if self.trans == "moe":
    #         z_state = self.temporal_encode(x, data["task_id"])[:, -1]
    #     else:
    #         z_state = self.temporal_encode(x)[:, -1]
    #
    #     subtask_vector = self.meta_id_layer(z_state)
    #     # print(subtask_vector.shape)   # (bs, 10)
    #     subtask_id = torch.argmax(subtask_vector, dim=1)
    #     # print(subtask_id.shape)   # (bs,)
    #     subtask_id = subtask_id.cpu().detach().numpy()
    #
    #     return {"subtask_id": subtask_id}

    def compute_loss(self, data, reduction="mean"):
        data = self.preprocess_input(data, train_mode=True)
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction=reduction)
        subtask_id_prediction = self.forward(data)
        subtask_id_prediction = subtask_id_prediction[:, :self.num_subtasks]  # masking (32, 2)
        # print(subtask_id_prediction.shape)

        loss = cross_entropy_loss(subtask_id_prediction.view(-1, self.num_subtasks), data["obs"]["id"].view(-1))  # (32,)

        return loss

    def _get_img_tuple(self, data):
        img_tuple = tuple(
            [data["obs"][img_name] for img_name in self.image_encoders.keys()]
        )
        return img_tuple

    def _get_aug_output_dict(self, out):
        img_dict = {
            img_name: out[idx]
            for idx, img_name in enumerate(self.image_encoders.keys())
        }
        return img_dict

    def preprocess_input(self, data, train_mode=True):
        if train_mode:  # apply augmentation
            if self.cfg.train.use_augmentation:
                img_tuple = self._get_img_tuple(data)
                aug_out = self._get_aug_output_dict(self.img_aug(img_tuple))
                for img_name in self.image_encoders.keys():
                    data["obs"][img_name] = aug_out[img_name]
            return data
        else:
            data = TensorUtils.recursive_dict_list_tuple_apply(
                data, {torch.Tensor: lambda x: x.unsqueeze(dim=1)}  # add time dimension
            )
            data["task_emb"] = data["task_emb"].squeeze(1)
            if "task_id" in data:
                data["task_id"] = data["task_id"].squeeze(1)
        return data

    def reset(self):
        """
        Clear all "history" of the policy if there exists any.
        """
        self.latent_queue = []
        self.current_subtask_id = 0
        self.counter = 0
        self.prev_subgoal_embedding = None
        for policy in self.skill_policies.values():
            policy.reset()

