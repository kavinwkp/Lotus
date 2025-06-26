import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
input_dir = "lotus/datasets/real_10"
output_file = "lotus/datasets/real_10/pickup_cube_demo.hdf5"

input_files = [
    f for f in os.listdir(input_dir)
    if f.startswith("episode") and f.endswith(".hdf5")
]

input_files.sort()
print(input_files)

actions_plot = []

with h5py.File(output_file, 'w') as out_f:
    data_group = out_f.create_group(f"data")

    for idx, filename in enumerate(input_files):
        filepath = os.path.join(input_dir, filename)

        print(f"process {idx} hdf5")

        with h5py.File(filepath, 'r') as in_f:
            actions = in_f['actions'][:]
            # actions_plot.append(actions)

            # for i in range(6):
            #     actions[:, i] = (actions[:, i] + np.pi) / (2 * np.pi)
            # print(actions.shape)
            eye_in_hand_rgb = in_f["obs"]["images"]["right"][:]
            # print(eye_in_hand_rgb.shape)
            agentview_rgb = in_f["obs"]["images"]["top"][:]
            # print(agentview_rgb.shape)
            joint_states = in_f["obs"]["qpos"][:]
            # print(joint_states.shape)

            demo_group = data_group.create_group(f"demo_{idx}")

            demo_group.attrs["num_samples"] = actions.shape[0]

            demo_group.create_dataset("actions", data=actions)

            obs_group = demo_group.create_group("obs")
            obs_group.create_dataset("agentview_rgb", data=agentview_rgb)
            obs_group.create_dataset("eye_in_hand_rgb", data=eye_in_hand_rgb)
            obs_group.create_dataset("joint_states", data=joint_states)
        # break

print(f"合并完成，结果保存至 {output_file}")

# num_plots = 7
# cols = 4  # 每行显示2个子图
# rows = (num_plots // cols) + int(num_plots % cols > 0)  # 计算需要的行数
# # 创建子图
# fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
#
# # 如果只有一行或一列，axes 不是二维数组，需要调整
# if rows == 1:
#     axes = np.array([axes])
# elif cols == 1:
#     axes = axes.reshape(-1, 1)
#
# # 遍历每个子图并绘制曲线
# for i in range(num_plots):
#     row = i // cols
#     col = i % cols
#     ax = axes[row, col]
#
#     for j, data in enumerate(actions_plot):
#         ax.plot(data[:, i], label=f'{j + 1}')
#
#     # ax.set_title(f'子图 {i + 1}: 维度 {i + 1}')
#     # ax.set_xlabel('数据点')
#     # ax.set_ylabel('值')
#     ax.legend()
#     ax.grid(True)
#
# # 如果子图数量不足，隐藏多余的子图
# if num_plots % cols != 0:
#     for i in range(num_plots, rows * cols):
#         row = i // cols
#         col = i % cols
#         axes[row, col].axis('off')
#
# plt.tight_layout()
# plt.show()

# print(output_file)
# f = h5py.File(output_file, "r")
#
# for i in range(5):
#     agentview_rgb = f[f"data/demo_{i}/obs/agentview_rgb"][()]
#     print(agentview_rgb.shape)

# file_path = 'lotus/datasets/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5'
# with h5py.File(output_file, 'r') as file:
#     print(file['data'].keys())  # ['demo_0',]
#     for i in range(len(file['data'].keys())):
#         num_samples = file[f"data/demo_{i}"].attrs["num_samples"]
#         print(f"Number of samples: {num_samples}")
