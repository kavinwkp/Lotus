import h5py
import os


input_dir = "lotus/datasets/real_10"
output_file = "lotus/datasets/real_10/pickup_cube_demo.hdf5"

input_files = [
    f for f in os.listdir(input_dir)
    if f.startswith("episode") and f.endswith(".hdf5")
]

# input_files.sort()
print(input_files)

with h5py.File(output_file, 'w') as out_f:
    data_group = out_f.create_group(f"data")

    for idx, filename in enumerate(input_files):
        filepath = os.path.join(input_dir, filename)

        print(f"process {idx} hdf5")

        with h5py.File(filepath, 'r') as in_f:
            actions = in_f['action'][:]
            # print(actions.shape)
            eye_in_hand_rgb = in_f["observations"]["images"]["right"][:]
            # print(eye_in_hand_rgb.shape)
            agentview_rgb = in_f["observations"]["images"]["top"][:]
            # print(agentview_rgb.shape)
            joint_states = in_f["observations"]["qpos"][:]
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
