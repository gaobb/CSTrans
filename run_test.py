import os
import numpy as np
import time
import copy
from time import gmtime
from tools.train_eval_tools import cal_metrics_using_count_dict
from tools.constants import PROJECT_DIR


st_time = time.time()
run_mode = "testing"
output_dir = f"{PROJECT_DIR}/outputs"
dir_list_list = [
    [
        f"{output_dir}/d=a_c1_o32_rt=swt_p2_e128_d22222_n88442_w7_TFTF_f=T_t=m_n=1"
        # f"{output_dir}/xxx",
        # f"{output_dir}/xxx",
        # f"{output_dir}/xxx",
        # ...
        # f"{output_dir}/xxx",
    ],
]

adapt = False  # whether to use test-time adaption
eval_mean_std = True  # whether to calculate mean and std
test_split_list = [
    "val",
    "test",
    "val_coco",
    "test_coco",
]
adapt_setting_str = "adapt" if adapt else "noadapt"
if eval_mean_std:
    box_group_index_list = [0, 1, 2]
else:
    box_group_index_list = [0]

reset_trbgn_to_tebgn = True

new_dir_list_list = copy.deepcopy(dir_list_list)
for dir_list_i in range(len(dir_list_list)):
    for cur_dir_i in range(len(dir_list_list[dir_list_i])):
        cur_dir = dir_list_list[dir_list_i][cur_dir_i]
        cur_reg_path = os.path.join(cur_dir, "regressor.pth")
        if (not os.path.exists(cur_reg_path)) or (".txt" in cur_dir):
            print("==> remove cur_dir: ", cur_dir)
            new_dir_list_list[dir_list_i].remove(cur_dir)
        else:
            print("====> cur_reg_path: ", cur_reg_path)
print("=" * 100 + "\n")

for dir_list in new_dir_list_list:
    if len(dir_list) >= 1:
        real_test_split_list = test_split_list
        tebgn = 1
        nproc_per_node = 1
        # nproc_per_node = 8
        testing_gpu_mode = "multiple_train_multiple_test"

        for test_split in real_test_split_list:
            dir_list.sort()

            for cur_dir_i, cur_dir in enumerate(dir_list):
                if reset_trbgn_to_tebgn:
                    tebgn = int(cur_dir.split("_n=")[-1].split("_")[0])

                # for mean and std
                MAE_list = []
                RMSE_list = []
                POS_ERR_list = []
                NEG_ERR_list = []
                for box_group_index in box_group_index_list:
                    print("\n\n" + "=" * 100)
                    print(f"==> dir_list: {dir_list}")
                    print(f"==> test_split: {test_split}")
                    print(f"==> cur_dir: {cur_dir}")
                    print(f"==> Processing {cur_dir_i + 1}/{len(dir_list)}")
                    assert (
                        os.system(
                            "python3 -m torch.distributed.launch --nproc_per_node={} --master_port 6666 test.py --output_dir {} --test_split {} --testing_gpu_mode {} --tebgn {} --adapt {} --box_group_index {}".format(
                                nproc_per_node,
                                cur_dir,
                                test_split,
                                testing_gpu_mode,
                                tebgn,
                                adapt,
                                box_group_index,
                            )
                        )
                        == 0
                    )

                    print("=" * 100)
                    print("cur_dir: ", cur_dir)
                    assert (
                        os.system(
                            "cat {}/results_{}_{}_tebgn={}_bgi={}.txt".format(
                                cur_dir,
                                test_split,
                                adapt_setting_str,
                                tebgn,
                                box_group_index,
                            )
                        )
                        == 0
                    )

                    # save results for calculating mean and std
                    save_json_file_dir = os.path.join(
                        cur_dir,
                        f"metrics_json_results_rm={run_mode}_s={test_split}_{adapt_setting_str}",
                    )
                    (
                        acc_MAE,
                        acc_RMSE,
                        acc_POS_ERR,
                        acc_NEG_ERR,
                        all_sample_num,
                    ) = cal_metrics_using_count_dict(save_json_file_dir, with_loss=False)
                    MAE_list.append(acc_MAE)
                    RMSE_list.append(acc_RMSE)
                    POS_ERR_list.append(acc_POS_ERR)
                    NEG_ERR_list.append(acc_NEG_ERR)

                all_metric_save_path = cur_dir.split("_rv")[0] + "_all_adapt={}_eval_mean_std={}.txt".format(
                    adapt, eval_mean_std
                )
                with open(all_metric_save_path, "a") as f:
                    for box_group_index in box_group_index_list:
                        with open(
                            "{}/results_{}_{}_tebgn={}_bgi={}.txt".format(
                                cur_dir,
                                test_split,
                                adapt_setting_str,
                                tebgn,
                                box_group_index,
                            ),
                            "r",
                        ) as cur_res_f:
                            cur_res_str = "[{}] ".format(cur_dir.split("/")[-1]) + cur_res_f.readlines()[-1]
                            f.write("cur_dir: {}\n".format(cur_dir))
                            f.write(cur_res_str)
                            f.write("-" * 50 + "\n")
                    print("#" * 100)
                    print("running times: ", len(MAE_list))
                    print("all mt path: ", all_metric_save_path)
                    print_str = "[Avg] [{}] num_imgs: {}, run_times: {}, MAE_MEAN: {:.2f}, MAE_STD: {:.2f}, RMSE_MEAN: {:.2f}, RMSE_STD: {:.2f}, POS_ERR: {:.2f}, NEG_ERR: {:.2f}".format(
                        test_split,
                        int(all_sample_num),
                        len(MAE_list),
                        np.mean(MAE_list),
                        np.std(MAE_list),
                        np.mean(RMSE_list),
                        np.std(RMSE_list),
                        np.mean(POS_ERR_list),
                        np.mean(NEG_ERR_list),
                    )
                    print(print_str)
                    f.write(print_str + "\n")
                    f.write("=" * 100 + "\n\n")
    else:
        print("[!] current dir_list is {}.".format(dir_list))

print("@" * 100)
print("Finished. Cost Time: {}.".format(time.strftime("%H:%M:%S", gmtime(time.time() - st_time))))
