#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2022/07/26 09:30
# @Author  : Allen Guo
# @Email   : 1309476338@qq.com
# @Profile : https://wilixx.github.io
# @File    : time_slot_allocation_xxx.py
# @Function: BinaryTree-based time slot allocation algorithm.

__author__ = 'Allen Guo'
import numpy as np
import os
from collections import defaultdict
from itertools import product
import datetime
import pickle
import argparse

# time_slot_num = 1536
time_slot_num = 512
# time_slot_num = 8
np.random.seed(2022)
cell_list = ["cell_1", "cell_2", "cell_3"]

cell_dir = "cell_dir\\"
def bt_init(cell_list = ["cell_1", "cell_2", "cell_3"]):
    if not os.path.exists(cell_dir):
        os.mkdir(cell_dir)

    for cell_id in cell_list:
        if os.path.exists(cell_dir+cell_id + "_allocation_A.pkl"):
            binary_tree_cpp = pickle.load(open(cell_dir + "btree.pkl", 'rb'))
            binary_tree_cpp_standard_form = pickle.load(open(cell_dir + "btreeIDs.pkl", 'rb'))
            status_label_tree_cpp = pickle.load(open(cell_dir + cell_id + "_allocation_A.pkl", 'rb'))
            # status_label_tree_cpp = pickle.load(open(cell_dir + cell_id + "_allocation_B.pkl", 'rb'))
            # status_label_tree_cpp = pickle.load(open(cell_dir + cell_id + "_allocation_C.pkl", 'rb'))
            break
        time_slot_vector = np.array([i+1 for i in range(time_slot_num)])
        status_label_vector = np.zeros_like(time_slot_vector)
        actual_status_label_vector = np.zeros_like(time_slot_vector)

        # print("time_slot_vector :", time_slot_vector)

        """ 补充数据结构 """
        big_num = time_slot_num
        layer_slots_pairs = defaultdict(int)
        layer_index = 0
        while big_num % 2 == 0:
            layer_index += 1
            layer_slots_pairs[layer_index] = big_num
            big_num = big_num / 2
        layer_slots_pairs[layer_index] = big_num

        # print("layer_slots_pairs :", layer_slots_pairs)
        # exit()
        """ 补充数据结构 """
        binary_tree = defaultdict(list)
        status_label_tree = defaultdict(list)

        layer_num = np.floor(np.log2(time_slot_num))  # 修正函数如下
        layer_num = layer_num if (time_slot_num / (2**(layer_num)) == 1) else layer_num-1

        leaves_num = time_slot_num / (2**(layer_num))
        # print("layer_num :", layer_num)
        # print("leaves_num :", leaves_num)

        binary_tree[1].append(time_slot_vector)
        status_label_tree[1].append(np.zeros_like(time_slot_vector))
        # print("binary_tree :", binary_tree)
        # exit()
        for layer_id in range(2, int(layer_num)+2):  # 从第二层开始构建树
            # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # print("layer_id: ", layer_id)

            for sub_branch_id in range(int(2**(layer_id-2))):
                # print("sub_branch_id: ", sub_branch_id)
                binary_tree[layer_id].append(binary_tree[layer_id-1][sub_branch_id][0::2])
                binary_tree[layer_id].append(binary_tree[layer_id-1][sub_branch_id][1::2])

                """ 状态信息树  """
                status_label_tree[layer_id].append(np.zeros_like(binary_tree[layer_id-1][sub_branch_id][0::2]))
                status_label_tree[layer_id].append(np.zeros_like(binary_tree[layer_id-1][sub_branch_id][1::2]))

        # print("binary_tree :", binary_tree)
        # print("status_label_tree :", status_label_tree)

        binary_tree_cpp = defaultdict(list)
        status_label_tree_cpp = defaultdict(list)
        binary_tree_cpp_standard_form = defaultdict(list)

        for k, v in binary_tree.items():
            binary_tree_cpp[10-k] = tuple([tuple(item-1) for item in v])
            binary_tree_cpp_standard_form[10 - k] = tuple(["A-" + str(item[0]-1) + "-" + str(10-k) for item in v])

        for k, v in status_label_tree.items():
            status_label_tree_cpp[10-k] = list([list(item) for item in v])

        # print("binary_tree_cpp :", binary_tree_cpp)
        # print("status_label_tree_cpp :", status_label_tree_cpp)
        # print("binary_tree_cpp_standard_form :", binary_tree_cpp_standard_form)

        pickle.dump(binary_tree_cpp, open(cell_dir + "btree.pkl",'wb'))
        pickle.dump(binary_tree_cpp_standard_form, open(cell_dir + "btreeIDs.pkl",'wb'))
        pickle.dump(status_label_tree_cpp, open(cell_dir+cell_id + "_allocation_A.pkl", 'wb'))
        pickle.dump(status_label_tree_cpp, open(cell_dir+cell_id + "_allocation_B.pkl", 'wb'))
        pickle.dump(status_label_tree_cpp, open(cell_dir+cell_id + "_allocation_C.pkl", 'wb'))

# bt_init()
# exit()
"""
Step-3: Service generate and scheduling based on binary tree.
"""
""" Source, sink, frequency """
"""
Step-4: Timeslot allocation on binary tree.
"""

start = datetime.datetime.now()
# print(">>> Start time is: {} ----------------------------------".format(start))

cell_id = "cell_1"
def binary_tree_allocation(cell_id, frequency):
    for group_label_temp in ["A", "B", "C"]:
        print("group_label_temp:", group_label_temp)
        group_label = group_label_temp
        if not os.path.exists(cell_dir + cell_id + "_allocation_" + group_label + ".pkl"):
            bt_init(cell_list=[cell_id])

        binary_tree_cpp = pickle.load(open(cell_dir + "btree.pkl", 'rb'))
        binary_tree_cpp_standard_form = pickle.load(open(cell_dir + "btreeIDs.pkl", 'rb'))
        status_label_tree_cpp = pickle.load(open(cell_dir + cell_id + "_allocation_" + group_label + ".pkl", 'rb'))

        frequency = frequency
        feasible_layer_id = 0

        for layer_id in range(0, len(list(binary_tree_cpp.keys())), 1):
            print("layer_id:", layer_id)
            """ 首先判断时隙数目 是否足够多，然后判断当前有没有剩余时隙未分配，如果没有再看上一层，实在没有就拒绝 """
            if 2 ** layer_id < frequency:
                continue
            if 2 ** layer_id >= frequency: # 找到了一个时隙，尝试分配之
                feasible_layer_id = layer_id
                # print("feasible_layer_id", feasible_layer_id)

            for group_id, slots_group_temp in enumerate(status_label_tree_cpp[feasible_layer_id]):
                if np.sum(slots_group_temp) > 0:
                    print("slots_group_temp", slots_group_temp)
                    continue
                if np.sum(slots_group_temp) == 0:
                    print("Time slot allocation successfully. ")
                    # print("Yes slots_group_temp)", slots_group_temp)

                    # exit()
                    """ 开始时隙分配过程 """
                    for slot_i in range(frequency):
                        """ actual_status_label_vector 用来统计时隙资源利用率 """
                        # status_label_tree_cpp[feasible_layer_id][group_id][slot_i] += 1 # 重复了
                        the_allocated_slot_name = binary_tree_cpp[feasible_layer_id][group_id][slot_i]
                        for layer_x in range(0, len(list(binary_tree_cpp.keys())), 1):
                            """ 首先判断时隙数目 是否足够多，然后判断当前有没有剩余时隙未分配，如果没有再看上一层，实在没有就拒绝 """
                            # print("Yes", binary_tree_cpp[layer_x])

                            for group_x, slots_xx in enumerate(binary_tree_cpp[layer_x]):
                                # print("binary_tree_cpp[layer_x][group_x]:", binary_tree_cpp[layer_x][group_x])
                                for slot_index, slot_t in enumerate(slots_xx):
                                    # print("binary_tree_cpp[layer_x][group_x][slot_index]=", binary_tree_cpp[layer_x][group_x][slot_index])
                                    if slot_t == the_allocated_slot_name:
                                        status_label_tree_cpp[layer_x][group_x][slot_index] += 1

                                        # print("status_label_tree_cpp[layer_x][group_x][slot_index]=", status_label_tree_cpp[layer_x][group_x][slot_index])
                                        assert status_label_tree_cpp[layer_x][group_x][slot_index] == 1

                    pickle.dump(status_label_tree_cpp, open(cell_dir + cell_id + "_allocation_" + group_label + ".pkl", 'wb'))
                    print("Allocated time slot block_id: ", group_label+binary_tree_cpp_standard_form[feasible_layer_id][group_id][1:])
                    print("Allocated time slot : ", binary_tree_cpp[feasible_layer_id][group_id])
                    return group_label+binary_tree_cpp_standard_form[feasible_layer_id][group_id][1:]

    print("No enough time slot in cell: {} . ".format(cell_id))
    return -1

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--cell_id', type=str, default="cell_1")
    arg.add_argument('--freq', type=int, default=167)
    arg = arg.parse_args()

    binary_tree_allocation(arg.cell_id, arg.freq)
    end = datetime.datetime.now()
    # print(">>> duration time is: {} --------------------------".format(end-start))
