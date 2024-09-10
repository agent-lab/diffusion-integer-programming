import os
import numpy as np
def cal_success_ratio(logs_num, instance_name, model_type, coverage):
    fail_num = 0.
    for i in range(logs_num):
        log_path = f'agents/solving_logs/IS_{model_type}_selectiveNet{coverage}_{i}.log'
        log_file = open(log_path, "r", encoding='UTF-8')
        lines = log_file.readlines()  # 读取文件的所有行
        assert  len(lines) >= 0, '文件行数不足'
        second_line = lines[1].split()  # 获取第二行内容
        if "failed" in second_line:  # 判断第二行是否包含"fail"单词
            fail_num += 1
    return fail_num/logs_num

if __name__ == '__main__':
    # logs_num = 100
    # instance_name = '4_independent_set'
    # model_type = 'ddpm'
    # coverage = 0.2
    # print(cal_success_ratio(logs_num, instance_name, model_type, coverage))
    instance = "SC"
    size = 3000
    solver = "ps_gurobi"

    if instance == 'SC':
        instance_file = "1_set_cover"
        log_name = "SC_instance"
        start = 900
        if size != None:
            instance_file = f"1_set_cover_{size}"
            start = 0
    elif instance == 'CA':
        instance_file = '2_combinatorial_auction'
        log_name = "CA_instance"
        start = 900
    elif instance == 'CF':
        instance_file = '3_capacity_facility'
        log_name = "CF_instance"
        start = 900
    elif instance == 'IS':
        instance_file = '4_independent_set'
        log_name = "IS_instance"
        start = 0

    obj_vals = []
    best_objs = []
    for i in range(100):
        log_path = f'logs/{solver}_logs/{instance_file}/{instance_file[2:]}_{start+i}.log'
        log_file = open(log_path, "r", encoding='UTF-8')
        lines = log_file.readlines()  # 读取文件的所有行
        if solver == 'scip':
            idx = 0
            for line in lines:
                idx += 1 
                if "time" in line:
                    break
            obj_val = float(lines[idx].split("|")[-3])
            obj_vals.append(obj_val)
        else:
            for line in lines:
                if 'heuristic' in line:
                    obj_val = float(line.split()[-1])
                if 'Best' in line and 'objective' in line:
                    best_obj = float(line.split()[2][:-1])
            obj_vals.append(obj_val)
            best_objs.append(best_obj)
    obj_vals = np.array(obj_vals)
    best_objs = np.array(best_objs)
    print(obj_vals.mean())
    print(best_objs.mean())