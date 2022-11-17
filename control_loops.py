from numba import cuda
from numba import njit, types, vectorize, prange
from numba.experimental import jitclass
from numba import int32, float64    # import the types
import numpy as np
import time
import math

def speed_control_loop(ACM_List, CTRL_List, reg_speed_List, ACM_num, blocks_per_grid, threads_per_block):
    # 关于速度的参数
    for i in range(ACM_num):
        reg_speed_List[i].Ref = CTRL_List[i].cmd_rpm_speed / 60 * 2 * np.pi * ACM_List[i].npp
        reg_speed_List[i].Fbk = CTRL_List[i].omega_elec

    # 参数数组化
    err = [reg_speed_List[i].Err for i in range(ACM_num)]
    errPrev = [reg_speed_List[i].ErrPrev for i in range(ACM_num)]
    ref = [reg_speed_List[i].Ref for i in range(ACM_num)]
    fbk = [reg_speed_List[i].Fbk for i in range(ACM_num)]
    out = [reg_speed_List[i].Out for i in range(ACM_num)]
    outPrev = [reg_speed_List[i].OutPrev for i in range(ACM_num)]
    kp = [reg_speed_List[i].Kp for i in range(ACM_num)]
    ki = [reg_speed_List[i].Ki for i in range(ACM_num)]

    # host to device 内存拷贝
    speed_err_deivce = cuda.to_device(err)
    speed_errPrev_deivce = cuda.to_device(errPrev)
    speed_ref_device = cuda.to_device(ref)
    speed_fbk_deivce = cuda.to_device(fbk)
    speed_out_device = cuda.to_device(out)
    speed_outPrev_device = cuda.to_device(outPrev)
    speed_kp_device = cuda.to_device(kp)
    speed_ki_device = cuda.to_device(ki)

    # 速度环
    incremental_pi[blocks_per_grid, threads_per_block](speed_err_deivce, speed_errPrev_deivce, speed_ref_device,
                                                       speed_fbk_deivce, speed_out_device, speed_outPrev_device,
                                                       speed_kp_device, speed_ki_device, ACM_num)
    # device to host 内存拷贝
    ary = np.empty(shape=speed_err_deivce.shape, dtype=speed_errPrev_deivce.dtype)
    speed_err_deivce.copy_to_host(ary)
    for i in range(ACM_num):
        reg_speed_List[i].Err = ary[i]

    ary = np.empty(shape=speed_errPrev_deivce.shape, dtype=speed_errPrev_deivce.dtype)
    speed_errPrev_deivce.copy_to_host(ary)
    for i in range(ACM_num):
        reg_speed_List[i].ErrPrev = ary[i]

    ary = np.empty(shape=speed_ref_device.shape, dtype=speed_ref_device.dtype)
    speed_ref_device.copy_to_host(ary)
    for i in range(ACM_num):
        reg_speed_List[i].Ref = ary[i]

    ary = np.empty(shape=speed_fbk_deivce.shape, dtype=speed_fbk_deivce.dtype)
    speed_fbk_deivce.copy_to_host(ary)
    for i in range(ACM_num):
        reg_speed_List[i].Fbk = ary[i]

    ary = np.empty(shape=speed_out_device.shape, dtype=speed_out_device.dtype)
    speed_out_device.copy_to_host(ary)
    for i in range(ACM_num):
        reg_speed_List[i].Out = ary[i]

    ary = np.empty(shape=speed_outPrev_device.shape, dtype=speed_outPrev_device.dtype)
    speed_outPrev_device.copy_to_host(ary)
    for i in range(ACM_num):
        reg_speed_List[i].OutPrev = ary[i]

    ary = np.empty(shape=speed_kp_device.shape, dtype=speed_kp_device.dtype)
    speed_kp_device.copy_to_host(ary)
    for i in range(ACM_num):
        reg_speed_List[i].Kp = ary[i]

    ary = np.empty(shape=speed_ki_device.shape, dtype=speed_ki_device.dtype)
    speed_ki_device.copy_to_host(ary)
    for i in range(ACM_num):
        reg_speed_List[i].Ki = ary[i]

def current_control_loop(reg_speed_List, CTRL_List, reg_i_List, ACM_num, blocks_per_grid, threads_per_block, cur_name):
    # 参数数组化
    # 判断是id还是iq
    if cur_name == "id":
        pid_i_ref = np.zeros(ACM_num)
        pid_i_fbk = [CTRL_List[i].idq[0] for i in range(ACM_num)]
    else:
        pid_i_ref = [reg_speed_List[i].Out for i in range(ACM_num)]
        pid_i_fbk = [CTRL_List[i].idq[1] for i in range(ACM_num)]
    # 数组化
    pid_i_err = [reg_i_List[i].Err for i in range(ACM_num)]
    pid_i_errPrev = [reg_i_List[i].ErrPrev for i in range(ACM_num)]
    pid_i_out = [reg_i_List[i].Out for i in range(ACM_num)]
    outPrev = [reg_i_List[i].OutPrev for i in range(ACM_num)]
    Kp = [reg_i_List[i].Kp for i in range(ACM_num)]
    Ki = [reg_i_List[i].Ki for i in range(ACM_num)]

    # host to device 内存拷贝
    pid_cur_err_deivce = cuda.to_device(pid_i_err)
    pid_cur_errPrev_deivce = cuda.to_device(pid_i_errPrev)
    pid_cur_ref_deivce = cuda.to_device(pid_i_ref)
    pid_cur_fbk_deivce = cuda.to_device(pid_i_fbk)
    pid_cur_out_device = cuda.to_device(pid_i_out)
    pid_cur_outPrev_device = cuda.to_device(outPrev)
    pid_cur_kp_device = cuda.to_device(Kp)
    pid_cur_ki_device = cuda.to_device(Ki)
    # 电流环

    start_time = time.time()

    incremental_pi[blocks_per_grid, threads_per_block](pid_cur_err_deivce, pid_cur_errPrev_deivce, pid_cur_ref_deivce,
                                                       pid_cur_fbk_deivce, pid_cur_out_device, pid_cur_outPrev_device,
                                                       pid_cur_kp_device, pid_cur_ki_device, ACM_num)

    # print("--- %s seconds ---" % (time.time() - start_time))


    # device to host 内存拷贝
    ary = np.empty(shape=pid_cur_err_deivce.shape, dtype=pid_cur_err_deivce.dtype)
    pid_cur_err_deivce.copy_to_host(ary)
    for i in range(ACM_num):
        reg_i_List[i].Err = ary[i]

    ary = np.empty(shape=pid_cur_errPrev_deivce.shape, dtype=pid_cur_errPrev_deivce.dtype)
    pid_cur_errPrev_deivce.copy_to_host(ary)
    for i in range(ACM_num):
        reg_i_List[i].ErrPrev = ary[i]

    ary = np.empty(shape=pid_cur_ref_deivce.shape, dtype=pid_cur_ref_deivce.dtype)
    pid_cur_ref_deivce.copy_to_host(ary)
    for i in range(ACM_num):
        reg_i_List[i].Ref = ary[i]

    ary = np.empty(shape=pid_cur_fbk_deivce.shape, dtype=pid_cur_fbk_deivce.dtype)
    pid_cur_fbk_deivce.copy_to_host(ary)
    for i in range(ACM_num):
        reg_i_List[i].Fbk = ary[i]

    ary = np.empty(shape=pid_cur_out_device.shape, dtype=pid_cur_out_device.dtype)
    pid_cur_out_device.copy_to_host(ary)
    for i in range(ACM_num):
        reg_i_List[i].Out = ary[i]

    ary = np.empty(shape=pid_cur_outPrev_device.shape, dtype=pid_cur_outPrev_device.dtype)
    pid_cur_outPrev_device.copy_to_host(ary)
    for i in range(ACM_num):
        reg_i_List[i].OutPrev = ary[i]

    ary = np.empty(shape=pid_cur_kp_device.shape, dtype=pid_cur_kp_device.dtype)
    pid_cur_kp_device.copy_to_host(ary)
    for i in range(ACM_num):
        reg_i_List[i].Kp = ary[i]

    ary = np.empty(shape=pid_cur_ki_device.shape, dtype=pid_cur_ki_device.dtype)
    pid_cur_ki_device.copy_to_host(ary)
    for i in range(ACM_num):
        reg_i_List[i].Ki = ary[i]


@cuda.jit
def incremental_pi(err, errPrev, ref, fbk, out, outPrev, kp, ki, kernals_dim):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < kernals_dim:
        err[idx] = ref[idx] - fbk[idx]
        out[idx] = outPrev[idx] + \
                  kp[idx] * (err[idx] - errPrev[idx]) + \
                  ki[idx] * errPrev[idx]