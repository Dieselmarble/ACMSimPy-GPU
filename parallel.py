from numba import njit, types, vectorize, prange
from numba.experimental import jitclass
from numba import int32, float64    # import the types
from matplotlib import pyplot as plt
import numpy as np
from control_loops import *
import math
import pdb
import time

@jitclass(
    spec=[
        # name plate data
        ('npp',   int32),
        ('npp_inv', float64),
        ('IN',  float64),
        # electrical parameters
        ('R',   float64),
        ('Ld',  float64),
        ('Lq',  float64),
        ('KE',  float64),
        # mechanical parameters
        ('Js',  float64),
        ('Js_inv', float64),
        ('B',   float64),
        # states
        ('NS',    int32),
        ('x',   float64[:]),
        ('Tem', float64),
        # inputs
        ('uab',   float64[:]),
        ('udq',   float64[:]),
        ('TLoad', float64),
        # output
        ('cosT', float64),
        ('sinT', float64),
        ('theta_d', float64),
        ('theta_mech', float64),
        ('omega_elec', float64),
        ('iab', float64[:]),
    ])
class The_AC_Machines:
    def __init__(self):
        # name plate data
        self.npp = 4
        self.npp_inv = 1.0/self.npp
        self.IN = 3 # Arms (line-to-line)
        # electrical parameters
        self.R = 1.1
        self.Ld = 5e-3
        self.Lq = 5e-3
        self.KE = 0.095
        # mechanical parameters
        self.Js = 0.0006168  # kg.m^2
        self.Js_inv = 1.0/self.Js
        self.B  = 0*0.7e-3 # Nm.s
        # states
        self.NS = 5
        self.x = np.zeros(self.NS, dtype=np.float64)
        self.Tem = 0.0
        # inputs
        self.uab = np.zeros(2, dtype=np.float64)
        self.udq = np.zeros(2, dtype=np.float64)
        self.TLoad = 0
        # output
        self.cosT = 0.0
        self.sinT = 0.0
        self.theta_d = 0.0
        self.theta_mech = 0.0
        self.omega_elec = 0.0
        self.iab = np.zeros(2, dtype=np.float64)

@jitclass(
    spec=[
        # constants
        ('CL_TS', float64),
        ('VL_TS', float64),
        # feedback / input
        ('theta_d', float64),
        ('omega_elec', float64),
        ('iab', float64[:]),
        # states
        ('timebase', float64),
        ('cosT', float64),
        ('sinT', float64),
        ('idq', float64[:]),
        # commands
        ('idq_cmd', float64[:]),
        ('udq_cmd', float64[:]),
        ('uab_cmd', float64[:]),
        ('cmd_rpm_speed', float64),
    ])

class The_Motor_Controller:
    def __init__(self, CL_TS, VL_TS):
        # constants
        self.CL_TS = CL_TS
        self.VL_TS = VL_TS
        # feedback / input
        self.theta_d = 0.0
        self.omega_elec = 0.0
        self.iab = np.zeros(2, dtype=np.float64)
        # states
        self.timebase = 0.0
        self.cosT = 0.0
        self.sinT = 0.0
        self.idq = np.zeros(2, dtype=np.float64)
        # commands
        self.idq_cmd = np.zeros(2, dtype=np.float64)
        self.udq_cmd = np.zeros(2, dtype=np.float64)
        self.uab_cmd = np.zeros(2, dtype=np.float64)
        self.cmd_rpm_speed = 0.0;

@jitclass(
    spec=[
        ('Kp', float64),
        ('Ki', float64),
        ('Err', float64),
        ('Ref', float64),
        ('Fbk', float64),
        ('Out', float64),
        ('OutLimit', float64),
        ('ErrPrev', float64),
        ('OutPrev', float64),
    ])
class The_PI_Regulator:
    def __init__(self, KP_CODE, KI_CODE, OUTPUT_LIMIT):
        self.Kp = KP_CODE
        self.Ki = KI_CODE
        self.Err = 0.0
        self.Ref = 0.0
        self.Fbk = 0.0
        self.Out = 0.0
        self.OutLimit = OUTPUT_LIMIT
        self.ErrPrev = 0.0
        self.OutPrev = 0.0

# @njit(nogil=True)
def MACHINE_DYNAMICS(x, ACM, CLARKE_TRANS_TORQUE_GAIN=1.5):
    fx = np.zeros(5)
    # 电磁子系统 (id, iq)
    # x[0]: i_d
    # x[1]: i_q
    # x[2]: w_r
    # K_E: amplitude of the flux
    # npp: number of pole pairs

    fx[0] = (ACM.udq[0] - ACM.R * x[0] + x[2]*ACM.Lq*x[1]) / ACM.Ld # d(i_d)/dt
    fx[1] = (ACM.udq[1] - ACM.R * x[1] - x[2]*ACM.Ld*x[0] - x[2]*ACM.KE) / ACM.Lq # d(i_q)/dt
    ACM.Tem = CLARKE_TRANS_TORQUE_GAIN * ACM.npp * \
        (x[1]*ACM.KE + (ACM.Ld - ACM.Lq)*x[0]*x[1]) # 电磁转矩 Torque 计算
    # fx[2] electric angular speed differentiation w_r
    fx[2] = (ACM.Tem - ACM.TLoad - ACM.B * x[2]) * ACM.npp/ACM.Js # d(w_r)/dt
    fx[3] = x[2]          # w_r d(theta)d/t  angular velocity
    fx[4] = x[2]/ACM.npp  # w_r/npp
    return fx

# 四阶龙格库塔法
def RK4(t, ACM, hs):
    NS = ACM.NS # NS = 5
    k1, k2, k3, k4 = np.zeros(NS), np.zeros(NS), np.zeros(NS), np.zeros(NS)
    xk, fx = np.zeros(NS), np.zeros(NS)
    for i in range(NS):
        k1[i] = fx[i] * hs
        xk[i] = ACM.x[i] + k1[i]*0.5 # 1/2时间间隔，仿真的最小时间间隔为1

    fx = MACHINE_DYNAMICS(xk, ACM)  # @timer.t+hs/2., # Euler method
    for i in range(NS):
        k2[i] = fx[i] * hs
        xk[i] = ACM.x[i] + k2[i]*0.5

    fx = MACHINE_DYNAMICS(xk, ACM)
    for i in range(NS):
        k3[i] = fx[i] * hs
        xk[i] = ACM.x[i] + k3[i]

    fx = MACHINE_DYNAMICS(xk, ACM)  # timer.t+hs,
    for i in range(NS):
        k4[i] = fx[i] * hs
        ACM.x[i] = ACM.x[i] + (k1[i] + 2*(k2[i] + k3[i]) + k4[i])/6.0

class The_ACM_current:
    def __init__(self, control_times):
        self.id = np.zeros_like(control_times)
        self.iq = np.zeros_like(control_times)
        self.ia = np.zeros_like(control_times)
        self.ib = np.zeros_like(control_times)
        self.speed = np.zeros_like(control_times)

class The_AC_Machines:
    def __init__(self):
        # name plate data
        self.npp = 4
        self.npp_inv = 1.0 / self.npp
        self.IN = 3  # Arms (line-to-line)
        # electrical parameters
        self.R = 1.1
        self.Ld = 5e-3
        self.Lq = 5e-3
        self.KE = 0.095
        # mechanical parameters
        self.Js = 0.0006168  # kg.m^2
        self.Js_inv = 1.0 / self.Js
        self.B = 0 * 0.7e-3  # Nm.s
        # states
        self.NS = 5
        self.x = np.zeros(self.NS, dtype=np.float64)
        self.Tem = 0.0
        # inputs
        self.uab = np.zeros(2, dtype=np.float64)
        self.udq = np.zeros(2, dtype=np.float64)
        self.TLoad = 0
        # output
        self.cosT = 0.0
        self.sinT = 0.0
        self.theta_d = 0.0
        self.theta_mech = 0.0
        self.omega_elec = 0.0
        self.iab = np.zeros(2, dtype=np.float64)

@jitclass(
    spec=[
        # constants
        ('CL_TS', float64),
        ('VL_TS', float64),
        # feedback / input
        ('theta_d', float64),
        ('omega_elec', float64),
        ('iab', float64[:]),
        # states
        ('timebase', float64),
        ('cosT', float64),
        ('sinT', float64),
        ('idq', float64[:]),
        # commands
        ('idq_cmd', float64[:]),
        ('udq_cmd', float64[:]),
        ('uab_cmd', float64[:]),
        ('cmd_rpm_speed', float64),
    ])

class The_Motor_Controller:
    def __init__(self, CL_TS, VL_TS):
        # constants
        self.CL_TS = CL_TS
        self.VL_TS = VL_TS
        # feedback / input
        self.theta_d = 0.0
        self.omega_elec = 0.0
        self.iab = np.zeros(2, dtype=np.float64)
        # states
        self.timebase = 0.0
        self.cosT = 0.0
        self.sinT = 0.0
        self.idq = np.zeros(2, dtype=np.float64)
        # commands
        self.idq_cmd = np.zeros(2, dtype=np.float64)
        self.udq_cmd = np.zeros(2, dtype=np.float64)
        self.uab_cmd = np.zeros(2, dtype=np.float64)
        self.cmd_rpm_speed = 0.0;

class The_PI_Regulator:
    def __init__(self, KP_CODE, KI_CODE, OUTPUT_LIMIT):
        self.Kp = KP_CODE
        self.Ki = KI_CODE
        self.Err = 0.0
        self.Ref = 0.0
        self.Fbk = 0.0
        self.Out = 0.0
        self.OutLimit = OUTPUT_LIMIT
        self.ErrPrev = 0.0
        self.OutPrev = 0.0

# @njit(nogil=True)
def initilize_all_motors(ACM_num, CL_TS, control_times, TIME):
    # 多电机集合
    ACM_List = []
    ACM_current_List = []
    CTRL_List = []
    reg_id_List = []
    reg_iq_List = []
    reg_speed_List = []
    speed_List = []
    # 遍历所有电机
    for ACM_id in range(ACM_num):
        # 初始化电机物理模型
        ACM = The_AC_Machines()
        ACM_List.append(ACM)
        # 电机控制指令
        CTRL = The_Motor_Controller(CL_TS, 4 * CL_TS)
        CTRL_List.append(CTRL)
        # 初始化控制电流
        ACM_current = The_ACM_current(control_times)
        ACM_current_List.append(ACM_current)
        # 初始化PI控制器参数
        omega_ob = 1
        reg_id = The_PI_Regulator(6.39955, 6.39955 * 237.845 * CTRL.CL_TS, 600)
        reg_iq = The_PI_Regulator(6.39955, 6.39955 * 237.845 * CTRL.CL_TS, 600)
        reg_speed = The_PI_Regulator(omega_ob * 0.0380362, omega_ob * 0.0380362 * 30.5565 * CTRL.VL_TS,
                                     1 * 1.414 * ACM.IN)
        reg_id_List.append(reg_id)
        reg_iq_List.append(reg_iq)
        reg_speed_List.append(reg_speed)
        speed_Time_List = np.zeros_like(control_times)
        speed_List.append(speed_Time_List)

    return ACM_List, ACM_current_List, CTRL_List, reg_id_List, reg_iq_List, reg_speed_List, speed_List


# @njit(nogil=True)
def null_id_control(ACM_List, CTRL_List, reg_id_List, reg_iq_List, reg_speed_List, ACM_num):
    # 并行线程数目n
    n = ACM_num
    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)
    # 速度环
    speed_control_loop(ACM_List, CTRL_List, reg_speed_List, ACM_num, blocks_per_grid, threads_per_block)
    # id电流环
    current_control_loop(reg_speed_List, CTRL_List, reg_id_List, ACM_num, blocks_per_grid, threads_per_block, "id")
    # id电流输出
    for i in range (ACM_num):
        CTRL_List[i].udq_cmd[0] = reg_id_List[i].Out
    # iq电流环
    for i in range(ACM_num):
        reg_id_List[i].Ref = reg_speed_List[i].Out
        reg_id_List[i].Fbk = CTRL_List[i].idq[1]
    current_control_loop(reg_speed_List, CTRL_List, reg_iq_List, ACM_num, blocks_per_grid, threads_per_block, "iq")
    # iq电流输出
    for i in range(ACM_num):
        CTRL_List[i].udq_cmd[1] = reg_iq_List[i].Out

# @njit(nogil=True)
def main(ACM_num=1024):
    # 总仿真时间
    TIME = 8
    # 时间精度
    MACHINE_TS = 1e-4
    CL_TS = 1e-3
    # 物理时间轴
    machine_times = np.arange(0, TIME, MACHINE_TS)
    # 控制时间轴
    control_times = np.arange(0, TIME, CL_TS)
    # 多电机集合 reg是给PID用的
    ACM_List, ACM_current_List, CTRL_List, reg_id_List, reg_iq_List, reg_speed_List, speed_List = \
        initilize_all_motors(ACM_num, CL_TS, control_times, TIME)
    # 控制频率与电机频率
    down_sampling_ceiling = int(CL_TS / MACHINE_TS)
    print('\tdown sample:', down_sampling_ceiling)
    # 观察时间轴
    watch_index = 0
    # 下采样时间轴
    jj = 0
    # 主时间轴
    for ii in range(len(machine_times)):
        # 电机时钟时间轴
        machine_clk = machine_times[ii]
        jj += 1
        if jj >= down_sampling_ceiling:
            jj = 0
            # 每个电机的物理状态
            for ACM_id in range(ACM_num):
                # 四阶Runge-Kutta法仿真
                RK4(machine_clk, ACM_List[ACM_id], hs=MACHINE_TS)
                # 读取电机模型反馈参数
                ACM_List[ACM_id].omega_elec = ACM_List[ACM_id].x[2]
                ACM_List[ACM_id].theta_d = ACM_List[ACM_id].x[3]
                ACM_List[ACM_id].theta_mech = ACM_List[ACM_id].x[4]
                ACM_List[ACM_id].cosT = np.cos(ACM_List[ACM_id].theta_d)
                ACM_List[ACM_id].sinT = np.sin(ACM_List[ACM_id].theta_d)
                # Inverse Park transformation
                ACM_List[ACM_id].iab[0] = ACM_List[ACM_id].x[0] * ACM_List[ACM_id].cosT + ACM_List[ACM_id].x[1] * -ACM_List[ACM_id].sinT
                ACM_List[ACM_id].iab[1] = ACM_List[ACM_id].x[0] * ACM_List[ACM_id].sinT + ACM_List[ACM_id].x[1] * ACM_List[ACM_id].cosT
                # 测量
                CTRL_List[ACM_id].theta_d = ACM_List[ACM_id].theta_d
                CTRL_List[ACM_id].omega_elec = ACM_List[ACM_id].omega_elec
                CTRL_List[ACM_id].iab[0] = ACM_List[ACM_id].iab[0]
                CTRL_List[ACM_id].iab[1] = ACM_List[ACM_id].iab[1]
                # do this once per control interrupt
                CTRL_List[ACM_id].cosT = np.cos(CTRL_List[ACM_id].theta_d)
                CTRL_List[ACM_id].sinT = np.sin(CTRL_List[ACM_id].theta_d)
                # Park transformation ab->dq
                CTRL_List[ACM_id].idq[0] = CTRL_List[ACM_id].iab[0] * CTRL_List[ACM_id].cosT + CTRL_List[ACM_id].iab[1] * CTRL_List[ACM_id].sinT
                CTRL_List[ACM_id].idq[1] = CTRL_List[ACM_id].iab[0] * -CTRL_List[ACM_id].sinT + CTRL_List[ACM_id].iab[1] * CTRL_List[ACM_id].cosT
                """ Console & Watch @ CL_TS """
                ACM_current_List[ACM_id].id[watch_index] = ACM_List[ACM_id].x[0]
                ACM_current_List[ACM_id].iq[watch_index] = ACM_List[ACM_id].x[1]
                ACM_current_List[ACM_id].ia[watch_index] = CTRL_List[ACM_id].iab[0]
                ACM_current_List[ACM_id].ib[watch_index] = CTRL_List[ACM_id].iab[1]
                # 周长*60转
                speed_List[ACM_id][watch_index] = CTRL_List[ACM_id].omega_elec / (2 * np.pi * ACM_List[ACM_id].npp) * 60

            # 观测时间轴递增
            watch_index += 1
            # 当所有电机跑完物理仿真后 进入控制函数
            # PID 速度环 电流环
            null_id_control(ACM_List, CTRL_List, reg_id_List, reg_iq_List, reg_speed_List, ACM_num)
            # 下发新的控制指令
            for ACM_id in range(ACM_num):
                # Inverse Park transformation
                CTRL_List[ACM_id].uab_cmd[0] = CTRL_List[ACM_id].udq_cmd[0] * CTRL_List[ACM_id].cosT + CTRL_List[ACM_id].udq_cmd[1] * -CTRL_List[ACM_id].sinT
                CTRL_List[ACM_id].uab_cmd[1] = CTRL_List[ACM_id].udq_cmd[0] * CTRL_List[ACM_id].sinT + CTRL_List[ACM_id].udq_cmd[1] * CTRL_List[ACM_id].cosT
                # Park transformation
                ACM_List[ACM_id].udq[0] = CTRL_List[ACM_id].uab_cmd[0] * ACM_List[ACM_id].cosT + CTRL_List[ACM_id].uab_cmd[1] * ACM_List[ACM_id].sinT
                ACM_List[ACM_id].udq[1] = CTRL_List[ACM_id].uab_cmd[0] * -ACM_List[ACM_id].sinT + CTRL_List[ACM_id].uab_cmd[1] * ACM_List[ACM_id].cosT
                # 调速曲线
                if machine_clk > 3:
                    CTRL_List[ACM_id].cmd_rpm_speed = 2000
                elif machine_clk > 0.0:
                    ACM_List[ACM_id].TLoad = 2
                    CTRL_List[ACM_id].cmd_rpm_speed = -200
                else:
                    CTRL_List[ACM_id].cmd_rpm_speed = 0

    print("end!")
    return control_times, reg_id_List, reg_iq_List, ACM_current_List, speed_List

if __name__ == "__main__":
    watch = main()
    control_times, reg_id_List, reg_iq_List, ACM_current_List, speed_List = watch
    plt.figure(figsize=(12, 6))
    plt.plot(control_times,  speed_List[2])
    plt.xlabel('Time')
    plt.ylabel('Rotation Speed')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(control_times, ACM_current_List[2].ia)
    plt.plot(control_times, ACM_current_List[2].ib)
    plt.xlabel('Time')
    plt.ylabel('Current a&b')
    plt.savefig("iab.jpg")
    plt.show()


    plt.figure(figsize=(12, 6))
    plt.plot(control_times, ACM_current_List[2].id)
    plt.plot(control_times, ACM_current_List[2].iq)
    plt.xlabel('Time')
    plt.ylabel('Current d&q')
    plt.savefig("idq.jpg")
    plt.show()
