import matplotlib.pyplot as plt

from acmsimpy import *
END = 6

@njit(nogil=True, parallel=False)  # THIS IS ACTUALLY FASTER
def RUN(end, TIME=5.5, MACHINE_TS=1e-4, CL_TS=1e-4):
    control_times = np.arange(0, TIME, CL_TS)
    list_speed = [np.zeros_like(control_times) for i in range(end-1)]
    parallel_index = 0
    for omega_ob in prange(1, int(end)):
        print("omega_ob = ", omega_ob, ' | ')
        controL_times, id, iq, ia, ib, speed = ACMSimPy(omega_ob=omega_ob, TIME=TIME, MACHINE_TS=MACHINE_TS, CL_TS=CL_TS)
        list_speed[parallel_index] = speed
        parallel_index += 1
    return controL_times, list_speed
end = END
controL_times, list_speed = RUN(end=end, TIME=5.5, MACHINE_TS=1e-4)


""" PLOT """
# plot speed curve respectively
plt.figure(figsize=(12,6))
for i in range(len(list_speed)):
    print('\t', min(list_speed[i]))
    plt.plot(controL_times, list_speed[i], label=str(i))
plt.legend(loc='upper left')
plt.show()


# THIS IS NOT WORKING AS PARALLEL!!!
@njit(nogil=True, parallel=True)
def RUN(end=END):
    for omega_ob in prange(1, int(end)):
        ACMSimPy(omega_ob=omega_ob, TIME=5.5, MACHINE_TS=1e-4, CL_TS=1e-4)
# run multiple motor simulations in sequence
RUN()
#RUN.parallel_diagnostics(level=4)

import multiprocessing
print(multiprocessing.cpu_count())
print('Conclusion: numba.jitclass is likely to not support multiprocessing.')

processes = []
for omega_ob in range(1,3):
    p = multiprocessing.Process(target=ACMSimPy, args=[omega_ob])
    p.start()
    processes.append(p)

for p in processes:
    p.join()
