# a function with profiles functions. i,e,. times the function and measures cpu, gpu and ram usage

import time
import psutil
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

def func_to_profile(x):
    return np.sin(x)

def function_profiler(f, *args, **kwargs):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_ram = psutil.virtual_memory().percent
        start_gpu = cp.cuda.memory.get_used_bytes()
        result = f(*args, **kwargs)
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_ram = psutil.virtual_memory().percent
        end_gpu = cp.cuda.memory.get_used_bytes()
        print('CPU usage: ' + str(end_cpu - start_cpu) + '%')
        print('RAM usage: ' + str(end_ram - start_ram) + '%')
        print('GPU usage: ' + str(end_gpu - start_gpu) + ' bytes')
        print('Time: ' + str(end_time - start_time) + ' seconds')
        return result
    return wrapper

# now we can profile any function
@function_profiler
def profiled_function(x):
    return func_to_profile(x)

profiled_function(np.arange(100000))