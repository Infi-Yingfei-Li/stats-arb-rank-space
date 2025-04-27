#%%
import pynvml
import numpy as np
import torch

pynvml.nvmlInit()
def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2

device_status = []
print(f"NVIDIA Driver version - {pynvml.nvmlSystemGetDriverVersion()}")
deviceCount = pynvml.nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    device_status.append(get_memory_free_MiB(i))
    print(f"Device {i} {pynvml.nvmlDeviceGetName(handle)} - free memory: {get_memory_free_MiB(i)} MiB")
cuda = "cuda:{}".format(np.argmax(device_status))
device = torch.device(cuda if torch.cuda.is_available() else "cpu")
print("Use GPU: ", cuda, " with free memory: ", np.max(device_status), " MiB")
print("Device: ", device)



# %%
