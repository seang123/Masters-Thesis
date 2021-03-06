import subprocess as sp
import re
import time
import datetime

def get_smi_out():
    x = sp.run(["nvidia-smi"], capture_output=True)
    return x.stdout.decode("utf-8")

def get_memory_usage():
    """ Query nvidia-smi for current gpu memory usage

    Return:
        list of memory usages values as strings
    """
    # run nvidia-smi command and capture output
    data = get_smi_out()

    # match mem usage string
    y = re.findall(r'(\d+MiB)', data)

    # get just the in-use memory stat
    mem_usage = [y[i] for i in range(0, len(y)) if i in [0,2,4]]

    # split number from MiB
    mem = []
    for i in mem_usage:
        mem.append(re.search(r'\d+', i)[0])

    return mem

def get_volatile_util():
    """Returns volatile-gpu usage percentage

    Sometimes gpu-fan % shows 'ERR!' which results in this function not returning proper values for each gpu
    """
    data = get_smi_out()

    y = re.findall(r'(\d+%)', data)

    util_usage = [y[i] for i in range(0, len(y)) if i in [1,3,5]]

    util = []
    for i in util_usage:
        util.append(re.search(r'\d+', i)[0])

    return util


def monitor():

    start = time.time()
    c = 0
    mem_blocked = True
    gpu_to_use = 1
    while mem_blocked:

        mem = get_memory_usage()

        for i in range(0, len(mem)):
            if int(mem[i]) <= 1600:
                mem_blocked = False
                gpu_to_use = i
                ts = datetime.datetime.now().strftime('%H:%M:%S - %d/%m/%Y')
                print(f"\n## Training on gpu {i} at {ts} ##")
                break

        if mem_blocked:
            print(f"waiting 5 more seconds | epoch {c} | {mem} | {(time.time() - start):.2f}", end='\r')
            time.sleep(5)
        
        c += 1

    return gpu_to_use


if __name__ == '__main__':

    mem = get_memory_usage()

    util = get_volatile_util()

    print(list(zip(mem, util)))
