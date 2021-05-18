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

    return list(map(int, mem))

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

    return list(map(int,util))


def monitor(threshold = 2000):
    """
    Wait till a gpu has <= threshold memory usage 
    """

    print(f"Monitoring GPU memory usage for availablility. Threshold set at {threshold} MiB.")

    start = time.time()
    c = 0
    mem_blocked = True
    gpu_to_use = 1

    while mem_blocked:
        mem = get_memory_usage()
        min_ = 10000
        arg_min_ = -1
        for k,v in enumerate(mem):
            if v <= min_:
                min_ = v
                arg_min_ = k

        if min_ <= threshold:
            mem_blocked = False
            gpu_to_use = arg_min_
            ts = datetime.datetime.now().strftime('%H:%M:%S - %d/%m/%Y')
            print(f"\n## Running on gpu {arg_min_} at {ts} ##")
        else:
            print(f"waiting 5 more seconds | epoch {c} | {mem} | {(time.time() - start):.2f} sec", end='\r')
            time.sleep(5)
        c += 1

    return gpu_to_use




if __name__ == '__main__':

    mem = get_memory_usage()

    print(min(mem))
    print(mem.index(min(mem)))

    util = get_volatile_util()

    print(list(zip(mem, util)))

    monitor(10000)
