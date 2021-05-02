import subprocess as sp
import re
import time
import datetime


def get_memory_usage():
    """ Query nvidia-smi for current gpu memory usage

    Return:
        list of memory usages values as strings
    """
    # run nvidia-smi command and capture output
    x = sp.run(["nvidia-smi"], capture_output=True)
    data = x.stdout.decode("utf-8")

    # match mem usage string
    y = re.findall(r'(\d+MiB)', data)

    # get just the in-use memory stat
    mem_usage = [y[i] for i in range(0, len(y)) if i in [0,2,4]]

    # split number from MiB
    mem = []
    for i in mem_usage:
        mem.append(re.search(r'\d+', i)[0])

    return mem


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
                print(f"\nTraining on gpu {i} at {ts}")
                break

        if mem_blocked:
            print(f"waiting 5 more seconds | epoch {c} | {mem} | {(time.time() - start):.2f}", end='\r')
            time.sleep(5)
        
        c += 1

    return gpu_to_use

if __name__ == '__main__':
    mem = get_memory_usage()
    print(mem)
