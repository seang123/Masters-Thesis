import subprocess as sp
import re
import datetime
import time

def check_mem():
    """
    Function for parsing nvidia-smi output. 

    Currenlty checks if any of the gpus have a memory load of less than LOAD, and if so returns the index of that gpu
    """
    # run nvidia-smi command and capture output
    x = sp.run(["nvidia-smi"], capture_output=True)
    data = x.stdout.decode("utf-8")

    # match mem usage string
    y = re.findall(r'(\d+MiB)', data)

    # get just the in use memory stat
    mem_usage = [y[i] for i in range(0, len(y)) if i in [0,2,4]]

    # split number from MiB
    mem = []
    for i in mem_usage:
        mem.append(re.search(r'\d+', i)[0])

    gpu_to_use = 0
    LOAD = 2000 # memory load of gpu

    start = time.time()
    c = 0 
    mem_blocked = True 
    while mem_blocked:
        for i in range(0, len(mem)):
            if int(mem[i]) < LOAD:
                mem_blocked = False
                gpu_to_use = i
                print(f"running main on gpu {i} at {datetime.datetime.now().strftime('%H:%M:%S -- %d/%m/%Y')}") 
                #gpu_var.update(f"{i} is free at {datetime.datetime.now().strftime('%H:%M:%S -- %d/%m/%Y')} after {(time.time() - start):.2f} seconds of waiting.")
            else:
                print(f"waiting 2 more seconds | epoch {c} | total time {(time.time() - start):.2f} sec", end='\r')
                time.sleep(2)

        c += 1

    return gpu_to_use
