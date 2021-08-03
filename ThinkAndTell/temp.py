import json
from parameters import parameters as param


with open(f'./data/IMG/time_dist/config.txt', 'w') as f:
    f.write(json.dumps(param)) # store model config dict
