import os
import os.path as op
import time
import numpy as np
from nsd_access import NSDAccess
from nsd_get_data import (get_betas,
                          get_conditions, get_conditions_515,
                          average_over_conditions)

total_time = time.time()
subnumber = 0

n_subjects = 8
n_jobs = 18

# ridge regression parameters
n_alphas = 20
fracs = np.linspace(0.001, 0.999, n_alphas).astype(np.float32)

# set up directories
base_dir = os.path.join('/rds', 'projects', 'c')
nsd_dir = os.path.join(base_dir, 'charesti-start', 'data', 'NSD')
proj_dir = os.path.join(base_dir, 'charesti-start', 'projects', 'NSD')
sem_dir = os.path.join(proj_dir, 'derivatives', 'semantics')

# where do we save the concatenated betas?
betas_dir = op.join(proj_dir, 'betas')
if not os.path.exists(betas_dir):
    os.makedirs(betas_dir)

nsda = NSDAccess(nsd_dir)

# we use the fsaverage space. [see get_betas()]
targetspace = 'fsaverage'

# sessions
n_sessions = 40

# subjects
subs = ['subj0{}'.format(x+1) for x in range(n_subjects)]
sub = subs[subnumber]

# get the condition list for the special 515
# these will be used as testing set for the guse predictions
conditions_515 = get_conditions_515(nsd_dir)

# extract conditions data
conditions = get_conditions(nsd_dir, sub, n_sessions)

# we also need to reshape conditions to be ntrials x 1
conditions = np.asarray(conditions).ravel()

# then we find the valid trials for which we do have 3 repetitions.
conditions_bool = [
    True if np.sum(conditions == x) == 3 else False for x in conditions]

# and identify those.
conditions_sampled = conditions[conditions_bool]

# find the subject's condition list (sample pool)
sample = np.unique(conditions[conditions_bool])
n_images = len(sample)
all_conditions = range(n_images)

# also identify which image in the sample is a conditions_515
sample_515_bool = [True if x in conditions_515 else False for x in sample]

# let's compile the betas or load the pre-compiled file
betas_mean_file = os.path.join(
        betas_dir, f'{sub}_betas_{targetspace}_averaged.npy'
)

if not os.path.exists(betas_mean_file):
    # get betas
    betas_mean = get_betas(
        nsd_dir,
        sub,
        n_sessions,
        targetspace=targetspace,
    )
    print(f'concatenating betas for {sub}')
    betas_mean = np.concatenate(betas_mean, axis=1).astype(np.float32)

    print(f'averaging betas for {sub}')
    betas_mean = average_over_conditions(
        betas_mean,
        conditions,
        conditions_sampled,
    ).astype(np.float32)

    # print
    print(f'saving condition averaged betas for {sub}')
    np.save(betas_mean_file, betas_mean)

else:
    print(f'loading betas for {sub}')
    betas_mean = np.load(betas_mean_file, allow_pickle=True)

good_vertex = [
    True if np.sum(np.isnan(x)) == 0 else False for x in betas_mean]

if np.sum(good_vertex) != len(good_vertex):
    print(f'found some NaN for {sub}')

betas_mean = betas_mean[good_vertex, :]

# have fun !
