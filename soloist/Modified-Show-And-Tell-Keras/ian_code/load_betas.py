import nsd_get_data as ngd
import sys
import pandas as pd
import numpy as np
import os, sys

##
# Generate a direcotry containing betas and captions
#
#

response = pd.read_csv("/home/seagie/NSD2/nsddata/ppdata/subj02/behav/responses.tsv", sep='\t', header=0)

stim_info_merged = pd.read_csv("/home/seagie/NSD2/nsddata/experiments/nsd/nsd_stim_info_merged.csv")

#print(response.columns)
#print(response.shape)
#print(response.head(1))

#print(stim_info_merged.columns)
#print(stim_info_merged.shape)
#print(stim_info_merged.head(1))

#print(stim_info_merged.shared1000.value_counts())
#print(stim_info_merged.subject2.value_counts())


data_dir = "/huge/seagie/data_meaned/"
nsd_dir = "/home/seagie/NSD2/"
subject = "subj02"
n_sessions = 40
targetspace = 'fsaverage'

conditions = ngd.get_conditions(nsd_dir, subject, n_sessions)
print("conditions:", len(conditions), conditions[0].shape)
conditions = np.asarray(conditions).ravel()
print("conditions.ravel:", conditions.shape)



conditions_bool = [True if np.sum(conditions == x) == 3 else False for x in conditions]
conditions_sampled = conditions[conditions_bool]

sample = np.unique(conditions[conditions_bool])
n_images = len(sample)
all_conditions = range(n_images)

print("conditions_bool", len(conditions_bool))
print("conditions_sampled", len(conditions_sampled))
print("sample", sample.shape)

betas_mean_file_conditions = os.path.join(data_dir, f"{subject}_betas_{targetspace}_averaged_unq_sample.txt")
with open(betas_mean_file_conditions, "w") as f:
    for i in sample:
        f.write(f"{i}\n")

raise Exception()

betas_mean_file = os.path.join(data_dir, 
        f"{subject}_betas_{targetspace}_averaged.npy")

if not os.path.exists(betas_mean_file):
    betas_mean = ngd.get_betas_mem_eff(nsd_dir, 
            subject, 
            n_sessions, 
            targetspace=targetspace
    )
    print(f"Betas mean: {betas_mean.shape}")
    print(f"concatenating betas for {subject}")
    #betas_mean = np.concatenate(betas_mean, axis=1).astype(np.float32)
    print(f"betas_mean concat: {betas_mean.shape}")

    print(f"averaging betas for {subject}")
    betas_mean = ngd.average_over_conditions(
            betas_mean,
            conditions,
            conditions_sampled,
    ).astype(np.float32)
    print("betas_mean avg: {betas_mean.shape}")

    print(f"saving condition averaged betas for {subject}")
    np.save(betas_mean_file, betas_mean)

good_vertex = [
        True if np.sum(np.isnan(x)) == 0 else False for x in betas_mean]

if np.sum(good_vertex) != len(good_vertex):
    print(f"Found NaN values in betas_mean for {subject}")

#betas_mean = betas_mean[good_vertex,:]





sys.exit(0)
#### OLD method - didnt properly average data
betas, nsd_keys = ngd.my_get_betas("/home/seagie/NSD2/", "subj02", 40, out_path = "/huge/seagie/data/", targetspace="fsaverage")

print("nsd_keys:", nsd_keys.shape)

print("Betas shape:", len(betas), betas[0].shape) # 40 x (327684, 750)
