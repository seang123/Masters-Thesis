import nsd_get_data as ngd
import sys
import pandas as pd

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




betas, nsd_keys = ngd.my_get_betas("/home/seagie/NSD2/", "subj02", 40, out_path = "/huge/seagie/data/", targetspace="fsaverage")

print("nsd_keys:", nsd_keys.shape)

print("Betas shape:", len(betas), betas[0].shape) # 40 x (327684, 750)
