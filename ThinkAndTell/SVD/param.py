'''
configuration for training NSD_autoencoder
'''


config = {"BATCHSIZE": 20,
            "EPOCHS": 400,
            "N_PARALLEL_DATASETS": 50,  # number of interleaved datasets
            "NSD_HOME": '/home/danant/NSD',
            "SUBJECTS": ['subj02'],
            "HDFLOC": '/home/danant/NSD/nsddata_betas/ppdata/subj02/fsaverage/betas_fithrf_GLMdenoise_RR/subj02.hdf5',
            "DATASET": 'subj02',
            "GLASSER_LH": '/home/danant/misc/lh.HCP_MMP1.mgz',
            "GLASSER_RH": '/home/danant/misc/rh.HCP_MMP1.mgz',
            "VISUAL_MASK": '/home/danant/misc/visual_parcels_glasser.csv',
            "FREESURFER": '/home/danant/NSD/nsddata/',
            "BETA_FOLDER": '/home/danant/NSD/nsddata_betas/ppdata/subj02/fsaverage/betas_fithrf_GLMdenoise_RR/',
            "PLOT_MIN": -24,
            "PLOT_MAX": 28,
            "DATASETS": "/fast/danant",
            "NSD_STAT": "/huge/danant/NSD_Statistics"
        }
            
