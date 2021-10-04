"""[nds_get_data]

    utilies for nsd
"""
import numpy as np
import nibabel as nb
from scipy.stats import zscore
from nsd_access import NSDAccess
import os, sys
import glob
import re


def get_masks(nsd_dir, sub, targetspace='func1pt8mm'):
    """[summary]

    Args:
        nsd_dir ([type]): [description]
        sub ([type]): [description]
        targetspace (str, optional): [description]. Defaults to 'func1pt8mm'.

    Returns:
        [type]: [description]
    """
    # initiate nsda
    nsda = NSDAccess(nsd_dir)

    brainmask = nsda.read_vol_ppdata(
        sub,
        filename='brainmask',
        data_format=targetspace)

    return brainmask


def get_1000(nsd_dir):
    """[get condition indices for the special 1000 images.]

    Arguments:
        nsd_dir {[os.path]} -- [where is the nsd data?]

    Returns:
        [lit of inds] -- [indices related to the 1000 special
                          stimuli in a coco format]
    """
    stim1000_dir = os.path.join(
        nsd_dir,
        'nsddata',
        'stimuli',
        'nsd',
        'shared1000',
        '*.png')

    stim1000 = [os.path.basename(x)[:-4] for x in glob.glob(stim1000_dir)]
    stim1000.sort()
    stim_ids = \
        [int(re.split('nsd', stim1000[x])[1]) for x, n in enumerate(stim1000)]

    stim_ids = list(np.asarray(stim_ids))
    return stim_ids


def get_100(nsd_dir):
    """[get condition indices for the special chosen 100 images.]

    Arguments:
        nsd_dir {[os.path]} -- [where is the nsd data?]

    Returns:
        [lit of inds] -- [indices related to the chosen 100 special
                          stimuli in a coco format]
    """

    stim_ids = get_1000(nsd_dir)
    # kendrick's chosen 100
    chosen_100 = [4, 8, 22, 30, 33, 52, 64, 69, 73, 137, 139,
                  140, 145, 157, 159, 163, 186, 194, 197,
                  211, 234, 267, 287, 300, 307, 310, 318, 326,
                  334, 350, 358, 362, 369, 378, 382, 404, 405,
                  425, 463, 474, 487, 488, 491, 498, 507, 520,
                  530, 535, 568, 570, 579, 588, 589, 591, 610,
                  614, 616, 623, 634, 646, 650, 689, 694, 695,
                  700, 727, 730, 733, 745, 746, 754, 764, 768,
                  786, 789, 790, 797, 811, 825, 853, 857, 869,
                  876, 882, 896, 905, 910, 925, 936, 941, 944,
                  948, 960, 962, 968, 969, 974, 986, 991, 999]
    # here we remove 1 to account for the difference
    # between matlab's 1-based indexing and python's 0-based
    # indexing.
    chosen_100 = np.asarray(chosen_100) - 1

    chosen_ids = list(np.asarray(stim_ids)[chosen_100])

    return chosen_ids


def get_conditions_515(nsd_dir, n_sessions=40):
    """[get condition indices for the special 515 images.]

    Arguments:
        nsd_dir {[os.path]} -- [where is the nsd data?]

    Returns:
        [lit of inds] -- [indices related to the special
                          515 stimuli in a coco format]
    """
    # first identify the 1000 shared images.
    stim_1000 = get_1000(nsd_dir)

    sub_conditions = []
    # loop over sessions
    for sub in range(8):
        subix = f'subj0{sub+1}'
        # extract conditions data and reshape conditions to be ntrials x 1
        conditions = np.asarray(
            get_conditions(nsd_dir, subix, n_sessions)).ravel()

        # find the 3 repeats
        conditions_bool = \
            [True if np.sum(
                conditions == x) == 3 else False for x in conditions]
        conditions = conditions[conditions_bool]

        conditions_1000 = [x for x in stim_1000 if x in conditions]
        print(f'{subix} saw {len(conditions_1000)} of the 1000')

        if sub == 0:
            sub_conditions = conditions_1000
        else:
            sub_conditions = [
                x for x in conditions_1000 if x in sub_conditions
                ]

    return sub_conditions


def get_conditions(nsd_dir, sub, n_sessions):
    """[summary]

    Args:
        nsd_dir {[os.path]} -- [where is the nsd data?]
        sub (str): subject id (e.g. subj01)
        n_sessions (int): number of sessions for which we want the data.

    Returns:
        conditions: list of conditions seen by the subject.
    """

    # initiate nsda
    nsda = NSDAccess(nsd_dir)

    # read behaviour files for current subj
    conditions = []

    # loop over sessions
    for ses in range(n_sessions):
        ses_i = ses+1
        print(f'\t\tsub: {sub} fetching condition trials in session: {ses_i}')

        # we only want to keep the shared_1000
        this_ses = np.asarray(
            nsda.read_behavior(subject=sub, session_index=ses_i)['73KID'])

        # these are the 73K ids.
        valid_trials = [j for j, x in enumerate(this_ses)]

        # this skips if say session 39 doesn't exist for subject x
        # (see n_sessions comment above)
        if valid_trials:
            conditions.append(this_ses)

    return conditions

def my_get_betas(nsd_dir, sub, n_sessions=40, zscore_data=True, out_path = "./data/", mask=None, targetspace='func1pt8mm'):
    """[summary]

    Args:
        nsd_dir (str): directory where nsd data lives
        sub (str): subject id (e.g. subj01)
        n_sessions (int): how many sessions that subject has seen. defalut = 40
        mask (bool array, optional): mask to apply to the data.
                                     Defaults to None.
        targetspace (str, optional): the data space we want to get.
                                     Defaults to 'func1pt8mm'.

    Returns:
        betas (array): the betas for the subject,
    """


    # initiate nsda
    nsda = NSDAccess(nsd_dir)

    data_folder = os.path.join(
        nsda.nsddata_betas_folder,
        sub,
        targetspace,
        'betas_fithrf_GLMdenoise_RR')

    betas = []
    # loop over sessions
    # trial_index=0
    for ses in range(n_sessions):
        ses_i = ses+1
        si_str = str(ses_i).zfill(2)

        # sess_slice = slice(trial_index, trial_index+750)
        print(f'\t\tsub: {sub} fetching betas for trials in session: {ses_i}')

        # we only want to keep the shared_1000
        this_ses = nsda.read_behavior(subject=sub, session_index=ses_i)

        # these are the 73K ids.
        ses_conditions = np.asarray(this_ses['73KID'])

        valid_trials = [j for j, x in enumerate(ses_conditions)]

        # this skips if say session 39 doesn't exist for subject x
        # (see n_sessions comment above)
        if valid_trials:

            if targetspace == 'fsaverage':
                conaxis = 1

                # load lh
                img_lh = nb.load(
                        os.path.join(
                            data_folder,
                            f'lh.betas_session{si_str}.mgh' # was .mgz
                            )
                        ).get_data().squeeze()

                # load rh
                img_rh = nb.load(
                        os.path.join(
                            data_folder,
                            f'rh.betas_session{si_str}.mgh'
                            )
                        ).get_data().squeeze()

                # concatenate
                all_verts = np.vstack((img_lh, img_rh))

                if zscore_data:
                    betas = zscore(all_verts, axis=conaxis).astype(np.float32)
                else:
                    betas = np.array(all_verts, dtype=np.float32)


                for beta, (row_idx, row) in zip(betas.T, this_ses.iterrows()):

                    file_name = f"betas_SUB{int(row['SUBJECT'])}_S{int(row['SESSION'])}_R{int(row['RUN'])}_T{int(row['TRIAL'])}_KID{int(row['73KID'])}.npy"
                    file_path = os.path.join(out_path, f"subj_{int(row['SUBJECT'])}", "betas/")
                    file_name = os.path.join(file_path, file_name)

                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                    # TODO: save beta.npy
                    with open(file_name, "wb") as f:
                        np.save(f, beta)

                    ## Save captions
                    captions = [i["caption"] for i in nsda.read_image_coco_info([int(row['73KID'])])]

                    cap_text = ""
                    for cap_idx, cap in enumerate(captions[:5]):
                        cap = cap.replace("\n", "")
                        cap_text += f"{file_name}#{cap_idx}\t{cap}\n"

                    # Write captions to file - 1 file with 5 captions per beta 
                    cur_cap_file = f"SUB{int(row['SUBJECT'])}_KID{int(row['73KID'])}.txt"
                    cur_cap_path = os.path.join(out_path, f"subj_{int(row['SUBJECT'])}", "captions")

                    if not os.path.exists(cur_cap_path):
                        os.makedirs(cur_cap_path)

                    with open(os.path.join(cur_cap_path, cur_cap_file), "w") as f:
                        f.write(cap_text)
                    

    return betas, ses_conditions

def get_betas_mem_eff(nsd_dir, sub, n_sessions, mask=None, targetspace='func1pt8mm'):
    """ Same as get_betas, but more memory efficient

    Args:
        nsd_dir (str): directory where nsd data lives
        sub (str): subject id (e.g. subj01)
        n_sessions (int): how many sessions that subject has seen. defalut = 40
        mask (bool array, optional): mask to apply to the data.
                                     Defaults to None.
        targetspace (str, optional): the data space we want to get.
                                     Defaults to 'func1pt8mm'.

    Returns:
        betas (array): the betas for the subject,
    """

    # initiate nsda
    nsda = NSDAccess(nsd_dir)

    data_folder = os.path.join(
        nsda.nsddata_betas_folder,
        sub,
        targetspace,
        'betas_fithrf_GLMdenoise_RR')

    betas = []
    betas = np.zeros((327684, 750 * n_sessions), dtype=np.float32)
    # loop over sessions
    # trial_index=0
    for ses in range(n_sessions):
        ses_i = ses+1
        si_str = str(ses_i).zfill(2)

        # sess_slice = slice(trial_index, trial_index+750)
        print(f'\t\tsub: {sub} fetching betas for trials in session: {ses_i}')

        # we only want to keep the shared_1000
        this_ses = nsda.read_behavior(subject=sub, session_index=ses_i)

        # these are the 73K ids.
        ses_conditions = np.asarray(this_ses['73KID'])

        valid_trials = [j for j, x in enumerate(ses_conditions)]

        # this skips if say session 39 doesn't exist for subject x
        # (see n_sessions comment above)
        if valid_trials:

            if targetspace == 'fsaverage':
                conaxis = 1

                # load lh
                img_lh = nb.load(
                        os.path.join(
                            data_folder,
                            f'lh.betas_session{si_str}.mgh' # was .mgz
                            )
                        ).get_data().squeeze()

                # load rh
                img_rh = nb.load(
                        os.path.join(
                            data_folder,
                            f'rh.betas_session{si_str}.mgh'
                            )
                        ).get_data().squeeze()

                # concatenate
                all_verts = np.vstack((img_lh, img_rh))

                # mask
                if mask is not None:
                    tmp = zscore(all_verts, axis=conaxis).astype(np.float32)

                    # you may want to get several ROIs from a list of
                    # ROIs at once
                    if type(mask) == list:
                        masked_betas = []
                        for mask_is in mask:
                            tmp2 = tmp[mask_is, :]
                            # check for nans
                            # good = np.any(np.isfinite(tmp2), axis=1)
                            masked_betas.append(tmp2)
                    else:
                        tmp2 = tmp[mask_is, :]
                        # good = np.any(np.isfinite(tmp2), axis=1)
                        masked_betas = tmp2

                    betas.append(masked_betas)
                else:
                    x = (zscore(
                            all_verts,
                            axis=conaxis)).astype(np.float32)
                    betas[:,750*ses:750+750*ses] = x
                    #betas.append(
                    #    (zscore(
                    #        all_verts,
                    #        axis=conaxis)).astype(np.float32)
                    #    )
            else:
                conaxis = 1
                img = nb.load(
                    os.path.join(data_folder, f'betas_session{si_str}.nii.gz'))

                if mask is not None:
                    betas.append(
                        (zscore(
                            np.asarray(
                                img.dataobj),
                            axis=conaxis)[mask, :]*300).astype(np.int16)
                        )
                else:
                    betas.append(
                        (zscore(
                            np.asarray(
                                img.dataobj),
                            axis=conaxis)*300).astype(np.int16)
                        )



    return betas


def get_betas(nsd_dir, sub, n_sessions, mask=None, targetspace='func1pt8mm'):
    """[summary]

    Args:
        nsd_dir (str): directory where nsd data lives
        sub (str): subject id (e.g. subj01)
        n_sessions (int): how many sessions that subject has seen. defalut = 40
        mask (bool array, optional): mask to apply to the data.
                                     Defaults to None.
        targetspace (str, optional): the data space we want to get.
                                     Defaults to 'func1pt8mm'.

    Returns:
        betas (array): the betas for the subject,
    """

    # initiate nsda
    nsda = NSDAccess(nsd_dir)

    data_folder = os.path.join(
        nsda.nsddata_betas_folder,
        sub,
        targetspace,
        'betas_fithrf_GLMdenoise_RR')

    betas = []
    # loop over sessions
    # trial_index=0
    for ses in range(n_sessions):
        ses_i = ses+1
        si_str = str(ses_i).zfill(2)

        # sess_slice = slice(trial_index, trial_index+750)
        print(f'\t\tsub: {sub} fetching betas for trials in session: {ses_i}')

        # we only want to keep the shared_1000
        this_ses = nsda.read_behavior(subject=sub, session_index=ses_i)

        # these are the 73K ids.
        ses_conditions = np.asarray(this_ses['73KID'])

        valid_trials = [j for j, x in enumerate(ses_conditions)]

        # this skips if say session 39 doesn't exist for subject x
        # (see n_sessions comment above)
        if valid_trials:

            if targetspace == 'fsaverage':
                conaxis = 1

                # load lh
                img_lh = nb.load(
                        os.path.join(
                            data_folder,
                            f'lh.betas_session{si_str}.mgh' # was .mgz
                            )
                        ).get_data().squeeze()

                # load rh
                img_rh = nb.load(
                        os.path.join(
                            data_folder,
                            f'rh.betas_session{si_str}.mgh'
                            )
                        ).get_data().squeeze()

                # concatenate
                all_verts = np.vstack((img_lh, img_rh))

                # mask
                if mask is not None:
                    tmp = zscore(all_verts, axis=conaxis).astype(np.float32)

                    # you may want to get several ROIs from a list of
                    # ROIs at once
                    if type(mask) == list:
                        masked_betas = []
                        for mask_is in mask:
                            tmp2 = tmp[mask_is, :]
                            # check for nans
                            # good = np.any(np.isfinite(tmp2), axis=1)
                            masked_betas.append(tmp2)
                    else:
                        tmp2 = tmp[mask_is, :]
                        # good = np.any(np.isfinite(tmp2), axis=1)
                        masked_betas = tmp2

                    betas.append(masked_betas)
                else:
                    betas.append(
                        (zscore(
                            all_verts,
                            axis=conaxis)).astype(np.float32)
                        )
            else:
                conaxis = 1
                img = nb.load(
                    os.path.join(data_folder, f'betas_session{si_str}.nii.gz'))

                if mask is not None:
                    betas.append(
                        (zscore(
                            np.asarray(
                                img.dataobj),
                            axis=conaxis)[mask, :]*300).astype(np.int16)
                        )
                else:
                    betas.append(
                        (zscore(
                            np.asarray(
                                img.dataobj),
                            axis=conaxis)*300).astype(np.int16)
                        )



    return betas


def average_over_conditions(data, conditions, conditions_to_avg):
    """[summary]

    Args:
        data (array): betas
        conditions (list): list of conditions
        conditions_to_avg (list): list of the desired conditions
                                  to average

    Returns:
        avg_data (array): averaged betas over repetitions
                          of the conditions specified in
                          conditions_to_avg
    """
    lookup = np.unique(conditions_to_avg)
    n_conds = lookup.shape[0]
    n_dims = data.ndim

    if n_dims == 2:
        n_voxels, _ = data.shape
        avg_data = np.empty((n_voxels, n_conds))
    else:
        x, y, z, _ = data.shape
        avg_data = np.empty(
            (x, y, z, n_conds)
            )

    for j, x in enumerate(lookup):

        conditions_bool = conditions == x
        if n_dims == 2:
            if np.sum(conditions_bool) == 0:
                break
            # print((j, np.sum(conditions_bool)))
            sliced = data[:, conditions_bool]

            avg_data[:, j] = np.nanmean(sliced, axis=1)
        else:
            avg_data[:, :, :, j] = np.nanmean(
                data[:, :, :, conditions_bool],
                axis=3
                )

    return avg_data
