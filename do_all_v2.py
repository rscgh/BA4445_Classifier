#!/usr/local/bin/python
# -*- coding: utf-8 -*-


'''
based off an collection of scripts found on 
Original Author(s):
Estrid Jakobsen, https://github.com/Estrid
Daniel Margulies, https://github.com/margulies

@2020

'''


import nibabel as nib
import numpy as np
#from scipy.io import savemat
from nilearn.decomposition.canica import CanICA
#from hcp_corr import t_series # seyma can help set this up
import os
import hdf5storage

#from helpers.load_hcp_annotated_rsc import t_series 
import hcp_tools
from partial_corr import partial_corr


KEEP_FIRST_SUBJ_INTERM_DATA = True;


#### code was updated and corrected; to be executed with python2.7 on the MPI CBS infrastructure
#### should try to port it soon to python3 as nilearn announces renouncing py2.7

hcp_all_path = '/data/t_hcp/S500_2014-06-25/_all'
out_folder_p = '/data/pt_02189/MARHCP/BrocaConn'

#subj_list_path = "/scr/murg2/HCP_Q3_glyphsets_left-only/subject_list_101.txt"
subj_list_path = '/data/pt_02189/MARHCP/BrocaConn/subject_list_101.txt'

# LEFT indices; normally should do this separately for each individual; as all have the same sequence and also the manual group parcellations
# are based on the same mapping though, this practically doesnt make any difference; (akin to the broca area mask i guess which i calculate individually though)
# img = nib.load('/a/documents/connectome/_all/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii')
img = nib.load(hcp_all_path + '/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii')
#-> caution: pixdim[1,2,3] should be non-zero; setting 0 dims to 1

# cort = img.header.matrix.mims[1].brainModels[0].vertexIndices.indices
# doesnt exist
# bms = list(img.header.matrix._mims[1].brain_models)
# cort = bms[0].vertex_indices._indices                     # cort is list with 29696 elements      with min = 0, max = 32491, mean = 16711, sum = 496251915; but not just [1,2,3,4,5 ...]
cort = list(img.header.matrix._mims[1].brain_models)[0].vertex_indices._indices

np.savetxt(out_folder_p + '/indices_LH_29696_of_32492.txt', cort)


###############################################################################
## Extract combined time series
###############################################################################

subs = np.loadtxt(subj_list_path, dtype=str)  # should contain individual rows of strings like '100307'

print("Process Subjects (n=", len(subs), "): ", subs)

#subs = np.loadtxt("/scr/murg1/HCP500_glyphsets/subject_list_HCP500.txt", dtype=str)
filenames = []
new_subs = [];

for sub in subs:
    # sub = '100307' 103414

    keep_all_files = KEEP_FIRST_SUBJ_INTERM_DATA and (sub == '103414');

    print "SUBJECT ", sub
    filename = os.path.join(out_folder_p ,'indv', '%s_cifti_tseries_shape_sV-1-1-T.nii.gz' % sub);

    ### if the file already exists, skip the anewed extraction of the time series:
    if os.path.exists(filename):
        print("Time series has already been extracted before as the file already exists: ")
        print(filename);
        print("Skip ahead to the extraction for the next subject ...")#
        filenames.append(filename)
        new_subs.append(sub)
        continue;


    #data = t_series(subject = hcp_all_path + "/%s" % sub, hemisphere='LH', N_first=0, N_cnt=32492)
    #dataxx = t_series(subject = hcp_all_path + "/%s" % sub, hemisphere='LH', N_first=0, N_cnt=32492, normalize=False)
    data = hcp_tools.preprocess_and_load_tseries(hcp_all_path + "/%s" % sub, sub, N_first = 0, N_cnt = 32492, smoothing = True, normalize=True, keep_tmp_files= keep_all_files, temp_dir=os.path.join(out_folder_p, "indv")) # returns (4800, 32492)

    # saving as real cifti for visualization
    if keep_all_files: 
     cimg = nib.Cifti2Image(data.T[:, :29696], nih_full)
     cimg.to_filename(os.path.join(out_folder_p, "indv",'rsfmri_103414_1-4.LH.smooth.dtseries.nii'));


    if len(data) == 0:
        print("Skip subject as data couldnt be found: ", sub)
        continue;

    new_subs.append(sub)

    #datash = data[cort, :]                                               # (29696, 4800),    same length as cort now; cort is not correct!
    datash = data[:29696, :]                                              # (29696, 4800),    same length as cort now
    datars = np.reshape(datash, (datash.shape[0],1,1,datash.shape[1]))    # (29696, 1, 1, 4800), nessesary for nifti and CanICA
        
    img = nib.Nifti1Image(datars, np.eye(4))
    img.to_filename(filename)
    filenames.append(filename)

    ''' producing odd pictures when using cort:
    data_striped = data[cort, :]
    cimg_odd = nib.Cifti2Image(data_striped.T, nih_full)
    cimg_odd.to_filename(os.path.join(out_folder_p, "sample",'rsfmri_103414_1-4.LH.odd_stripes_dueto_cort.dtseries.nii'));
    

    cimg_odd_unnorm = nib.Cifti2Image(dataxx[cort, :].T, nih_full)
    cimg_odd_unnorm.to_filename('rsfmri_103414_1-4.LH.odd_stripes_dueto_cort.unnorm.dtseries.nii');

    '''


subs = new_subs;

###############################################################################
## Group ICA
###############################################################################


group_ica_fn = out_folder_p + '/ica_HCP_all' + str(len(filenames)) +'_LH29k_20comps.mat'

if os.path.exists(group_ica_fn):
 print("Group ICA components have already been extracted: ")
 print(group_ica_fn);
 print("Skip ahead to the extraction for the next subject ...")
else:
 # create artificial mask, again saved in the wrong format but nessesary for CanICA
 mask = np.ones((29696,1,1));
 img = nib.Nifti1Image(mask, np.eye(4))      # -> (29696, 1, 1)
 img.to_filename(out_folder_p + '/mask.nii.gz')
 
 # run ICA on group level:
 canica = CanICA(mask=os.path.join(out_folder_p,'mask.nii.gz'), n_components=20, smoothing_fwhm=0., threshold=None, verbose=10, random_state=0, n_jobs=20)
 canica.fit(filenames)   # -> canica.components_ of shape (20, 29696)

 # Retrieve the independent components in brain space
 components_img = canica.masker_.inverse_transform(canica.components_)   # -> components_img.shape == (29696, 1, 1, 20)

 group_ica = components_img.get_data().squeeze() # components_img.get_data().squeeze() is of shape (29696, 20) != canica.components_

 #np.save('/scr/murg2/MachineLearning/partialcorr/ICA/ICA_HCP/ica_HCP101_output_%s.npy' % str(n_components), A)
 print("Saving Group ICA: ", group_ica_fn)
 hdf5storage.savemat(group_ica_fn, {'ic':group_ica})


'''
# alternate way of hopefully correctly saving a cifti file ...
bm_left = nib.cifti2.BrainModelAxis.from_mask(np.ones(32492), name="cortex_left")
nih = nib.cifti2.Cifti2Header.from_axes((bm_left,nib.cifti2.ScalarAxis(['comp' + str(i) for i in range(20)])))
cimg_ica_comp = nib.Cifti2Image(A, nih)
cimg_ica_comp.to_filename(out_folder_p + '/ica_HCP' + str(len(filenames)) +'_20comps_output.dscalar.nii');
'''


###############################################################################
## Individual ICA
## - Input: individual timeseries stored as nii.gz of shape (29696, 1, 1, 4800)
##   /data/pt_02189/MARHCP/BrocaConn/indv/100307_cifti_tseries_shape_sV-1-1-T.nii.gz'
## - Output: 20 individal ICA components of shape (32492, 20)
##   /data/pt_02189/MARHCP/BrocaConn/indv/100307_ica_HCP_20comps_output.mat'
###############################################################################

print("Do indv ica: ", subs, filenames)


## RUN ICA ON INDIVIDUAL LEVEL:
for (sub, filename) in zip(subs, filenames):
    
    ind_ica_fn =os.path.join(out_folder_p,"indv", "ica_HCP_indv_" + sub+"_LH29k_20comps.mat")
    
    if os.path.exists(ind_ica_fn):
        print("Ica already has been extracted for the subject and is stored in: ")
        print(ind_ica_fn);
        print("Skip ahead to the extraction for the next subject ...")
        continue;

    canica_ind = CanICA(mask=os.path.join(out_folder_p,'mask.nii.gz'), n_components=20, smoothing_fwhm=0., threshold=None, verbose=10, random_state=0, n_jobs=20)


    canica_ind.fit(filename)
        
    # Retrieve the independent components in brain space
    components_img_ind = canica_ind.masker_.inverse_transform(canica_ind.components_)
        
    Aind = components_img_ind.get_data().squeeze() # (29696,n_components)
    print("  Saving Indv ICA: ", ind_ica_fn)
    hdf5storage.savemat(ind_ica_fn, {'ic':Aind})

    #nib.Cifti2Image(Aind, nih).to_filename(out_folder_p + '/individual/' + sub + 'ica_HCP_20comps_output.dscalar.nii');
        
        
###############################################################################
## Create Group Correlation Matrix
##
## - Input: individual timeseries stored as nii.gz of shape (29696, 1, 1, 4800)
##   /data/pt_02189/MARHCP/BrocaConn/indv/100307_cifti_tseries_shape_sV-1-1-T.nii.gz'
##    * indices that were used to recude the left hemisphere from 32492 svox to 29696
## - Output: 
##    * binary brocas mask of size (32492,) 
##        saved in /data/pt_02189/MARHCP/BrocaConn/indv/BROCAMSK_HCP_indv_brocas_mask_1396_of_32492_bin.txt
##     * (optional) Individual correlation matrices of shape: (29696, 1396)
##        saved as /data/pt_02189/MARHCP/BrocaConn/indv/100307_corrmat_broca_29696x 1396_nosmt.mat
##     * Group level correlation matrix of shape (29696, 1396)
##        saved as /data/pt_02189/MARHCP/BrocaConn/HCP101_corrmat_broca_29696x 1396_nosmt.mat
## 
###############################################################################


#cort2 = np.readtxt(out_folder_p + '/indices_LH_29696_of_32492.txt', cort).astype(np.int).tolist()
cort2 = np.loadtxt(out_folder_p + '/indices_LH_29696_of_32492.txt').astype(np.int).tolist()


def get_individual_IFG(sub, hcp_all_path = '/data/t_hcp/S500_2014-06-25/_all'):
    anatlabp = os.path.join(hcp_all_path, sub, 'MNINonLinear/fsaverage_LR32k/%s.L.aparc.32k_fs_LR.label.gii' % (sub))
    AnatLabels = nib.load(anatlabp) #AnatLabels2 = nib.gifti.giftiio.read(anatlabp)
    #AnatLabels.print_summary()
    #AnatLabels.get_labeltable().labels[20].label #-> u'L_parstriangularis'
    #AnatLabels.darrays[0].data.shape #-> (32492,); elements: array([10, 29, 24, ..., 15, 15, 15], dtype=int32)
    AnatLabelsData= AnatLabels.darrays[0].data
    op = AnatLabelsData == 18;                          # shape: (32492,)
    tri = AnatLabelsData == 20;
    #np.count_nonzero((op+tri)) # -> 989 voxels in the combined region
    return [op, tri];


# get manual probability maps
prob44p = '/data/hu_robertscholz/Desktop/owncloud-gwdg/code/neuralgradients/BrocaLabels/HCP/manual_labels/ProbabilityMap_BA44.1D'
prob44b = np.loadtxt(prob44p)           # -> .shape == (32492,), np.count_nonzero(prob44) -> 738
prob44b[prob44b>0] = True
prob45p = '/data/hu_robertscholz/Desktop/owncloud-gwdg/code/neuralgradients/BrocaLabels/HCP/manual_labels/ProbabilityMap_BA45.1D'
prob45b = np.loadtxt(prob45p)           # -> .shape == (32492,), np.count_nonzero(prob44) -> 869
prob45b[prob45b>0] = True

# create the full broca mask 
#broca = prob44b + prob45b + op + tri;       # has 1396 non-zero svoxels (range 1-3, so sum is 2596)
# np.savetxt(out_folder_p + '/brocas_mask_'+ str(np.count_nonzero(broca)) + '_of_' + str(broca.shape[0]) +'_bin.txt', broca)


#indv_conn_tmpls_44_prelim = np.zeros((len(sub), len(cont2)))
#indv_conn_tmpls_45_prelim= np.zeros((len(sub), len(cont2)))

indv_conn_tmpls_44_prelim = np.zeros((len(subs), 29696))
indv_conn_tmpls_45_prelim= np.zeros((len(subs), 29696))
print(indv_conn_tmpls_44_prelim.shape)



print(list(range(len(subs))) ,subs[:3], filenames[:3])

for (n, sub, filename) in zip(list(range(len(subs))) ,subs, filenames):

    print(n, "th connectivity matrix. its for subject: ", sub)
    icm_fn = out_folder_p + '/indv/CON_HCP_indv_' + sub + '_broca_LH29k_con.mat'; 
    ict_fn = out_folder_p + '/indv/CON_HCP_indv_' + sub + '_broca_LH29k_con_tmpls_44_45.prelim.mat'; 
    if os.path.exists(icm_fn) and os.path.exists(ict_fn):
     print("connectivity mat has already has been computed for the subject and is stored in: ")
     print(icm_fn);
     print("Load and skip ahead to the extraction for the next subject ...")
     ind_prelim_conn_tmpls = hdf5storage.loadmat(ict_fn) 
     # some problem with hdf5 storage reading in mat files with multiple matrices yieling unicode keys for dict
     [ct44k, ct45k] = [x for x in ind_prelim_conn_tmpls.keys()]
     print([ct44k, ct45k])
     indv_conn_tmpls_44_prelim[n,:] = ind_prelim_conn_tmpls[ct44k]
     indv_conn_tmpls_45_prelim[n,:] = ind_prelim_conn_tmpls[ct45k]
     continue;


    ts_filename = '/data/pt_02189/MARHCP/BrocaConn/' + sub +'_cifti_tseries_shape_sV-1-1-T.nii.gz'
    time_seriesf = nib.load(filename)    #(29696, 1, 1, 4800)
    time_series = time_seriesf.get_fdata().squeeze();  #(29696, 4800)

    # matlab: corrcoef(A) where Dim(A) = (5 observations/time points, 3 voxels to correlate with each other)  ~ rows represent observations.
    # numpy: Each row of x represents a variable, and each column a single observation of all those variables. Also see rowvar below. ~ columns represent observations
    cmat = np.corrcoef(time_series) # shape (29696, 29696) each voxel with each other
    # cmat[1000,2000] == cmat[2000,1000]

    [op,tr] = get_individual_IFG(sub, hcp_all_path = hcp_all_path);
    broca = prob44b + prob45b + op+tr;
    # reduce the broca mask to the voxels present in the corrmatrix
    broca_reduced = broca[cort].astype(np.bool) # from shape 32492 to 29696; contains 1396 x True
    bfn = os.path.join(out_folder_p,'indv', 'BROCAMSK_HCP_indv_%s_%i_of_%i_bin.txt' % (sub, np.count_nonzero(broca_reduced) , len(cort)))
    np.savetxt(bfn, broca_reduced)

    # make the small corrmat
    cmat_small = cmat[:,broca_reduced];    #  shape: (29696, 1396); subsection from  29696 x 29696 matrix

    ict44_prelim = cmat[:,(prob44b + op)[cort].astype(np.bool)].mean(axis=1) #shape:29696
    ict45_prelim = cmat[:,(prob45b + tr)[cort].astype(np.bool)].mean(axis=1)

    indv_conn_tmpls_44_prelim[n,:] = ict44_prelim # shape (29696)
    indv_conn_tmpls_45_prelim[n,:] = ict45_prelim # shape (29696)

    ## those are aparently not yet good enough, and hence we should not save them
    ## but rather only use them to compute the group level connectivity
    ## and only later set individual level connectivity maps again
    ## when regressing out other factors of the resting state, as proxied by the ICA

    print("Saving Indv connectivity template: ", ict_fn)
    hdf5storage.savemat(ict_fn, {'ct44':ict44_prelim.astype(np.float32), 'ct55': ict45_prelim.astype(np.float32)}) 
    print("Saving Indv redcued connectivity matrix: ", icm_fn)
    hdf5storage.savemat(icm_fn, {'icm': cmat_small.astype(np.float32)})



# fishers r-to-z tansfrom
ict44pz = np.arctanh(indv_conn_tmpls_44_prelim)  # shape (20, 29696)
ict45pz = np.arctanh(indv_conn_tmpls_45_prelim)

# average and reverse fisher transform
grp_conn_tmpl_44 = np.tanh(ict44pz.mean(axis=0)) #shape (, 29696)
grp_conn_tmpl_45 = np.tanh(ict45pz.mean(axis=0))

grconntmplfn = out_folder_p + '/CON_HCP_all' + str(len(filenames)) +'_broca_LH29k_con_tmpls_44_45.mat'
hdf5storage.savemat(grconntmplfn, {'ct44':grp_conn_tmpl_44, 'ct45': grp_conn_tmpl_45 })


brainmodel = hcp_tools.get_LH_29k_brainmodel()
dscfn = grconntmplfn.rstrip('.mat') + '.dscalar.nii'
hcp_tools.save_dscalar(dscfn, np.vstack((grp_conn_tmpl_44, grp_conn_tmpl_45)), brainmodel = brainmodel, scalar_names = ['grp_conn_tmpl_44', 'grp_conn_tmpl_45'])





###############################################################################
## Classify Ba44/45 in subjects now ....
###############################################################################

print("\n\n\n\nFinally Start with the classification: ")


#  variables for classifier 
corrthresh = 0.4;

from partial_corr import pengouin_part_corr_copy
import glob

'''
import hcp_tools, glob, hdf5storage, os, numpy as np, nibabel as nib
grconntmplfn = glob.glob(os.path.join(out_folder_p, 'CON_HCP_all*_broca_LH29k_con_tmpls_44_45.mat'))[0]
grp_conn_tmpl_44 =  hdf5storage.loadmat(grconntmplfn)['ct44']
grp_conn_tmpl_45 =  hdf5storage.loadmat(grconntmplfn)['ct45']

# also needs icm_fn = out_folder_p + '/indv/CON_HCP_indv_' + sub + '_broca_LH29k_con.mat'; 


'''
print("Load group ica info: ",group_ica_fn)
group_ica_fn = glob.glob(os.path.join(out_folder_p, 'ica_HCP_all*_LH29k_20comps.mat'))[0]
group_ica = hdf5storage.loadmat(group_ica_fn)['ic']


### run spatial correlation between group_connectivity_templates/maps and group_level_ICA_components to find which ICA component corresponds to either areas connecctivity pattern
### the ica components that do not correspond to areals connectivity as then later used as confound regressors

ica_gcon44_corr = np.zeros(20)
ica_gcon45_corr = np.zeros(20)

#https://stackoverflow.com/questions/29481518/python-equivalent-of-matlab-corr2
#np.corrcoef(grp_conn_tmpl_44, group_ica[:, 0])[0,1] == corr2(grp_conn_tmpl_44, group_ica[:, 0])

#group_ica of shape (32492,20)
for icn  in range(group_ica.shape[1]):
  #ica_gcon44_corr[icn] = corr2(grp_conn_tmpl_44, group_ica[cort, icn])
  ica_gcon44_corr[icn] = np.corrcoef(grp_conn_tmpl_44, group_ica[:, icn])[0,1]
  ica_gcon45_corr[icn] = np.corrcoef(grp_conn_tmpl_45, group_ica[:, icn])[0,1]

#ica_gcon44_corr.max() ~ 0.45; #ica_gcon45_corr.max() ~ 0.66


### find the voxels in the indivial connectivity matrix that correlate best with the group connectivity map when controlling for other group ICA components
### this is then used as individual per region connectvity template 

# get the indices of the x biggest correlations
#ica_gcon44_corr.argsort()[-2:][::-1]

# get the indices of correlation that are higher than the threshold
ica_44_ctrl_excl = np.argwhere(ica_gcon44_corr>corrthresh)
ica_45_ctrl_excl = np.argwhere(ica_gcon45_corr>corrthresh)

iica_incl_ind =[x for x in range(group_ica.shape[1]) if x not in np.concatenate((ica_44_ctrl_excl, ica_45_ctrl_excl)).squeeze()]

# also add the opposites area connectivity map as control
'''
grp_conn_tmpl_45.shape => (29696,), after expand_dims (29696,1)
group_ica. shape => (29696, 20)
group_ica[:,ica_44_ctrl_ind].shape = (29696, 17, 1); after squeeze: (29696, 17)

'''
controls44 = np.concatenate((group_ica[:,iica_incl_ind].squeeze(), np.expand_dims(grp_conn_tmpl_45, axis=1)), axis=1) # shape i.e (29696, 18)
controls45 = np.concatenate((group_ica[:,iica_incl_ind].squeeze(), np.expand_dims(grp_conn_tmpl_44, axis=1)), axis=1)

#hdf5storage.savemat('/data/pt_02189/MARHCP/BrocaConn/indv/CON_HCP_group_broca_LH29k_controls44.mat', {'controls44':controls44 })


### find the voxels in the indivial connectivity matrix that correlate best with the individual connectivity template/map (just derived) when controlling for individual other ICA components 
# this makes it nicer?


for sub in subs:

    #sub = '103414'

    area_assignment_file = os.path.join(out_folder_p, "indv", 'AA_HCP_indv_%s_ba44_45_none.LH.dscalar.nii' % (sub))
    if os.path.exists(area_assignment_file):
      print("Classification output already exists: \n", area_assignment_file)
      continue;

    print("------------------------------------------------------------\n")
    print("Start classification for ", sub)

    icm_fn = out_folder_p + '/indv/CON_HCP_indv_' + sub + '_broca_LH29k_con.mat'; 
    ind_cor_mat_broca =  hdf5storage.loadmat(icm_fn)['icm']

    print("Load: ",icm_fn)

    # find the broca voxel that has (have) the most sterotypical connectivity for eiteher area 44 or 45, use the ICA group components as control

    # probably need to be reduced to cort2 both group conn tmpl and controls
    #partcorr44 = partial_corr(ind_cor_mat_broca, grp_conn_tmpl_44, controls44)
    # takes input in the format of one image/data_vector per row, hence we need to transform
    partcorr44 = pengouin_part_corr_copy(ind_cor_mat_broca.T, grp_conn_tmpl_44.T, controls44.T)
    partcorr45 = pengouin_part_corr_copy(ind_cor_mat_broca.T, grp_conn_tmpl_45.T, controls45.T)
    #hdf5storage.savemat('/data/pt_02189/MARHCP/BrocaConn/indv/partcorr44.mat', {'partcorr44':partcorr44 })
    # correlates with the matlab results with r = corr(pc44,partcorr44') == 0.9966

    # 5% of the total voxel count; for estimation of the indivial connectivity maps/templates
    # average over these average_n_voxels voxels that have the highest correlation with the group connectivtiy tamplates, when correcting for other group ica compoennts and the opposites area connectivity map
    average_n_voxels = int(0.05* ind_cor_mat_broca.shape[1]) #top 5% of voxels or top 1 voxel

    ## check this again !!
    ict44 = ind_cor_mat_broca[:,partcorr44.argsort()[-average_n_voxels:]].mean(axis=1)
    ict45 = ind_cor_mat_broca[:,partcorr45.argsort()[-average_n_voxels:]].mean(axis=1)

    #if KEEP_FIRST_SUBJ_INTERM_DATA and (sub == '103414'):
    ictnew_fn = os.path.join(out_folder_p, 'indv', 'CON_HCP_indv_%s_broca_LH29k_con_tmpls_44_45.dscalar.nii' % (sub) ); 
    hcp_tools.save_dscalar( ictnew_fn, np.vstack((ict44, ict45)), brainmodel = brainmodel, scalar_names = ['ict44','ict45'])
    #hcp_tools.save_dscalar( ictnew_fn, np.expand_dims(ict44, axis=0), brainmodel = brainmodel, scalar_names = ['ict44'])
    print("Save refined individual connectivity templates to:\n",ictnew_fn)


    ### find strength of correlation of the voxels in the indivial connectivity matrix with the individual conn template when controlling for ICA components
    ### basically repeats the previous steps, just refines it by using individual data

    # first we got to find again the ICA components that correspond to connectivity best, as we dont want to use them as control components (as that would regress out variable that could be explained by the individual connectivity template)
    ind_ica_fn =os.path.join(out_folder_p,"indv", "ica_HCP_indv_" + sub+"_LH29k_20comps.mat")
    indv_ica = hdf5storage.loadmat(ind_ica_fn)['ic']
    print("Load: ",ind_ica_fn)

    ica_icon44_corr = np.zeros(20)
    ica_icon45_corr = np.zeros(20)
    #indv_ica of shape (32492,20)
    for icn  in range(indv_ica.shape[1]):
      #ica_icon44_corr[icn] = corr2(ict44, indv_ica[:, icn])
      ica_icon44_corr[icn] = np.corrcoef(ict44, indv_ica[:, icn])[0,1]
      ica_icon45_corr[icn] = np.corrcoef(ict45, indv_ica[:, icn])[0,1]

    # get the indices of correlation that are smaller than the threshold
    # these are the nn connectivity related components, hopefully
    iica_44_exl_ind = np.argwhere(ica_icon44_corr>corrthresh)
    iica_45_exl_ind = np.argwhere(ica_icon45_corr>corrthresh)
    iica_incl_ind   =[x for x in range(indv_ica.shape[1]) if x not in np.concatenate((iica_44_exl_ind, iica_45_exl_ind)).squeeze()]
    remaining_iica_ctrl = indv_ica[:,iica_incl_ind];    

    # also add the opposites area individual connectivity map as control
    indcontrols44 = np.concatenate((remaining_iica_ctrl, np.expand_dims(ict45, axis=1) ), axis=1)
    indcontrols45 = np.concatenate((remaining_iica_ctrl, np.expand_dims(ict44, axis=1) ), axis=1)

    # then do better partial correlations
    #indpartcorr44 = partialcorr(ind_cor_mat_broca, ict44, indcontrols44)    # shape (nbroca,)
    indpartcorr44 = pengouin_part_corr_copy(ind_cor_mat_broca.T, ict44.T, indcontrols44.T)
    indpartcorr45 = pengouin_part_corr_copy(ind_cor_mat_broca.T, ict45.T, indcontrols45.T)


    # Optional: run partial correlations for remaining ICA components

    
    # shape: (number of vertices in broca, number of ICA components)
    partcorrIC = np.zeros((  ind_cor_mat_broca.shape[1], remaining_iica_ctrl.shape[1]))
    for x in range(remaining_iica_ctrl.shape[1]):
      controls = np.concatenate(( np.delete(remaining_iica_ctrl, x, axis=1), np.expand_dims(ict44, axis=1), np.expand_dims(ict45, axis=1) ), axis=1)
      partcorr = pengouin_part_corr_copy(ind_cor_mat_broca.T, remaining_iica_ctrl[:,x].T, controls.T);
      partcorrIC[:,x] = partcorr;



    ##### create spatial weighting maps and do weighting
    # this basically would function as a prior i guess

    prob44 = np.loadtxt(prob44p)[cort]      # shape? ->29696???
    prob45 = np.loadtxt(prob45p)[cort]
    # log whatever 

    # reload the brca mask from before
    [op,tr] = get_individual_IFG(sub, hcp_all_path = hcp_all_path);
    broca = prob44b + prob45b + op + tr
    # reduce the broca mask to the voxels present in the corrmatrix
    broca_reduced = broca[cort].astype(np.bool) # from shape 32492 to 29696; contains 1396 x True

    prob44br = prob44[broca_reduced];
    prob45br = prob45[broca_reduced];
    prob45br = np.where(prob45br==0, prob45br[prob45br>0].min(), prob45br) #  set zeros within broca to minimum nonzero value
    prob44br = np.where(prob44br==0, prob45br[prob44br>0].min(), prob44br)

    log10 = np.log10(prob45br); norm45br = (log10 - log10.min()) / (log10.max()-log10.min())
    log10 = np.log10(prob44br); norm44br = (log10 - log10.min()) / (log10.max()-log10.min())


    #weight the individual part corrleation results with the individual anatomical information
    indpartcorr44_sw  = norm44br * indpartcorr44
    indpartcorr45_sw  = norm45br * indpartcorr45
    
    ##### WTA

    #imaps= np.concatenate((brain_ind_part_corr44_sw, brain_ind_part_corr45_sw ''', partcorrIC'''), axis=0);

    imaps = np.concatenate(( np.expand_dims(indpartcorr44_sw,axis=1), np.expand_dims(indpartcorr45_sw,axis=1), partcorrIC), axis=1)
    #wta_classification 1 ~ 44, 2 ~ 45, 3+ ~ ICA components
    area_assignments_ba44_45_ICA = np.argmax( imaps, axis=1) + 1;

    #wta_classification 1 ~ 44, 2 ~ 45, 0 ~ nieghter
    area_assignments_ba44_45_None = np.where(area_assignments_ba44_45_ICA>2, 0, area_assignments_ba44_45_ICA)

    '''def fill_sub(subset_indices, subset ,tar_size=29696): x = np.zeros(tar_size) x[subset_indices] = subset; return x;'''

    fbaa1 = np.zeros((29696)); fbaa1[broca_reduced] = area_assignments_ba44_45_None
    fbaa2 = np.zeros((29696)); fbaa2[broca_reduced] = area_assignments_ba44_45_ICA
    fbaa3 = np.zeros((29696)); fbaa3[broca_reduced] = indpartcorr44_sw
    fbaa4 = np.zeros((29696)); fbaa4[broca_reduced] = indpartcorr45_sw

    E3A = np.loadtxt('/data/hu_robertscholz/Desktop/owncloud-gwdg/code/neuralgradients/BrocaLabels/HCP/automated_labels/results/%s_ICA_indiv_SW_rm_0p4.1D' % (sub))
    EMW44 = np.loadtxt('/data/hu_robertscholz/Desktop/owncloud-gwdg/code/neuralgradients/BrocaLabels/HCP/manual_labels/area44/BA44_%s.1D' % (sub))
    EMW45 = np.loadtxt('/data/hu_robertscholz/Desktop/owncloud-gwdg/code/neuralgradients/BrocaLabels/HCP/manual_labels/area45/BA45_%s.1D' % (sub)) *2
    #hcp_tools.save_dscalar( e3afile, np.expand_dims(E3A[cort],axis=0), brainmodel = brainmodel, scalar_names = ['Estrids_Stuff'])

    indv_assignm_file =os.path.join(out_folder_p,"results", "AutoAreaLabelFinal_HCP_indv_" + sub+"_LH29k_WTA44_45.dscalar.nii")

    hcp_tools.save_dscalar( indv_assignm_file, np.stack( (fbaa1,fbaa2,E3A[cort], (EMW44+EMW45)[cort],fbaa3,fbaa4)), brainmodel = brainmodel, scalar_names = ['WTA_44_45_None', 'WTA_44_45_ICA', 'E3A-EstridsAutoAreaAssignm', 'EMW - EstrisManualWork','BA44weight', 'BA45weight'])
    print("Saved all important results to: ",indv_assignm_file)




