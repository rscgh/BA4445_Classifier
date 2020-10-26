#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
from nilearn.decomposition.canica import CanICA
import os
import hdf5storage



from partial_corr import partial_corr
from partial_corr import pengouin_part_corr_copy
#import hcp_tools
from hcp_tools import *


'''
Robert Scholz

based off an collection of scripts found on 
Original Author(s):
Estrid Jakobsen, https://github.com/Estrid
Daniel Margulies, https://github.com/margulies

@2020


last tested using python3; worked on python2 before also
requires wb_command to be on the system path
(and all the other impored libraries from above)
'''

# for testing purpose only
sub = '103414'


### Input/Resource Data; as in requires the following files: 

hcp_all_path = '/data/t_hcp/S500_2014-06-25/_all'
out_folder_p = '/data/pt_02189/MARHCP/BrocaConn/output'

subj_list_path = '/data/pt_02189/MARHCP/BrocaConn/subject_list_101.txt'

grconntmplfn = 'res/CON_HCP_all100_broca_LH29k_con_tmpls_44_45.mat'
group_ica_fn = 'res/ica_HCP_all100_LH29k_20comps.mat'

prob44p = 'res/HCP_101_ManualProbabilityMap_BA44.1D'
prob45p = 'res/HCP_101_ManualProbabilityMap_BA45.1D'; 

indices_fn = 'res/indices_LH_29696_of_32492.txt'


### Output files:

ind_ica_fn_tmpl_ds    =os.path.join(out_folder_p, "ica_HCP_indv_%s_LH29k_20comps.mat")
ind_ica_fn_tmpl_cifti =os.path.join(out_folder_p, "ica_HCP_indv_%s_LH29k_20comps.dscalar.nii")

bfn_tmpl = os.path.join(out_folder_p, 'BROCAMSK_HCP_indv_%s_%i_of_%i_bin.txt') 
icm_fn_tmpl = os.path.join(out_folder_p, 'CON_HCP_indv_%s_broca_LH29k_con.mat');

indv_assignm_file_tmpl = os.path.join(out_folder_p, "AutoAreaLabelFinal_HCP_indv_%s_LH29k_WTA44_45.dscalar.nii")
ictnew_fn_tmpl = os.path.join(out_folder_p, 'CON_HCP_indv_%s_broca_LH29k_con_tmpls_44_45.dscalar.nii'); 

### Load required data:


'''
img = nib.load('/data/t_hcp/S500_2014-06-25/_all/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii')
cort = list(img.header.matrix._mims[1].brain_models)[0].vertex_indices._indices
'''

subs = np.loadtxt(subj_list_path, dtype=str)  # should contain individual rows of strings like '100307'
subs = [sub for sub in subs if sub != '111009'] # this subject is not existing in the HCP500 distribution
cort = np.loadtxt(indices_fn).astype(np.int).tolist()


# get manual group probability maps; as binary and as "normal"
#prob44p = '/data/hu_robertscholz/Desktop/owncloud-gwdg/code/neuralgradients/BrocaLabels/HCP/manual_labels/ProbabilityMap_BA44.1D'
prob44b = np.loadtxt(prob44p)           # -> .shape == (32492,), np.count_nonzero(prob44) -> 738
prob44b[prob44b>0] = True
#prob45p = '/data/hu_robertscholz/Desktop/owncloud-gwdg/code/neuralgradients/BrocaLabels/HCP/manual_labels/ProbabilityMap_BA45.1D'; 
prob45b = np.loadtxt(prob45p)           # -> .shape == (32492,), np.count_nonzero(prob44) -> 869
prob45b[prob45b>0] = True

prob44 = np.loadtxt(prob44p)[cort]      # shape? ->29696???
prob45 = np.loadtxt(prob45p)[cort]

grp_conn_tmpl_44 =  hdf5storage.loadmat(grconntmplfn)['ct44']
grp_conn_tmpl_45 =  hdf5storage.loadmat(grconntmplfn)['ct45']

group_ica = hdf5storage.loadmat(group_ica_fn)['ic']



###############################################################################
## Start of classifier proper


# timeseries should be only include the left hemisphere and be of shape (29696, 4800)
def classify_subject(subid, timeseries=None, corrthresh = 0.4, save_intermediate=False):

  sub = subid
  #  variables for classifier 
  #corrthresh = 0.4;
  #timeseries = None


  ###############################################################################
  ## Get the timeseries
  ###############################################################################

  if timeseries is None:

      data = preprocess_and_load_tseries(hcp_all_path + "/%s" % sub, sub, N_first = 0, N_cnt = 32492, smoothing = True, normalize=True, temp_dir= out_folder_p, keep_tmp_files= save_intermediate) # returns (4800, 32492)

      # get only what is relevant for the left side
      timeseries = data[:29696, :];

      #cimg = nib.Cifti2Image(data.T[:, :29696], nih_full)
      #cimg.to_filename(os.path.join(out_folder_p, "indv",'rsfmri_103414_1-4.LH.smooth.dtseries.nii'));

      tsimg = nib.Nifti1Image(timeseries.reshape((29696, 1, 1, timeseries.shape[1])), np.eye(4)) # (29696, 1, 1, 4800); img.to_filename(filename)
      print("Timeseries.shape: ", timeseries.shape)

  ###############################################################################
  ## Individual ICA
  ## - Input: individual timeseries
  ## - Output: 20 individal ICA components of shape (32492, 20)    -> can be saved as cift or mat file
  ##
  ###############################################################################


  ## RUN ICA ON INDIVIDUAL LEVEL:

  ind_ica_fn =os.path.join(out_folder_p, "ica_HCP_indv_" + sub+"_LH29k_20comps.mat")

  msk = nib.Nifti1Image(np.ones((29696,1,1)), np.eye(4))
  canica_ind = CanICA(mask=msk, n_components=20, smoothing_fwhm=0., threshold=None, verbose=10, random_state=0, n_jobs=20)
  canica_ind.fit(tsimg)  
      
  # Retrieve the independent components in brain space
  components_img_ind = canica_ind.masker_.inverse_transform(canica_ind.components_)

  indv_ica = components_img_ind.get_data().squeeze()

  #print("  Saving Indv ICA: ", ind_ica_fn)
  #hdf5storage.savemat(ind_ica_fn_tmpl.replace("%s",sub), {'ic': indv_ica}) # (29696,n_components)
  #nib.Cifti2Image(Aind, nih).to_filename(ind_ica_fn_tmpl_cifti.replace("%s",sub));

  if save_intermediate:
    quick_cifti_ds(indv_ica.T, dsnames = ['comp' + str(x+1) for x in range(indv_ica.shape[1])], fn=ind_ica_fn_tmpl_cifti.replace("%s",sub))


          
  ###############################################################################
  ## Create Individual Correlation Matrix
  ##
  ## - Input: individual timeseries stored as nii.gz of shape (29696, 1, 1, 4800)
  ##   /data/pt_02189/MARHCP/BrocaConn/indv/100307_cifti_tseries_shape_sV-1-1-T.nii.gz'
  ##    * indices that were used to recude the left hemisphere from 32492 svox to 29696
  ## - Output: 
  ##    * binary brocas mask of size (32492,) 
  ##        saved in /data/pt_02189/MARHCP/BrocaConn/indv/BROCAMSK_HCP_indv_brocas_mask_1396_of_32492_bin.txt
  ##     * (optional) Individual correlation matrices of shape: (29696, 1396)
  ##        saved as /data/pt_02189/MARHCP/BrocaConn/indv/100307_corrmat_broca_29696x 1396_nosmt.mat
  ## 
  ###############################################################################


  #timeseries is of shape (29696, 4800)

  # matlab: corrcoef(A) where Dim(A) = (5 observations/time points, 3 voxels to correlate with each other)  ~ rows represent observations.
  # numpy: Each row of x represents a variable, and each column a single observation of all those variables. Also see rowvar below. ~ columns represent observations
  cmat = np.corrcoef(timeseries) # shape (29696, 29696) each voxel with each other
  # cmat[1000,2000] == cmat[2000,1000]

  [op,tr] = get_individual_IFG(sub, hcp_all_path = hcp_all_path);
  broca = prob44b + prob45b + op+tr;
  # reduce the broca mask to the voxels present in the corrmatrix
  broca_reduced = broca[cort].astype(np.bool) # from shape 32492 to 29696; contains 1396 x True
  bfn = bfn_tmpl % (sub, np.count_nonzero(broca_reduced) , len(cort))
  if save_intermediate: np.savetxt(bfn, broca_reduced)

  # make the small corrmat
  ind_cor_mat_broca = cmat[:,broca_reduced];    #  shape: (29696, 1396); subsection from  29696 x 29696 matrix

  if save_intermediate:
    icm_fn=icm_fn_tmpl % (sub)
    print("Saving Indv redcued connectivity matrix: ", icm_fn)
    hdf5storage.savemat(icm_fn, {'icm': ind_cor_mat_broca.astype(np.float32)})



  ###############################################################################
  ## Classify Ba44/45 in subjects now ....
  ###############################################################################

  print("\n\n\n\nFinally Start with the classification: ")


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


  ### find the voxels in the indivial connectivity matrix that correlate best with the individual connectivity template/map (just derived) when controlling for individual other ICA components; this makes it nicer?


  print("------------------------------------------------------------\n")
  print("Start classification for ", sub)

  print("Individual correlation matrix is already prepared: ind_cor_mat_broca of shape ",ind_cor_mat_broca.shape)

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
  #ictnew_fn = os.path.join(out_folder_p, 'indv', 'CON_HCP_indv_%s_broca_LH29k_con_tmpls_44_45.dscalar.nii' % (sub) ); 
  #hcp_tools.save_dscalar( ictnew_fn, np.vstack((ict44, ict45)), brainmodel = brainmodel, scalar_names = ['ict44','ict45'])
  #hcp_tools.save_dscalar( ictnew_fn, np.expand_dims(ict44, axis=0), brainmodel = brainmodel, scalar_names = ['ict44'])

  if save_intermediate:
    quick_cifti_ds(np.vstack((ict44, ict45)), dsnames = ['ict44','ict45'] , fn = ictnew_fn_tmpl % (sub))
    print("Save refined individual connectivity templates to:\n", ictnew_fn_tmpl % (sub))


  ### find strength of correlation of the voxels in the indivial connectivity matrix with the individual conn template when controlling for ICA components
  ### basically repeats the previous steps, just refines it by using individual data

  # first we got to find again the ICA components that correspond to connectivity best, as we dont want to use them as control components (as that would regress out variable that could be explained by the individual connectivity template)

  #ind_ica_fn =os.path.join(out_folder_p,"indv", "ica_HCP_indv_" + sub+"_LH29k_20comps.mat")
  #indv_ica = hdf5storage.loadmat(ind_ica_fn)['ic']
  print("Individual ICA data is already prepared: indv_ica of shape: ", indv_ica.shape)

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


  #hcp_tools.save_dscalar( indv_assignm_file_tmpl % (sub), np.stack( (fbaa1,fbaa2,fbaa3,fbaa4)), brainmodel = brainmodel, scalar_names = ['WTA_44_45_None', 'WTA_44_45_ICA','BA44weight', 'BA45weight'])

  quick_cifti_ds(np.stack( (fbaa1,fbaa2,fbaa3,fbaa4)), dsnames = ['WTA_44_45_None', 'WTA_44_45_ICA','BA44weight', 'BA45weight'] , fn = indv_assignm_file_tmpl % (sub))
  print("Saved all important results to: ", indv_assignm_file_tmpl % (sub))




