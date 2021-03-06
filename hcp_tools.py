#!/usr/local/bin/python
# -*- coding: utf-8 -*-

'''

preprocess_and_load_tseries is based off the original script by Şeyma Bayrak, https://github.com/sheyma
repository located on: https://github.com/NeuroanatomyAndConnectivity/hcp_corr

code was a bit more streamlined and put up to date
furthermore the option for smoothing was added; for that the wb_command needs to be available from the systems terminal

'''



## called from ICA_HCP like:
## t_series(subject = "/scr/murg2/HCP_Q3_glyphsets_left-only/100307", hemisphere='LH', N_first=0, N_cnt=32492)
## K = t_series(subject = "/data/t_hcp/S500_2014-06-25/_all/100307", hemisphere='LH', N_first=0, N_cnt=32492)

# from load_hcp_annotated_rsc import t_series

from glob import glob
import os
import nibabel as nib
import numpy as np
import subprocess

from matplotlib import pyplot as plt


############################
# Global import data


cort = np.loadtxt('res/indices_LH_29696_of_32492.txt').astype(np.int).tolist()

mask = np.zeros((32492)); np.put(mask, cort, 1)
bm_leftown = nib.cifti2.BrainModelAxis.from_mask(mask, "LEFT_CORTEX")




############################
# Saving files



def save_LH_29k_dtseries():
    pass;




# this is not really functional yet ...
def get_LH_29k_brainmodel(mask_indices = None, extract_brain_mode_from_file = None, area_list = ["LH"] ):

  if mask_indices is None: 
      if extract_brain_mode_from_file is None:
        extract_brain_mode_from_file = '/data/t_hcp/S500_2014-06-25/_all/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii';

      img = nib.load(extract_brain_mode_from_file)
      cort = list(img.header.matrix._mims[1].brain_models)[0].vertex_indices._indices
      mask_indices = cort

  mask = np.zeros((32492)); np.put(mask, mask_indices, 1)
  brainmodel = nib.cifti2.BrainModelAxis.from_mask(mask, "LEFT_CORTEX")
  return brainmodel



# data should be numpy array of (n_scalars, n_vertices) i.e. 10, 29696
def save_dscalar(filename, data, brainmodel, scalar_names = None, subset = None):

  n_scalars = data.shape[1];
  n_vertices = brainmodel.size;    # i.e. 29696 for only left hemisphere in 32k_FS_LR

  if scalar_names is None: 
    scalar_names = [str(x) for x in range(data.shape[1])]

  new_scalar_axis = nib.cifti2.ScalarAxis(scalar_names);
  ni_header = nib.cifti2.Cifti2Header.from_axes((new_scalar_axis, brainmodel))

  if not(subset is None):
    newdata = np.zeros((n_scalars, n_vertices))
    newdata[:,subset] = data;
    data = newdata;

  nib.Cifti2Image( data, ni_header).to_filename(filename);
  return;


# Visualization and saving helper functions:

def quick_cifti_ds(data, dsnames = None, fn = None, return_img=False):
  global bm_leftown;
  if dsnames is None: dsnames = ['img%i' % (x) for x in range(data.shape[0])]
  cimgvis4 = nib.Cifti2Image(data, nib.cifti2.Cifti2Header.from_axes((nib.cifti2.ScalarAxis(dsnames), bm_leftown)))
  if not (fn is None): cimgvis4.to_filename(fn);
  if return_img: return cimgvis4;
  return;

def quick_show_FS32k(mesh="flat", ref="S1200"):
  #global ...
  pass


from matplotlib.ticker import MaxNLocator

def imtlshow(img, plot = plt, ax = plt.gca(), show = True):
 plt.imshow(img.T)
 tlshow(show=show)

def tlshow(plot = plt, ax = plt.gca(), show=True):
    #ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    #ax.yaxis._update_ticks()
    #ax.yaxis.set_ticks(ax.yaxis.get_major_ticks()[::-1]) # set y-ticks
    #ax.yaxis.tick_left()                    # remove right y-Ticks
    ax.invert_yaxis()
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.xaxis.tick_top()                     # and move the X-Axis      
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if show: plt.show()









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







#previously called: t_series
def preprocess_and_load_tseries(subject_dir,      # i.e. "/scr/murg2/HCP_Q3_glyphsets_left-only/100307"
             subject_id,                                 # i.e. 100307
             template = None,
             cnt_files=4,
             hemisphere='LH',
             N_first=None,      # i.e. 0
             N_cnt=None,        # i.e. 32492    svoxels of one hemisphere
             dtype=None,
             smoothing=False,   # recently added, done before normalization and concatenation; 
             temp_dir ="",      # used for storing the intermediate smoothing files, if none is given, the script execution dir is used
             keep_tmp_files = False,
             normalize=True):
                
    """Load/write Human Connectome Project (HCP) neuroimaging files via NiBabel 
    module. The HCP data is released in GIFTI format (*nii extention) for 
    almost 500 subjects. This script aims to get concetanation of all 
    time-series for each subject. 
    
    subject : string
         subject = data_path + subject_id. 
         e.g. subject = '/a/documents/connectome/_all/100307'
     
    template : string
        Template is the sketch-name of *.nii files (GIFTI format), it is hard-
        coded as template_flag and template_orig...
        
    cnt_files : int
        Number of *.nii files of interest for a subject. The template above
        has 4 forms in total, therefore cnt_files = 4
   
    hemisphere : string

        # LH: CORTEX_LEFT  >> N_first = 0, N_cnt = 29696 
        # RH: CORTEX_RIGHT  >> N_first = 29696, N_cnt = 29716
        # UL: ACCUMBENS_LEFT >> N_first = 59412, N_cnt = 135
        # UR: ACCUMBENS_RIGHT >> N_first = 59547, N_cnt = 140
        # ML: AMYGDALA_LEFT  >> N_first = 59687, N_cnt = 315
        # MR: AMYGDALA_RIGHT  >> N_first = 60002, N_cnt = 332
        # BS: BRAIN_STEM   >> N_first = 60334, N_cnt = 3472
        # CL: CAUDATE_LEFT  >> N_first = 63806, N_cnt = 728
        # CR: CAUDATE_RIGHT >> N_first = 64534, N_cnt = 755
        # EL: CEREBELLUM_LEFT >> N_first = 65289, N_cnt = 8709  
        # ER: CEREBELLUM_RIGHT  >> N_first = 73998, N_cnt = 9144
        # DL: DIENCEPHALON_VENTRAL_LEFT  >> N_first = 83142, N_cnt = 706 
        # DR: DIENCEPHALON_VENTRAL_RIGHT  >> N_first = 83848, N_cnt = 712
        # HL: HIPPOCAMPUS_LEFT  >> N_first = 84560, N_cnt = 764
        # HR: HIPPOCAMPUS_RIGHT  >> N_first = 85324, N_cnt = 795 
        # PL: PALLIDUM_LEFT  >> N_first = 86119, N_cnt = 297
        # PR: PALLIDUM_RIGHT >> N_first = 86416, N_cnt = 260
        # AL: PUTAMEN_LEFT  >> N_first = 86676, N_cnt = 1060
        # AR: PUTAMEN_RIGHT  >> N_first = 87736, N_cnt = 1010
        # TL: THALAMUS_LEFT  >> N_first = 88746, N_cnt = 1288
        # TR: THALAMUS_RIGHT >> N_first = 90034, N_cnt = 1248
        # full : all of them >> N_first = 0, N_cnt = 91282

    K : output, numpy.ndarray
        Concetanation of time-series matrices obtained from each *.nii file. 
    
    References :
        http://www.humanconnectome.org/
        https://github.com/satra/nibabel/tree/enh/cifti2
        right nibabel version to download:        
        $ git clone --branch enh/cifti2 https://github.com/satra/nibabel.git
    
    """
            

    '''
    subject='/data/t_hcp/S500_2014-06-25/_all/100307'
    subject='/data/t_hcp/S500_2014-06-25/_all/103414'
    '''

    template_flat = 'rfMRI_REST?_??_Atlas_hp2000_clean.dtseries.nii'
    template_orig = 'MNINonLinear/Results/rfMRI_REST?_??/rfMRI_REST?_??_Atlas_hp2000_clean.dtseries.nii'

    '''
    from glob import glob
    glob(os.path.join(subject, template_flat))
    files = glob(os.path.join(subject, template_orig))  -> yields 4 files:
        /data/t_hcp/S500_2014-06-25/_all/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii
        /data/t_hcp/S500_2014-06-25/_all/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST[1|2]_[LR|RL]_Atlas_hp2000_clean.dtseries.nii

    '''

    # search files in given and default templates
    files = []
    if template != None:
        files = [val for val in sorted(glob(os.path.join(subject_dir, template)))]
    if len(files) == 0:
        files = [val for val in sorted(glob(os.path.join(subject_dir, template_flat)))]
    if len(files) == 0:
        files = [val for val in sorted(glob(os.path.join(subject_dir, template_orig)))]

    if len(files) < cnt_files:
        return []
        #raise Exception('Not enough files found!')

    files = files[:cnt_files]
    print(files)
    smooth_files = []

    ### Smoothing 

    if smoothing:

      for filen in files:

        local_tar_filename = os.path.join(temp_dir,"smooth_tmp_%s_" %(subject_id) + os.path.split(filen)[-1])
        left_surface_file = os.path.join(subject_dir, "MNINonLinear/fsaverage_LR32k/%s.L.midthickness.32k_fs_LR.surf.gii" % (subject_id))
        right_surface_file = os.path.join(subject_dir, "MNINonLinear/fsaverage_LR32k/%s.R.midthickness.32k_fs_LR.surf.gii" % (subject_id))
        print("left_surface_file: ", left_surface_file)

        if os.path.exists(local_tar_filename):
            smooth_files.append(local_tar_filename)
            continue; 

        command = "wb_command -cifti-smoothing %s 2 2 COLUMN %s -left-surface %s -right-surface %s" % (filen, local_tar_filename, left_surface_file, right_surface_file)
        print("Smoothing now: ", filen, " using the following command:\n", command)
        subprocess.call(command.split())
        print("Done.")
        smooth_files.append(local_tar_filename)

      files = smooth_files;

    print("Final smooth files: ", smooth_files)

    
    # dictionary for brain structures
    label_index = { 'LH':0, 'RH':1, 'UL':2, 'UR':3, 'ML':4, 'MR':5, 'BS':6,
                 'CL':7, 'CR':8, 'EL':9, 'ER':10, 'DL':11, 'DR':12, 'HL':13,
                 'HR': 14, 'PL':15, 'PR':16, 'AL': 17, 'AR':18, 'TL':19, 
                 'TR':20 }
    
    for x in range(0, cnt_files):

        # x = 1; x=2...
        # import nibabel as nb

        print("load file: ", files[x])
        img = nib.load(files[x])
        
        # if beginning and end indices given manually        
        if (N_first != None and N_cnt != None):
            # img.data is decrepeted; now usw img.get_fdata() (should be the same as img.get_data())
            # img.get_fdata().shape ~ (1200, 91282), 1200 timepoints x 91282 svoxels; just select the left hemisphere voxels
            #single_t_series = img.data[:, N_first:N_first+N_cnt].T
            single_t_series = img.get_fdata()[:, N_first:N_first+N_cnt].T
            # yields a numpy.ndarray of shape (32492, 1200) with floating point values float64 ...

        # if a particular brain structure wanted
        # seems like hemisphere is ignored if N_first and N_cnt are given
        elif hemisphere != 'full':
            
            # find out the indices of brain structure of interest
            hem = label_index[hemisphere]       # yields 0 for 'LH'
    
            print("BRAIN STRUCTURE: ")            
            # print img.header.matrix.mims[1].brainModels[hem].brainStructure
            print(list(img.header.matrix._mims[1].brain_models)[hem].brain_structure)    # -> CIFTI_STRUCTURE_CORTEX_LEFT
            
            N_first = list(img.header.matrix._mims[1].brain_models)[hem].index_offset    # -> 0
            N_cnt = list(img.header.matrix._mims[1].brain_models)[hem].index_count       # -> 29696
                
            #single_t_series = img.data[:, N_first:N_first+N_cnt].T
            single_t_series = img.get_fdata()[:, N_first:N_first+N_cnt].T

        # if all brain nodes wanted
        elif hemisphere == 'full':
            
            N_first = 0
            hem = 1
            N_tmp = list(img.header.matrix._mims[1].brain_models)[hem].index_offset
            N_cnt = list(img.header.matrix._mims[1].brain_models)[hem].index_count 
            N_cnt += N_tmp
            
            #single_t_series = img.data[:, N_first:N_first+N_cnt].T
            single_t_series = img.get_fdata()[:, N_first:N_first+N_cnt].T

        # length of time series 
        m = single_t_series.shape[1]
        n = single_t_series.shape[0]        
        
        m_last = m
        n_last = n

        if x == 0:
            # In first loop we initialize matrix K to be filled up and returned
            # By default we are using the same dtype like input file (float32)
            init_dtype = single_t_series.dtype if dtype == None else dtype
            K = np.ndarray(shape=[n,m], dtype=init_dtype, order='F')
        else:
            if  m_last != m:
                print("Warning, %s contains time series of different length" % (subject_dir))
            if  n_last != n:
                print("Warning, %s contains different count of brain nodes" % (subject_dir))
            K.resize([n, K.shape[1]+m])

        # concatenation of (normalized) time-series, column-wise
        if normalize:
            mean_series = single_t_series.mean(axis=1)
            std_series = single_t_series.std(axis=1)
            K[:, -m:] = ((single_t_series.T - mean_series) / std_series).T            

        else:
            K[:, -m:] = single_t_series
        del img
        del single_t_series

    # remove the tempoary smooth files
    if not keep_tmp_files: 
     print("Remove files: ", smooth_files)
     for file in smooth_files: os.remove(file)

        
    return K


