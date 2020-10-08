

'''
cd /data/pt_02189/MARHCP/BrocaConn/sample/

103414_cifti_tseries_shape_sV-1-1-T.nii.gz -> shape (29696, 1, 1, 4800);
dont use this one, just do anew, because here important info might already be lost ...
'''




small = nib.load("rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii"); #shape (1200, 91282)
#small.header.get_axis(1).name.shape -> (91282,)
#this is really only relevant for later on display, but not for internal saving ...
#small_lh_indices =list(small.header.matrix._mims[1].brain_models)[0].vertex_indices._indices


# make a longer time series axis
old_series_axis = small.header.get_axis(0);
new_series_axis = nib.cifti2.SeriesAxis(old_series_axis.start, old_series_axis.step, 4800)



# also just take the left hemisphere
mask = np.zeros((32492)); np.put(mask, cort, 1)
bm_leftown = nib.cifti2.BrainModelAxis.from_mask(mask, "LEFT_CORTEX")

# combine the new header
#nih_old = nib.cifti2.Cifti2Header.from_axes((old_series_axis, small.header.get_axis(1)))

# get just one structure from the images' brain model
lh_bma = list(small.header.get_axis(1).iter_structures())[0][2]

#nih_sigl = nib.cifti2.Cifti2Header.from_axes((old_series_axis, bm_leftown)) -> is a little bigger
nih_sigl = nib.cifti2.Cifti2Header.from_axes((old_series_axis, lh_bma))
nih_full = nib.cifti2.Cifti2Header.from_axes((new_series_axis, lh_bma))



cimg = nib.Cifti2Image( small.get_fdata()[:, :29696], nih_sigl)
cimg.to_filename('rfMRI_REST1_LR_Atlas_hp2000_clean.LH.dtseries.nii');


data = t_series(subject = hcp_all_path + "/%s" % sub, N_first = 0, N_cnt = 32492, normalize=True).T
# returns (4800, 32492), then transformed to (4800, 32492)

cimg = nib.Cifti2Image(data[:, :29696], nih_full)
cimg.to_filename('rsfmri_103414_1-4.LH.dtseries.nii');


'''
(1) The functional time-series
data of the left cerebral cortex was extracted for each of four 15-min
rfMRI scans (TR=0.7 s) per subject, (2) surface-based smoothing with
a 2 mm FWHM kernel was applied

## both of the following approaches seem to work ...


wb_command -cifti-smoothing <cifti> <surface-kernel> <volume-kernel> <direction> <cifti-out>

wb_command -cifti-smoothing rsfmri_103414_1-4.LH.dtseries.nii 2 2 COLUMN rsfmri_103414_1-4.LH.smooth.dtseries.nii -left-surface /data/t_hcp/S500_2014-06-25/_all/103414/MNINonLinear/fsaverage_LR32k/103414.L.midthickness.32k_fs_LR.surf.gii


wb_command -cifti-smoothing rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii 2 2 COLUMN rfMRI_REST1_LR_Atlas_hp2000_clean.smooth2.dtseries.nii -left-surface /data/t_hcp/S500_2014-06-25/_all/103414/MNINonLinear/fsaverage_LR32k/103414.L.midthickness.32k_fs_LR.surf.gii -right-surface /data/t_hcp/S500_2014-06-25/_all/103414/MNINonLinear/fsaverage_LR32k/103414.R.midthickness.32k_fs_LR.surf.gii 

wb_view
open the /data/t_hcp/S500_2014-06-25/_all/103414/MNINonLinear/fsaverage_LR32k/ 
aspect file
then open the individual files generated above and use them as overlays ...
'''








### Scalar Image:

prob44p = '/data/hu_robertscholz/Desktop/owncloud-gwdg/code/neuralgradients/BrocaLabels/HCP/manual_labels/ProbabilityMap_BA44.1D'
prob44b = np.loadtxt(prob44p)

new_scalar_axis = nib.cifti2.ScalarAxis(['prob44']);

mask = np.zeros((32492)); np.put(mask, cort, 1)
bm_leftown = nib.cifti2.BrainModelAxis.from_mask(mask, "LEFT_CORTEX")

#nih_scal = nib.cifti2.Cifti2Header.from_axes((new_scalar_axis, bm_leftown))
nih_scal = nib.cifti2.Cifti2Header.from_axes((new_scalar_axis, lh_bma))

#cimg = nib.Cifti2Image( prob44b.reshape((1,32492)), nih_scal)
cimg = nib.Cifti2Image( prob44b[:29696].reshape((1,29696)), nih_scal)
cimg.to_filename('prob44b.LH.dscalar.nii');

### this is the correct version:
cimg = nib.Cifti2Image( prob44b[cort].reshape((1,29696)), nih_scal)
cimg.to_filename('prob44b_cort.LH.dscalar.nii');




#### Plotting surface data:

https://nilearn.github.io/modules/generated/nilearn.plotting.plot_surf.html

from nilearn import plotting
import numpy as np

out_folder_p = '/data/pt_02189/MARHCP/BrocaConn'
cort2 = np.loadtxt(out_folder_p + '/indices_LH_29696_of_32492.txt').astype(np.int).tolist()

cimg = nib.load("/data/pt_02189/MARHCP/BrocaConn/sample/prob44b_cort.LH.dscalar.nii")
ctimg = nib.load("/data/pt_02189/MARHCP/BrocaConn/indv/103414_cifti_tseries_shape_sV-1-1-T.nii.gz")

fimg = ctimg.get_fdata().squeeze()[:,0];
fimg = cimg.get_fdata().squeeze();

sulci = nib.load("/data/t_hcp/S500_2014-06-25/_all/103414/MNINonLinear/fsaverage_LR32k/103414.sulc.32k_fs_LR.dscalar.nii").get_fdata().squeeze()
sulci_LH = sulci[:32492]
sulci_LH = np.zeros((32492))
sulci_LH[cort2] =  sulci[cort2]
#plotting.plot_surf(surf_mesh=LH_mesh_inflated, surf_map=sulci_LH)


fimg_big = np.zeros((32492))
fimg_big[cort2] = fimg

LH_mesh_flat = "/data/t_hcp/S500_2014-06-25/_all/103414/MNINonLinear/fsaverage_LR32k/103414.L.flat.32k_fs_LR.surf.gii"
LH_mesh_inflated = "/data/t_hcp/S500_2014-06-25/_all/103414/MNINonLinear/fsaverage_LR32k/103414.L.inflated.32k_fs_LR.surf.gii"
#LH_mesh_inflated_other = "/data/t_hcp/S500_2014-06-25/_all/156233/MNINonLinear/fsaverage_LR32k/156233.L.inflated.32k_fs_LR.surf.gii"	# dont do because it leads to distortions
LH_mesh_mid = "/data/t_hcp/S500_2014-06-25/_all/103414/MNINonLinear/fsaverage_LR32k/103414.L.midthickness.32k_fs_LR.surf.gii"

plotting.plot_surf(surf_mesh=LH_mesh_inflated, surf_map=fimg_big, bg_map=sulci_LH)
plotting.plot_surf(surf_mesh=LH_mesh_flat, surf_map=fimg_big, bg_map=sulci_LH, view='ventral')	#view: {‘lateral’, ‘medial’, _‘dorsal’_, ‘ventral’, ‘anterior’, ‘posterior’},
plotting.show()


gii = nib.load(LH_mesh)
gii.print_summary()







######################################### Oddities of Cifti / HCP ################################################################################################
##### differences ...

## LH is internally stored in an array of size 0:29696
## LH is mappend (internally) onto a 32k space for each hemisphere, that additionally inludes the medial wall	

## this mapping may be valid for Natie as well as fs32k_LH space (? not sure on this one)

# should be registered to fs32k_LH?
hcp_all_path = '/data/t_hcp/S500_2014-06-25/_all'
img = nib.load(hcp_all_path + '/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii')

# aparently for whatever reason (maybe compatibility to before or coherence with the gii/mesh file) cortex structures are given bigger than they are actually internally encoded
surf_axis = img.header.get_axis(1)	#-> <BrainModelAxis object>; nvertices = {'CIFTI_STRUCTURE_CORTEX_RIGHT': 32492, 'CIFTI_STRUCTURE_CORTEX_LEFT': 32492}; size= (91282,)
np.argwhere(surf_axis.name=='CIFTI_STRUCTURE_CORTEX_LEFT').astype(int).squeeze().tolist() == [x for x in range(29696)]; # surf_axis.name.shape = (91282,),
structs = list(surf_axis.iter_structures())

for x in structs:: print(x)
'''
(u'CIFTI_STRUCTURE_CORTEX_LEFT', slice(0, 29696, None), <nibabel.cifti2.cifti2_axes.BrainModelAxis object at 0x7f68ce328b90>)
(u'CIFTI_STRUCTURE_CORTEX_RIGHT', slice(29696, 59412, None), <nibabel.cifti2.cifti2_axes.BrainModelAxis object at 0x7f68d460add0>)
(u'CIFTI_STRUCTURE_ACCUMBENS_LEFT', slice(59412, 59547, None), <nibabel.cifti2.cifti2_axes.BrainModelAxis object at 0x7f68d74ccdd0>)
'''

structs[0][2] #.nvertices -> {'CIFTI_STRUCTURE_CORTEX_LEFT': 32492}, size -> 29696, type is BrainModelAxis; this can help creating cifti headers for partial cifti files ...


[x[0] for x in structs]
'''
[u'CIFTI_STRUCTURE_CORTEX_LEFT', u'CIFTI_STRUCTURE_CORTEX_RIGHT', u'CIFTI_STRUCTURE_ACCUMBENS_LEFT', u'CIFTI_STRUCTURE_ACCUMBENS_RIGHT', u'CIFTI_STRUCTURE_AMYGDALA_LEFT', u'CIFTI_STRUCTURE_AMYGDALA_RIGHT', u'CIFTI_STRUCTURE_BRAIN_STEM', u'CIFTI_STRUCTURE_CAUDATE_LEFT', u'CIFTI_STRUCTURE_CAUDATE_RIGHT', u'CIFTI_STRUCTURE_CEREBELLUM_LEFT', u'CIFTI_STRUCTURE_CEREBELLUM_RIGHT', u'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT', u'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT', u'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT', u'CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT', u'CIFTI_STRUCTURE_PALLIDUM_LEFT', u'CIFTI_STRUCTURE_PALLIDUM_RIGHT', u'CIFTI_STRUCTURE_PUTAMEN_LEFT', u'CIFTI_STRUCTURE_PUTAMEN_RIGHT', u'CIFTI_STRUCTURE_THALAMUS_LEFT', u'CIFTI_STRUCTURE_THALAMUS_RIGHT']'''


surf_axis.vertex	# 91282; 
''' 
this maps internal vertice index to vertex number in the target/reference/appropriate 3D mesh/gii file:
																 [ Left Hemisphere,  29696 vertices starting at 0, thus ending with 29695                  ]  [Right Hemisohere...]
# indices of surf_axis.vertex ~ internal vertex index/number   : [0,1,2,...,6,7,8, ... 19, 20, 21, ...,  75,  76, ...                              ... 29695,  29696, 29697, ..., ]
# surf_axis.vertex            ~ target/gii vertex index/number : [0,1,2,...,6,8,9, ... 20, 68, 69, ..., 123, 162, 163, 164, ... 187, 235, 236, 237 ... 32491,      0,     1,   2, ]

for each structure we seem to begin counting with zero again; missing numbers in target/gii correspond to the medial wall which is not meaningful and hence doesnt need to be saved)
non-surface structures (and hence non existing vertices) receive a -1 as mapping value
'''

# mims gives the second dimension of the data array (so the surface) and the first structure in it (0) gives the Left hemisphere
cort = list(img.header.matrix._mims[1].brain_models)[0].vertex_indices._indices
# mapping of data array structure - given by the structs - to fs32k_LR mesh/.gii - given here
# cort                     : [0,1,2,...,6,8,9, ... 20, 68, 69, ..., 123, 162, 163, 164, ... 187, 235, 236, 237 ... 32491,
# list(img.header.matrix._mims[1].brain_models)[0].index_offset index_count -> 0, 29696
# hence the first part of the vertex mapping corresponds exactly to cort:
surf_axis.vertex[:29696].astype(int).tolist() == cort	# yields true







# should be registered to fs32k_LH?
img2 = nib.load('/data/pt_02189/MARHCP/BrocaConn/sample/prob44b_cort.LH.dscalar.nii')
surf_axis2 = img2.header.get_axis(1) #-> <BrainModelAxis object>; nvertices = {'CIFTI_STRUCTURE_CORTEX_LEFT': 32492}; size= (29696,)

np.argwhere(surf_axis2.name=='CIFTI_STRUCTURE_CORTEX_LEFT').astype(int).squeeze().tolist() == [x for x in range(29696)]; # surf_axis.name.shape = (29696,),
# so basically all name entities are CIFTI_STRUCTURE_CORTEX_LEFT

structs2 = list(surf_axis2.iter_structures())
#(u'CIFTI_STRUCTURE_CORTEX_LEFT', slice(0, None, None), <nibabel.cifti2.cifti2_axes.BrainModelAxis object at 0x7f68d8ec9d90>)
# we got only one structure

surf_axis2.vertex	# shape: (29696,)	-> provides the mapping to the standard surface
# surf_axis.vertex            ~ target/gii vertex index/number : [0,1,2,...,6,8,9, ... 20, 68, 69, ..., 123, 162, 163, 164, ... 187, 235, 236, 237 ... 32491,      0,     1,   2, ]
# this mapping is comprehensive for all strucutres, but aparently unnessesary


'''

https://www.humanconnectome.org/software/workbench-command/-cifti-help
http://www.nitrc.org/projects/cifti/
https://www.nitrc.org/forum/attachment.php?attachid=341&group_id=454&forum_id=1955

'''


Dense Connectivity
Intent_code: 3001, NIFTI_INTENT_CONNECTIVITY_DENSE
Intent_name: ConnDense
File extension: .dconn.nii
AppliesToMatrixDimension 0: brain models
AppliesToMatrixDimension 1: brain models
This file type represents connectivity between points in the brain. A row is the connectivity from
a single vertex or voxel in the mapping that applies along the second dimension, to all vertices
and voxels along the first dimension. This specification of “from” and “to” is not intended to
imply that the data is always directed, but to establish a convention so that directed data is
stored consistently, and to ensure that interactive usage loads rows from the matrix, in order to
maximize responsiveness. Note that this type can have a single mapping apply to both
dimensions, but can also have separate, different mappings for rows and columns, for instance
only containing connectivity from left cortex to cerebellum

Parcellated Connectivity
Intent_code: 3003, NIFTI_INTENT_CONNECTIVITY_PARCELLATED
Intent_name: ConnParcels
File extension: .pconn.nii
 14
AppliesToMatrixDimension 0: parcels
AppliesToMatrixDimension 1: parcels
This file type represents connectivity between areas or parcels of the brain. Similarly to Dense
Connectivity, a row is the connectivity from a single parcel in the mapping that applies along the
second dimension, to all parcels along the first dimension. Note that this type can have a single
mapping apply to both dimensions, but can also be asymmetric, for example if a parcellated
connection was from a subset of cortical areas to all of them (e.g. injection sites for tracer data).


##########################################################################################################################################################
