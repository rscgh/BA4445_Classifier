




# HCP Processing Pipeline:

Usually human connectome data seems to be aligned to the 32k_FS_LR atlas space (but also alignments to fsaverage are available) based on the Conte69 "mesh".

Alignment as part of the HCP Pipeline works as follows: 

* PreFreesurfer Part (Volume Spaces):
[1] subject’s undistorted native volume space (rigidly “aligned” to the axes of MNI space
[2]  Standard MNI space, which is more accurately aligned by nonlinear volume registration; especially better for subcortical areas 

* FreeSurfer Part (fsaverage space):
[3] extracts white and pial cortical surfaces and transformation into parcels registered using Freesurfers standard folding-based surface registration to their surface atlas (fsaverage)

* PostFreeSurfer Part (164k and 32k_FS_LR space)
[4] Nifti(cifti?)+Gifti files for workbench; along with applying the surface
registration (to the Conte69 surface template (Van Essen et al., 2012b)), downsampling registered surfaces for connectivity analysis, creating the final brain mask, and creating myelin maps

The Pipeline is described in: Glasser, M. F., Sotiropoulos, S. N., Wilson, J. A., Coalson, T. S., Fischl, B., Andersson, J. L., ... & Van Essen, D. C. (2013). The minimal preprocessing pipelines for the Human Connectome Project. Neuroimage, 80, 105-124.


## Conte 69
Reference: Van Essen, D.C., Glasser, M.F., Dierker, D.L., Harwell, J., Coalson, T., 2012b. Parcellations and hemispheric asymmetries of human cerebral cortex analyzed on surface-based atlases. Cerebral Cortex 22, 2241-2262.

There is two versions of this atlas, a hires one using 147 vertices; and a low res one at 32k vertices per hemisphere for everyday usage. As the original atlas cannot be really retrieved anymore from their website since a technical error occured, I downloaded it from:

https://github.com/MidnightScanClub/MSCcodebase/tree/master/Utilities/Conte69_atlas-v2.LR.32k_fs_LR.wb

This LR space supposedly was first developed/described in: Van Essen, David C. “A population-average, landmark-and surface-based (PALS) atlas of human cerebral cortex.” Neuroimage 28.3 (2005): 635-662.




# HCP Data Releases
## Human HCP S900 release (Dec 2015)
## Human HCP initial 7T release (June 2016) releases
## Human HCP S1200 Structural + fMRI Atlas (March 2017; Reference)

This Human reference dataset comprises data from the Human Connectome Project (HCP) 1200 Subjects (S1200) data release; Group-average structural and functional (task) MRI data and  Selected individual-subject anatomical maps for each of the 1096 subjects having at least 1 rfMRI run 

* cites again the Glasser2013 minimal preprocessing pipelines from before
* can be seen as an extension to the conte atlas?
https://balsa.wustl.edu/reference/show/pkXDZ

MNINonLinear/ contains cortical surfaces and other data volumetrically registered to MNI152 space (using nonlinear FNIRT) followed by surface registration to Conte69 ‘164k_fs_LR’ mesh (Van Essen et al. 2012b) (via FreeSurfer fsaverage as an intermediate). *A Connectome Workbench-readable 164k_fs_LR.wb.spec file is included for quickly reading and visualizing many of these files in Workbench in the Human HCP S1200 release*

https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf 

Human.Composite_VGD11.32k_fs_LR.dlabel.nii file is derived from a composite cortical parcellation containing 52 distinct areas accurately mapped to the fs_LR atlas surface and based on architectonic or retinotopic fMRI maps (Van Essen et al. 2012). Abbreviations associated with labels of cortical areas (FRB08, OFP03, etc.) refer to the publication that defined a particular cortical area in the composite map (cf. Table 3 of Van Essen et al. 2012). http://cercor.oxfordjournals.org/content/22/10/2241.long

The Gordon333.32k_fs_LR.dlabel.nii and Gordon333_Key.txt are derived from a cortical parcellation generated using resting state functional connectivity boundary maps to identify putative borders between cortical areas Gordon et al. 2014. https://academic.oup.com/cercor/article-lookup/doi/10.1093/cercor/bhu239

RSN-networks.32k_fs_LR.dlabel.nii file displays resting state network cortical parcellation maps from Yeo et al. 2011 (7 and 17 network maps) and the Resting State network consensus communities (with and without gaps in the communities filled) from Power et al. 2011.


# Online Resource: BALSA (+ visualization assistence files ...)

Reference: DC Van Essen, J Smith, MF Glasser, J Elam, CJ Donahue, DL Dierker, EK Reid, TS Coalson, J Harwell (2016) The Brain Analysis Library of Spatial maps and Atlases (BALSA) Database.  NeuroImage (2016) PMID: 27074495; https://balsa.wustl.edu/study/show/WG33
 
 BALSA Reference is a curated repository of reference data accurately mapped to brain atlas surfaces and volumes, including various types of anatomically and functionally derived spatial maps as well as brain connectivity

Includes files for: 

Four human cortical parcellations on HCP S900 surface (left hemisphere)
Four human cortical parcellations on HCP S900 surface (montage views)
VDG11b 52-surface-mapped cortical areas (annotated)
Brodmann (1909) areas (annotated)
Gordon et al. (2016) areas (annotated)
Yeo et al. (2011) 17-network RSNs (annotated)


# HCP Atlas 

## HCP Multimodal Parcellation v1.0 (MMP1.0), Glasser et al. (Nature, 2016) 

Just describes how the parcellation is done, not which dataset?
>> Using multi-modal magnetic resonance images from the Human Connectome Project (HCP) and an objective semi-automated neuroanatomical approach, we delineated 180 areas per hemisphere bounded by sharp changes in cortical architecture, function, connectivity, and/or topography in a precisely aligned group average of 210 healthy young adults. https://pubmed.ncbi.nlm.nih.gov/27437579/

Usual File Names:
Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii






# Useful Links:

## As of October 2020

NIfTI, CIFTI, GIFTI in the HCP and Workbench: a primer 
http://mvpa.blogspot.com/2014/03/nifti-cifti-gifti-in-hcp-and-workbench.html

Getting started with Connectome Workbench 1.4.2 
https://mvpa.blogspot.com/2020/03/getting-started-with-connectome.html

Connectome Workbench: making a surface version of a volumeric image 
https://mvpa.blogspot.com/2018/02/connectome-workbench-making-surface.html

***Unofficial*** Guide to the HCP surface file formats
https://emmarobinson01.com/2016/02/10/unofficial-guide-to-the-hcp-surface-file-formats/
An important thing to recognise first about the HCP surface file format is that it has two versions of the atlas space: 164k_FS_LR and 32k_FS_LR. These atlases are regularly spaced and represent a left-right symmetric atlas developed  by Washu U in [3]. FS stands for FreeSurfer, and indicates the atlas is related to the FreeSurfer atlas fsaverage.
