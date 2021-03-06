# BA4445_Classifier

Reimplementation of the classifier developed by [Jakobsen et al., 2018](https://www.sciencedirect.com/science/article/pii/S1053811916305468) using a spatial prior for BA44 and 45 plus region-stereotyped connectivity profiles based on alot of manual annotaions (100 subjects, see [Jakobsen et al, 2016](https://www.researchgate.net/profile/Rudolf_Ruebsamen/publication/284888318_Subdivision_of_Broca's_region_based_on_individual-level_functional_connectivity/links/5cff5bd192851c874c5d9ff6/Subdivision-of-Brocas-region-based-on-individual-level-functional-connectivity.pdf)) based on morphology and known differential connectivity (only connectivity of the left hemisphere is regarded)

The standalone classifier uses the reference data generated by the do_all.py. This data is already pregenerated and available in the res directory and hence to use the standalone classifer, it is not nessesary to download the HCP subjects, nor execute the do_all.py.



## Usage with HCP-released subject data / with resting state runs in fs_LR32k format:

Before starting using the script, make sure you adjust the paths in both `standalone_classifier.py` as well as `hcp_tools.py` to fit your local system and HCP-dataset version. Normally all important paths should be defined in the beginning of the file.

```python
# python3

from standalone_classifier import *

for sub in ['103414', '105216']: # do it for all the subjects to be classified
  classify_subject(subid = sub);
```

Other parameters to `classify_subject` are: 
* *corrthresh* = 0.4, which influences which ICA components are selected as controls and which are considered to be part of intrinsic BA44 or 45 connecitivity (a correlation coefficient typically in range 0-1 is compared against this)
* *timeseries*, which should be a numpy array only containing data for the left hemisphere and be of shape (29696, n_timepoints) and already smoothed and normalized; if None, which is the default, the 4 resting state series are gathered from the HCP reslease and automatically preprocessed using a wb_command
* *save_intermediate*, which if set to True will save alot of intermediate files to the output directory (i.e. the 4 smoothed resting state timeseries, ICA components etc)
* *template*, that is the glob-match string to the resting state run(s), relative to the subjects folder, i.e. `MNINonLinear/Results/rfMRI_REST?_??/rfMRI_REST?_??_Atlas_hp2000_clean.dtseries.nii`
* *cnt_files* = 4, defines the number of resting state runs to be used, in case the template/selector matches multiple files

The example file in output_example also includes Estrids previous automatic and manual annotation. This is not part of the standalone_classifier and hence wont be contained in its output. But it should be present in do_all_v2.py

Files can be viewed on top of standard reference meshs (i.e. S1200 or conte69 both which are aligned to FS_32k space) or the individual subject ones (MNINonlinear/fsaverage_LR32k) using i.e. the [Connectome Workbench](https://humanconnectome.org/software/connectome-workbench)

## Usage with other datasets

First you will have to convert the resting state runs of your subjects into the HCP FS32k space. One way to do that is to start out with a volumetric dataset that at least needs to include a structural image and a resting state run. Prior to classification this would need to be processed as follows

* restructuring into a [bids](https://bids.neuroimaging.io/) complient format
* extraction of the surface and conversion into the HCP fs_LR32k format


### Transformation into a bids compliant dataset

You might start out with a dataset looking like:

```
/data/pt_02189/Data/RawDataset/...
   sub_0001Dirk/rs.nii.gz
               /t1.nii.gz
   ...
```

As it is not yet BIDS compliant, you can use tools such as [bidsify](https://github.com/NILAB-UvA/bidsify) to make it:

`pip3 install bidsify`

You will need to create a `bidsify_config.yml` file to define the mappings:

```YAML

options:
    mri_ext: nifti  # alternatives: nifti/dcm/DICOM
    debug: True  # alternative: True, prints out a lot of stuff
    n_cores: -1  # number of CPU cores to use (for some operations)
    subject_stem: sub_ # subject identifier
    deface: True  # whether to deface structural scans
    spinoza_data: False  # only relevant for data acquired at the Spinoza Centre

mappings:
    bold: rs
    T1w: t1
    
anat:
    anatT1:  # this name doesn't matter
        id: t1  # identifier to this type of scan

func:
    metadata:
      RepetitionTime: 3.2
      TaskName: ShortRestingState
    
    rs01:
        id: rs.nii.gz
        task: rest01
```

Then you can run bidsify from the command line or from within python:

```python

## actual bidsify procedure

import bidsify
bidsify.bidsify("/data/pt_02189/Data/bidsify_config.yml", "/data/pt_02189/Data/RawDataset", "/data/pt_02189/Data/BidsData", False)

## custom: as bidsify currently still omits creating json files from time to time, we have to create them by hand

import glob, nibabel as nib
rsimgs = glob.glob("/data/pt_02189/Data/BidsData/*/func/*rest*.nii.gz")
for f in rsimgs:
	TR = nib.load(f).get_header().get_zooms()[-1]
	open(f[:-7]+".json", "w").write(r'{ "TaskName":"ShortRestingState", "RepetitionTime": %s }' % (str(TR)) )

t1imgs = glob.glob("/data/pt_02189/Data/BidsData/*/anat/*T1*.nii.gz")
for f in t1imgs: open(f[:-7]+".json", "w").write(r'{"DwellTime":"irrelevant"}')

```

Furthermore you might have to modify the `dataset_description.json`, such that the funding becomes an array:

```
"Funding": "Put your funding sources here" 	<- wrong wrong
"Funding": ["Put your funding sources here"]	<- correct
```

The final dataset structure looks like:#
```
/data/pt_02189/Data/BidsData/...
  participants.tsv
  dataset_description.json
  sub-0001Dirk/
     anat/sub-0001Dirk_T1w.nii.gz
          sub-0001Dirk_T1w.json
     func/sub-0001Dirk_task-rest01_bold.json
          sub-0001Dirk_task-rest01_bold.json
  ...
```  

You can now valudate your dataset using a [validator](https://github.com/INCF/bids-validator) (this one works within the browser and does not seem to upload data).


### Transformation into LR32k space using the HCP preprocessing pipeline

You can then run the standard HCP preprocessing pipeline (in case you do not have T2 files, make use of the legacy version; execution of a single subject might take around 20h; this example makes use of the bids/hcppipelines app and furthermore uses singularity for container management):

```
# singularity commands may be different for your local setup (?)

# download the bids hcppipelines app as docker container and convert to singularity
singularity build /data/pt_02189/Data/bids_hcppipelines.img docker://bids/hcppipelines

# run the pipeline on our subject dir
## abstract command: singularity run -B folder_to_be_available container_image input_dir output_dir analysis_level [further params and flags]
##
## make the bids input and hcppipeline output directory available to the container using the -B 
## license key is invalid here, get your own ... (its only the part after the asterix \* in the license key file until the line break)
##
singularity run -i \
-B /data/pt_02189/Data/BidsData/ -B /data/pt_02189/Data/HcpData \
/data/pt_0  2189/Data/bids_hcppipelines.img \
/data/pt_02189/Data/BidsData /data/pt_02189/Data/HcpData participant --license_key C7MQIxYdAAwI --processing-mode legacy --participant_label 0001Dirk
```

The useable resting state runs are now located under:

`/data/pt_02189/Data/HcpData/sub-0001Dirk/MNINonLinear/Results/task-rest00_acq-huhu_bold/task-rest*_bold_Atlas.dtseries.nii`

### Running the classification

Now copy the `standalone_classifier.py` as `standalone_classifier_new.py` and edit the paths in there:
```python
hcp_all_path = '/data/pt_02189/Data/HcpData'
out_folder_p = '/data/pt_02189/Data/HcpData_classifyBA4445'    # make sure this folder exisits
```

You can now run the classifier with the following parameters (classification only takes around 1-2mins):

```python
from standalone_classifier_new import *
classify_subject(subid = "sub-0001Dirk", template="MNINonLinear/Results/task-rest00_bold/task-rest*_bold_Atlas.dtseries.nii", cnt_files=1);
```

The result will be:

`/data/pt_02189/Data/HcpData_classifyBA4445/AutoAreaLabelFinal_HCP_indv_sub-0001Dirk_LH29k_WTA44_45.dscalar`




## References

[Jakobsen, E., Böttger, J., Bellec, P., Geyer, S., Rübsamen, R., Petrides, M., & Margulies, D. S. (2016). Subdivision of Broca's region based on individual‐level functional connectivity. European Journal of Neuroscience, 43(4), 561-571.](https://www.researchgate.net/profile/Rudolf_Ruebsamen/publication/284888318_Subdivision_of_Broca's_region_based_on_individual-level_functional_connectivity/links/5cff5bd192851c874c5d9ff6/Subdivision-of-Brocas-region-based-on-individual-level-functional-connectivity.pdf)

[Jakobsen, E., Liem, F., Klados, M. A., Bayrak, Ş., Petrides, M., & Margulies, D. S. (2018). Automated individual-level parcellation of Broca's region based on functional connectivity. NeuroImage, 170, 41-53.](https://www.sciencedirect.com/science/article/pii/S1053811916305468)



