# BA4445_Classifier


## Usage with HCP-released subject data:

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

Files can be viewed on top of standard reference meshs (i.e. S1200 or conte69 both which are aligned to FS_32k space) or the individual subject ones (MNINonlinear/fsaverage_LR32k).

## Usage with other datasets

First you will have to convert the resting state runs of your subjects into the HCP FS32k space. One way to do that is to start out with a BIDS complient dataset that at least needs to include a structural image and a resting state run. 


### Transformation into a bids compliant dataset
In case your dataset is not yet BIDS compliant, you can use tools such as [bidsify]():
`pip3 install bidsify`

```YAML



```

```python
import bidsify
bidsify.bidsify("/data/pt_02189/Data/bidsify_config.yml", "/data/pt_02189/Data/RawDataset", "/data/pt_02189/Data/BidsData", False)

## as bidsify currently still omits creating json files from time to time, we have to create them by hand

import glob, nibabel as nib
rsimgs = glob.glob("/data/pt_02189/Data/BidsData/*/func/*rest*.nii.gz")
for f in rsimgs:
	TR = nib.load(f).get_header().get_zooms()[-1]
	open(f[:-7]+".json", "w").write(r'{ "TaskName":"ShortRestingState", "RepetitionTime": %s }' % (str(TR)) )

t1imgs = glob.glob("/data/pt_02189/Data/BidsData/*/anat/*T1*.nii.gz")
for f in t1imgs: open(f[:-7]+".json", "w").write(r'{"DwellTime":"irrelevant"}')

```

Furthermore you might have to modify the `dataset.json`, such that the funding becomes an array:

The result will be a BIDS compliant dataset in the style of ...


### Transformation into LR32k space using the HCP preprocessing pipeline

You can then run the standard HCP preprocessing pipeline (in case you do not have T2 files, make use of the legacy version; execution of a single subject might take around 20h; this example makes use of the bids/hcppipelines app and furthermore uses singularity for container management):

```
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

# remove (if present):
subj_list_path = '/data/pt_02189/MARHCP/BrocaConn/subject_list_101.txt'
subs = np.loadtxt(subj_list_path, dtype=str)  # should contain individual rows of strings like '100307'
subs = [sub for sub in subs if sub != '111009'] # this subject is not existing in the HCP500 distribution

```

You can now run the classifier with the following parameters (classification only takes around 1-2mins):

```python
from standalone_classifier_new import *
classify_subject(subid = "sub-0001Dirk", template="MNINonLinear/Results/task-rest00_acq-huhu_bold/task-rest*_bold_Atlas.dtseries.nii", cnt_files=1);
```








