# BA4445_Classifier


## Usage with HCP-released subject data:

Before starting using the script, make sure you adjust the paths in both `standalone_classifier.py` as well as `hcp_tools.py` to fit your local system and HCP-dataset version. Normally all important paths should be defined in the beginning of the file.

```
python3

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

First you will have to convert the resting state runs of your subjects into the HCP FS32k space. One way to do that is to start out with a BIDS complient dataset that at least needs to include a structural image and a resting state run. In case your dataset is not yet BIDS compliant, you can use tools such as 



The result will be a BIDS compliant dataset in the style of ...


You can then run the standard HCP preprocessing pipeline (in case you do not have T2 files, make use of the legacy version):



The useable resting state runs are now located under:


You can now run the classifier with the following parameters:








