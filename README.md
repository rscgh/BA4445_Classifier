# BA4445_Classifier


## Usage:

Before starting using the script, make sure you adjust the paths in both `standalone_classifier.py` as well as `hcp_tools.py` to fit your local system and HCP-dataset version. Normally all important paths should be defined in the beginning.

```
python3

from standalone_classifier import *

for sub in ['103414', '105216']: # do it for all the subjects to be classified
  classify_subject(subid = sub);
```

Other parameters to `classify_subject` are: 
* corrthresh = 0.4, which influences which ICA components are selected as controls and which are considered to be part of intrinsic BA44 or 45 connecitivity (a correlation coefficient typically in range 0-1 is compared against this)
* timeseries, which should be a numpy array only containing data for the left hemisphere and be of shape (29696, n_timepoints) and already smoothed and normalized; if None, which is the default, the 4 resting state series are gathered from the HCP reslease and automatically preprocessed using a wb_command
* save_intermediate, which if set to True will save alot of intermediate files to the output directory (i.e. the 4 smoothed resting state timeseries, ICA components etc)