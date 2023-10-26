# A non-intrusive reduced order model using deep l    earning for realistic wind data generation for small unmanned aerial systems in urban spaces


Code and tools used for AIP Advances Manuscript [**" A non-intrusive reduced order model using deep learning for realistic wind data generation for small unmanned aerial systems in urban spaces "**](https://doi.org/10.1063/5.0098835)


## How to use
### Input
Please use the input.yaml to change the parameters to run the code

#### Params
The parameters are divided between converting .nc to .mat file, data, runtime, detrending, training, misc parameters

#### Flags
Flags for convertion, runtime, plotting, gif generation, backend plotting are all included in the file.

### Files
* init.py         : File with all the initialisation  
* imp_lib.py      : File to import the necessary libraries 
* functions.py    : File with all the functions used
* main.py         : File to train and run the model
* misc_plotting.py: File with functions included to plot the data



## Packages used
* tensorflow
* sklearn
* h5py
* tqdm
* numpy
* netcdf4
* scipy
* hurst
* matplotlib
* time
* cmocean

## Tested with 
* Python version 3.6 
* Python version 3.8
