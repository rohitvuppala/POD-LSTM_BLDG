#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 16:44:42 2021

author: Rohit K S S Vuppala
         Graduate Student, 
         Mechanical and Aerospace Engineering,
         Oklahoma State University.

@email: rvuppal@okstate.edu

"""

#Some things to run init
mp.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
plt.style.use('seaborn-whitegrid')
colors = pylab.cm.get_cmap('coolwarm', 10)

os.system('rm -rf logs/')

#make folder for plots
print("-------------------------Checking for Folder--------------------------")
if (os.path.isdir('plots')):
    print('Folder exists')
    os.system('rm -rf plots/*')
    print('Removed existing folder contents')
else:
    print('Folder doesnot exist')
    os.system('mkdir plots')
    print('Created the folder')

#%%
"""
Function to read the input file
"""

print_status("Reading Input File")
import yaml
with open('input.yaml') as file:
    input_data = yaml.load(file, Loader=yaml.FullLoader)
file.close()

itrain         = input_data['itrain']
idata_in       = input_data['idata_in']
iric_plot      = input_data['iric_plot']
iric_lim       = input_data['iric_lim']
itrumodes_plot = input_data['itrumodes_plot']
irecon_nc      = input_data['irecon_nc']
il2norm        = input_data['il2norm'] 
ihurst         = input_data['ihurst']

inc2mat = input_data['inc2mat']
nc2mat_fname_nc  = input_data['nc2mat_fname_nc']
nc2mat_fname_mat = input_data['nc2mat_fname_mat']
nc2mat_ntot      = input_data['nc2mat_ntot']
nc2mat_start      = input_data['nc2mat_start']

data_fname = input_data['data_fname']
data_ntot  = input_data['data_ntot']

n_offset= input_data['n_offset']
n_train = input_data['n_train']
n_pred  = input_data['n_pred'] 
n_modes = input_data['n_modes']
ric_lim = input_data['ric_lim']

n_trumodes_plot   = input_data['n_trumodes_plot']
nrow_trumodes_plot= input_data['nrow_trumodes_plot']
ncol_trumodes_plot= input_data['ncol_trumodes_plot']

n_lookback = input_data['n_lookback']
n_epochs   = input_data['n_epochs']
n_batchsize= input_data['n_batchsize']
n_neurons  = input_data['n_neurons']
n_hidlayers= input_data['n_hidlayers'] 
rec_dropout= input_data['rec_dropout']
dropout    = input_data['dropout']
model_fname= input_data['model_fname']

i_detrend    = input_data['i_detrend']
nord_detrend = input_data['nord_detrend']
 
n_pages  = input_data['n_pages']
n_perpage= input_data['n_perpage']
nrow_mlmodes_plot= input_data['nrow_mlmodes_plot']
ncol_mlmodes_plot= input_data['ncol_mlmodes_plot']

n_plot_start           = input_data['n_plot_start']
n_plot_step           = input_data['n_plot_step']
n_plot_num           = input_data['n_plot_num']

t_start = input_data['t_start']
t_end   = input_data['t_end']

igif_tru       = input_data['igif_tru']
nstart_gif_tru = input_data['nstart_gif_tru']
nstep_gif_tru  = input_data['nstep_gif_tru']

igif_mlorg     = input_data['igif_mlorg']
nstep_mlorg    = input_data['nstep_mlorg']

imp_backend = input_data['imp_backend']
mp_backend = input_data['mp_backend']
cm         = input_data['cmap']
print_status("Completed Reading Input File")