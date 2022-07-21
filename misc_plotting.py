#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:42:23 2021

author: Rohit K S S Vuppala
         Graduate Student, 
         Mechanical and Aerospace Engineering,
         Oklahoma State University.

@email: rvuppal@okstate.edu

"""
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.io import savemat,loadmat
import os
from tqdm import tqdm
from hurst import compute_Hc
from functions import *
#%%
"""
Function to print status
"""
def print_status(string):
    print()
    print('---------------------------------------')
    print(string)
    print('---------------------------------------')
    print()    
#%%
"""
Function for calculating the L2norm between ROM and Recon values
"""
def calc_l2norm_rom(u,u_true,n_pred,n_train,nx,ny,nz):
    
    l2_norm_diff = np.zeros(n_pred)
    for i in range(n_pred):
        diff = u_true[:,i].reshape(nz,ny,nx) - u[:,i].reshape(nz,ny,nx)  
        l2_norm_diff[i] = np.linalg.norm(diff)/np.sqrt(diff.shape[0])
    
    l2_max = np.max(l2_norm_diff[:n_train])
    l2_norm_rel = l2_norm_diff/l2_max 
    print_status("Max L2norm between Recon and ML-ROM data for given modes= "+str(np.max(l2_norm_diff)))
    
    
    plt.figure(figsize=(10,5))
    plt.title("Relative L2 error between ML-ROM and Reconstructed data",fontsize=14)
    plt.xlabel("Snapshot number",fontsize=12)
    plt.ylabel("Relative L2 Error",fontsize=12)
    plt.xlim(0,n_pred)
    plt.tight_layout()
    plt.axvspan(0, n_train, alpha=0.2, color='green')
    plt.axvspan(n_train,n_pred,alpha=0.2, color='red')
    plt.plot(l2_norm_rel,'k')
    plt.savefig('plots/l2norm_rom-recon.png',dpi=300)
    plt.show()

    return l2_norm_diff,l2_norm_rel
#%%
"""
Function for writing .mat files if necessary
"""
def nc2mat(nc2mat_fname_nc,nc2mat_fname_mat,nc2mat_ntot,nc2mat_start):
    file= nc2mat_fname_nc
    fh = Dataset(file, mode='r')
    
    nstart= nc2mat_start
    ntot  = nc2mat_ntot
    nend  = nstart + nc2mat_ntot #fh.variables["time"].size          # Number of time stamps of data
    
    u_org  = np.array(fh.variables["u"][nstart:nend,0:20,10:50,0:60])
    v_org  = np.array(fh.variables["v"][nstart:nend,0:20,10:50,0:60])
    w_org  = np.array(fh.variables["w"][nstart:nend,0:20,10:50,0:60])
    
    x  = np.array(fh.variables["x"][0:60])
    x  = x.reshape(x.shape[0],1)
    y  = np.array(fh.variables["y"][10:50])
    y  = y.reshape(y.shape[0],1)
    z  = np.array(fh.variables["zu_3d"][0:20])
    z  = z.reshape(z.shape[0],1)
    
    
    fh.close()
    
    u = np.reshape(u_org,[ntot,-1]).T
    v = np.reshape(v_org,[ntot,-1]).T
    w = np.reshape(w_org,[ntot,-1]).T
    
    arr = {'x':x,'y':y,'z':z,'u':u,'v':v,'w':w}
    savemat(nc2mat_fname_mat,arr)
    
    del u_org,v_org,w_org,u,v,w,x,y,z    
    
    return None
#%%
"""
Function to load data from file
"""

def load_data(idata_in,data_ntot,n_offset,data_fname):
    if (idata_in==0):

        print_status("Loading Data from .nc file given")

        file= data_fname
        fh = Dataset(file, mode='r')
        
        ntot = data_ntot #fh.variables["time"].size          # Number of time stamps of data
        
        u_org  = np.array(fh.variables["uinterp"][n_offset:ntot+n_offset,:,:,:])
        v_org  = np.array(fh.variables["vinterp"][n_offset:ntot+n_offset,:,:,:])
        w_org  = np.array(fh.variables["winterp"][n_offset:ntot+n_offset,:,:,:])
        
        x  = np.array(fh.variables["xh"][:])
        x  = x.reshape(x.shape[0],1)
        y  = np.array(fh.variables["yh"][:])
        y  = y.reshape(y.shape[0],1)
        z  = np.array(fh.variables["zh"][:])
        z  = z.reshape(z.shape[0],1)
        
        
        fh.close()
        
        u = np.reshape(u_org,[ntot,-1]).T
        #v = np.reshape(v_org,[ntot,-1]).T
        #w = np.reshape(w_org,[ntot,-1]).T
        
        nx,ny,nz = x.shape[0],y.shape[0],z.shape[0]
        del u_org,v_org,w_org,fh

    elif(idata_in==1):

        print_status("Loading Data from .mat file given")
     
        file = loadmat(data_fname)
        
        ntot = data_ntot 
        u = file['u'][:,n_offset:ntot+n_offset]
        #v = file['v'][:,:ntot]
        #w = file['w'][:,:ntot]
        x = file['x']
        y = file['y']
        z = file['z']
        
        nx = x.shape[0]
        ny = y.shape[0]
        nz = z.shape[0]
        
        del file
        print_status("Loaded Data from .mat file given")

    
    else:
        print_status("E: Error incorrect idata_in chosen")

        
    return (u,x,y,z,nx,ny,nz)
#%%
"""
Function to plot RIC
"""
def plot_ric(n_train,nr,S2_nr_ric,RIC):
    plt.figure()
    gspace = np.linspace(1,n_train,n_train)
    fig, axs = plt.subplots(1, 1, figsize=(7,5))#, constrained_layout=True)
    axs.loglog(gspace,S2_nr_ric, lw = 1, marker="o", linestyle='--', color='k',label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
    axs.set_xlim([1,n_train])
    #axs.axvspan(0, nr, alpha=0.4, color='green')
    #axs.axvspan(nr,n_train,alpha=0.4, color='red')
    axs.axhspan(RIC,100,alpha=0.4, color='red')
    axs.set_xlabel('$nr$-Number of modes',fontsize=14)
    axs.set_ylabel('$RIC$-Relative Information Content',fontsize=14)
    
    #axs.text(nr,RIC-10,'Cut-off mode')
    label = 'Cutoff-mode'
    plt.annotate(f"$\\bf{label}$",
      xy=(nr, RIC), xytext=(40, -40),
      textcoords='offset points', ha='center', va='bottom',
      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.fill_between([1,nr], [RIC,RIC],alpha=0.4, color='lime')
    plt.fill_between([nr,400], [RIC,RIC],alpha=0.4, color='red')
    
    axs.plot(nr, RIC, marker="*", markersize=20, markeredgecolor="black", markerfacecolor="black")
    plt.title("Number of modes taken= '{0}' , RIC= '{1}'" .format(int(nr),round(RIC,2)),fontsize=16)
    fig.tight_layout()
     
    axs.set_ylim(0,100)
    axs.set_xlim(0,400)
    fig.savefig('plots/ric.png', dpi=300)
    plt.show()
    #plt.close()
    #plt.close('all')
    
    return None
#%%
"""
Function to plot true modes
"""
def plot_trumodes(n_train,n_trumodes_plot,nrow_trumodes_plot,ncol_trumodes_plot,at):
    plt.figure()
    gspace_2 = np.linspace(1,n_train,n_train)
    
    
    fig, ax = plt.subplots(nrows=nrow_trumodes_plot,ncols=ncol_trumodes_plot,figsize=(12,8),sharex=True)
    ax = ax.flat
    nrs = n_trumodes_plot
    
    for i in range(nrs):
        ax[i].plot(gspace_2,at[:,i],'k',label=r'True Values')
        ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
        ax[-1].set_xlim([gspace_2[0],gspace_2[-1]])
    
    ax[-2].set_xlabel(r'$t$',fontsize=14)    
    ax[-1].set_xlabel(r'$t$',fontsize=14)    
    fig.tight_layout(rect=[0.,0.,1,0.95])
    
    fig.subplots_adjust(bottom=0.1)
    line_labels = ["True"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
    plt.suptitle("True Modes 1-"+str(nrs))
    
    fig.savefig('plots/true_modes.png', dpi=300)
    plt.show()
    
    
    del gspace_2,fig,ax
    
    #plt.close()
    #plt.close('all')
    return None
#%%
"""
Function to plot detrended true modes
"""
def plot_dtmodes(n_train,n_trumodes_plot,nrow_trumodes_plot,ncol_trumodes_plot,at_detrend):
    plt.figure()
    gspace_2 = np.linspace(1,n_train,n_train)
    
    fig, ax = plt.subplots(nrows=nrow_trumodes_plot,ncols=ncol_trumodes_plot,figsize=(12,8),sharex=True)
    ax = ax.flat
    nrs = n_trumodes_plot
    
    for i in range(nrs):
        ax[i].plot(gspace_2,at_detrend[:,i],'k',label=r'True Values')
        ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
        ax[-1].set_xlim([gspace_2[0],gspace_2[-1]])
    
    ax[-2].set_xlabel(r'$t$',fontsize=14)    
    ax[-1].set_xlabel(r'$t$',fontsize=14)    
    fig.tight_layout(rect=[0.,0.,1,0.95])
    
    fig.subplots_adjust(bottom=0.1)
    line_labels = ["True"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
    plt.suptitle("Detrended True Modes 1-"+str(nrs))
    
    fig.savefig('plots/true_modes_detrend.png', dpi=300)
    plt.show()
    
    
    del gspace_2,fig,ax
    return None
#%%
"""
Function to plot first snapshot
"""
def plot_nthsnapshot_org(u,u_true,x,y,z,nx,ny,nz,nplot,cm,idx,pos):
    plt.figure()
    
    #cmap = copy.copy(mp.cm.get_cmap("jet"))
    #cmap.set_bad(color='black',alpha=1.0)
    
    #Plot them  at the first snapshot
    X,Y = np.meshgrid(x,z)
    fig,axs = plt.subplots(1,2,figsize=(12,5))
    
    u_3d      = np.copy(u[:,nplot].reshape(nz,ny,nx))
    u_true_3d = np.copy(u_true[:,nplot].reshape(nz,ny,nx))
    
    
    
    levels = np.linspace(np.nanmin(u_3d),np.nanmax(u_3d),120)
    mid    = idx#int(ny/2)
    
    #Nan values for building
    u_3d[0:10,mid,20:30] = np.nan
    u_true_3d[0:10,mid,20:30] = np.nan
    
    #Plot from the data 
    axs[0].set_title("From Original Data")
    axs[0].set_xlabel("x location in metres",fontsize=12)
    axs[0].set_ylabel("z location in metres",fontsize=12)
    cs = axs[0].contourf(X,Y,u_3d[:,mid,:],levels=levels,cmap=cm,extend='both')
    cs1 = axs[0].contour(X,Y,u_3d[:,mid,:],levels=[-2,-1.5,-1,-0.5,0,2,4,6,8,9],colors='k',extend='both')
    cbar = fig.colorbar(cs, ax=axs[0], orientation='vertical',shrink=1)
    cbar.ax.set_title('u-vel',fontsize=12)
    plt.clabel(cs1,inline=True,fontsize=12)
    
    #Plot from reconstruction
    axs[1].set_title("From Reconstruction")
    axs[1].set_xlabel("x location in metres",fontsize=12)
    axs[1].set_ylabel("z location in metres",fontsize=12)
    cs = axs[1].contourf(X,Y,u_true_3d[:,mid,:],levels=levels,cmap=cm,extend='both')
    cs1 = axs[1].contour(X,Y,u_true_3d[:,mid,:],levels=[-2,-1.5,-1,-0.5,0,2,4,6,8,9],colors='k',extend='both')
    cbar = fig.colorbar(cs, ax=axs[1], orientation='vertical',shrink=1)
    cbar.ax.set_title('u-vel',fontsize=12)
    plt.clabel(cs1,inline=True,fontsize=12)
    
    fig.tight_layout(rect=[0, 0., 1, 0.95])
    plt.suptitle("u-vel contour comparision at "+pos+" for first snapshot Data and Reconstructed-Data")
    
    fig.savefig('plots/first_snapshot_'+pos+'.png',dpi=300)
    plt.show()    
    
    #plt.close()
    
    #plt.close('all')
    return None
#%%
"""
Function to plot ML training 
"""
def plot_training(model_fname,model,history):
    fname= model_fname
    model.save(fname)
    # Plot the training characteristics
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    avg_mae = history.history['coeff_metric']
    val_avg_mae = history.history['val_coeff_metric']
    epochs = range(1, len(loss) + 1)
    
    plt.figure()
    plt.semilogy(epochs, loss, 'b', label='Training loss')
    plt.semilogy(epochs, val_loss, 'r', label='Validationloss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('plots/loss_train&val.png', dpi=300)
    plt.show()
   
    #plt.close() 
    
    plt.figure()
    plt.semilogy(epochs, avg_mae, 'b', label=f'Average $R_2$')
    plt.semilogy(epochs, val_avg_mae, 'r', label=f'Validation Average $R_2$')
    plt.title('Evaluation metric')
    plt.legend()
    plt.savefig('plots/metric_train&val.png')
    plt.show()
    
    #plt.close()
     
    #plt.close('all')
    return None
#%%
"""
Function to plot ML and Org after prediction
"""
def plot_orgmlmodes(n_pages,n_perpage,nrow_mlmodes_plot,ncol_mlmodes_plot,n_pred,n_train,at2,ytest_ml,fname,tname):
    for j in range(n_pages):
    
        plt.figure()
        fig, ax = plt.subplots(nrows=nrow_mlmodes_plot,ncols=ncol_mlmodes_plot,figsize=(12,8),sharex=True)
        ax = ax.flat
        nrs = at2.shape[1]
        t = np.linspace(1,n_pred,n_pred)
        
        for i in range(n_perpage):
            ax[i].plot(t,at2[:,i+j*8],'k',label=r'True Values')
            ax[i].plot(t,ytest_ml[:,i+j*8],'b-',label=r'ML ')
            ax[i].set_xlabel(r'$t$',fontsize=14)
            ax[i].set_ylabel(r'$a_{'+str(i+1+j*8) +'}(t)$',fontsize=14)    
            ax[-1].set_xlim([t[0],t[-1]])
            ax[i].axvspan(0, t[n_train-1], alpha=0.2, color='darkorange')
            ax[i].axvspan(t[n_train], t[-1], alpha=0.2, color='green')
        
        ax[-2].set_xlabel(r'$t$',fontsize=14)    
        ax[-1].set_xlabel(r'$t$',fontsize=14)    
        
        
        fig.subplots_adjust(bottom=0.1)
        line_labels = ["True", "ML"]#, "ML-Test"]
        plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0., fontsize=14)
        #plt.suptitle(tname,fontsize=24) 
        #fig.tight_layout()
        fig.savefig('plots/'+str(fname)+str(j+1) +'.png', dpi=200)
        plt.show()
        
        
        #plt.close()
    del fig,ax,t 
    #plt.close('all')
    return None
#%%
"""
Function to write recon files
"""
def write_recon(u_ml,u_recon,x,y,z,nx,ny,nz,t_start,t_end,n_pred):
    print_status("Writing Results to NC File")
    
    os.system('rm -rf recon/')
    try:
        os.system('mkdir recon')
    except:
        print('folder exists')
        os.system('rm -rf recon/*')
    
    file = Dataset('recon/recon.nc','w',format='NETCDF4_CLASSIC')
    # Create dimensions:
    x_dim = file.createDimension('x', nx)
    y_dim = file.createDimension('y', ny)
    z_dim = file.createDimension('z', nz)
    t_dim = file.createDimension('time',None)
        
         
    xnc = file.createVariable('x', np.float32, ('x',))
    xnc.axis = 'X'
    xnc.units= 'km'
    ync = file.createVariable('y', np.float32, ('y',))
    ync.axis = 'Y'
    ync.units= 'km'
    znc = file.createVariable('z', np.float32, ('z',))
    znc.axis = 'Z'
    znc.units= 'km'
    tnc = file.createVariable('time', np.float64, ('time',))
    tnc.units= 'sec'                                   
            
    # transfer the coordinate variables:
    xnc[:] = x.flatten().tolist()
    ync[:] = y.flatten().tolist()
    znc[:] = z.flatten().tolist()
    
    unc  = file.createVariable('u_ml', np.float64 , ('time','z','y','x'))
    unc1 = file.createVariable('u_recon', np.float64 , ('time','z','y','x'))
    
    
    for i in tqdm(range(n_pred)):
        u_slice = u_ml[:,i]
        u_slice1 = u_recon[:,i]
        #v_slice = v[:,0]
        #w_slice = w[:,0]

        # Reshape for 2D
        u_data = np.reshape(u_slice,(nz,ny,nx))
        u_data1 = np.reshape(u_slice1,(nz,ny,nx))

        # transfer the data variables:
        unc[i,:,:,:] = u_data
        unc1[i,:,:,:] = u_data1
        
        #set -9999 in building position
        unc[i,0:10,15:25,20:30] = -9999
        unc1[i,0:10,15:25,20:30] = -9999
        
        #file.close()

    dt = (t_end - t_start)/(n_pred)    
    tnc[:] = t_start + np.arange(n_pred) + dt   

    file.close()
    return None
#%%
"""
Function to plot nth snapshot in xy
"""
def plot_nthsnapshot_xy(u,u_ml,x,y,z,nx,ny,nz,n_plot,cm,idx,pos):
    
    #Plot them
    plt.figure() 
    X,Y = np.meshgrid(x,y)
    
    #n_plot = 359
    fig,axs = plt.subplots(1,2, figsize=(12,5))
    
    current_cmap = plt.cm.get_cmap('hot')
    current_cmap.set_bad(color='white',alpha=1.0)
    
    
    u_3d_org = np.copy(u[:,n_plot-1].reshape(nz,ny,nx))
    u_3d_ml  = np.copy(u_ml[:,n_plot-1].reshape(nz,ny,nx))
    
    levels = np.linspace(np.min(u_3d_org),np.max(u_3d_org),120)
    mid    = idx
    
    #Nan values for building
    u_3d_org[0:10,15:25,20:30] = np.nan
    u_3d_ml[0:10,15:25,20:30] = np.nan
    
    #Plot from the data 
    axs[0].set_title("From Reconstructed Data")
    axs[0].set_xlabel("x location in metres",fontsize=12)
    axs[0].set_ylabel("y location in metres",fontsize=12)
    cs = axs[0].contourf(X,Y,u_3d_org[mid,:,:],levels,cmap=cm,extend='both')
    cs1 = axs[0].contour(X,Y,u_3d_org[mid,:,:],levels=[-2,-1.5,-1,-0.5,0,2,4,6,8,9],colors='k',extend='both')
    cbar=fig.colorbar(cs, ax=axs[0], orientation='vertical',shrink=1.0)
    cbar.ax.set_title('u-vel',fontsize=12)
    plt.clabel(cs1,inline=True)
    
    #Plot from ML
    axs[1].set_title("From ROM-LSTM Prediction")
    axs[1].set_xlabel("x location in metres",fontsize=12)
    axs[1].set_ylabel("y location in metres",fontsize=12)
    cs = axs[1].contourf(X,Y,u_3d_ml[mid,:,:],levels,cmap=cm,extend='both')
    cs1 = axs[1].contour(X,Y,u_3d_ml[mid,:,:],levels=[-2,-1.5,-1,-0.5,0,2,4,6,8,9],colors='k',extend='both')
    cbar=fig.colorbar(cs, ax=axs[1], orientation='vertical',shrink=1)
    cbar.ax.set_title('u-vel',fontsize=12)
    plt.clabel(cs1,inline=True)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("u-velocity contour comparison for "+str(n_plot)+"th snapshot at "+pos)
    plt.savefig("plots/"+str(n_plot)+"_snapshotxy_"+pos+".png",dpi=300)
    plt.show() 
    
    #plt.close()
    #plt.close('all')
    return None
#%%
"""
Function to plot nth snapshot
"""
def plot_nthsnapshot(u,u_ml,x,y,z,nx,ny,nz,n_plot,cm,idx,pos):
    
    #Plot them
    plt.figure() 
    X,Y = np.meshgrid(x,z)
    
    #n_plot = 359
    fig,axs = plt.subplots(1,2, figsize=(12,5))
    
    current_cmap = plt.cm.get_cmap('hot')
    current_cmap.set_bad(color='white',alpha=1.0)
    
    
    u_3d_org = np.copy(u[:,n_plot-1].reshape(nz,ny,nx))
    u_3d_ml  = np.copy(u_ml[:,n_plot-1].reshape(nz,ny,nx))
    
    levels = np.linspace(np.min(u_3d_org),np.max(u_3d_org),120)
    mid    = idx
    
    #Nan values for building
    u_3d_org[0:10,mid,20:30] = np.nan
    u_3d_ml[0:10,mid,20:30] = np.nan
    
    #Plot from the data 
    axs[0].set_title("From Reconstructed Data")
    axs[0].set_xlabel("x location in metres",fontsize=12)
    axs[0].set_ylabel("z location in metres",fontsize=12)
    cs = axs[0].contourf(X,Y,u_3d_org[:,mid,:],levels,cmap=cm,extend='both')
    cs1 = axs[0].contour(X,Y,u_3d_org[:,mid,:],levels=[-2,-1.5,-1,-0.5,0,2,4,6,8,9],colors='k',extend='both')
    cbar=fig.colorbar(cs, ax=axs[0], orientation='vertical',shrink=1.0)
    cbar.ax.set_title('u-vel',fontsize=12)
    plt.clabel(cs1,inline=True)
    
    #Plot from ML
    axs[1].set_title("From ROM-LSTM Prediction")
    axs[1].set_xlabel("x location in metres",fontsize=12)
    axs[1].set_ylabel("z location in metres",fontsize=12)
    cs = axs[1].contourf(X,Y,u_3d_ml[:,mid,:],levels,cmap=cm,extend='both')
    cs1 = axs[1].contour(X,Y,u_3d_ml[:,mid,:],levels=[-2,-1.5,-1,-0.5,0,2,4,6,8,9],colors='k',extend='both')
    cbar=fig.colorbar(cs, ax=axs[1], orientation='vertical',shrink=1)
    cbar.ax.set_title('u-vel',fontsize=12)
    plt.clabel(cs1,inline=True)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("u-velocity contour comparison for "+str(n_plot)+"th snapshot at "+pos)
    plt.savefig("plots/"+str(n_plot)+"_snapshot_"+pos+".png",dpi=300)
    plt.show() 
    
    #plt.close()
    #plt.close('all')
    return None
#%%
"""
Function to plot and save gif for trumodes
"""
def plot_giftru(nstart_gif_tru,nstep_gif_tru,nrow_trumodes_plot,ncol_trumodes_plot,n_train,u_variation_test,PHIw):
    os.system('rm -rf plots/giftru')
    os.system('mkdir plots/giftru')    
    plt.figure()
    gspace_out = np.linspace(1,n_train,n_train)
    at = get_coeff(u_variation_test, PHIw)
    
    fig, ax = plt.subplots(nrows=nrow_trumodes_plot,ncols=ncol_trumodes_plot,figsize=(12,8),sharex=True)
    ax = ax.flat
    nrs = nrow_trumodes_plot*ncol_trumodes_plot
    fig.tight_layout(rect=[0.,0.,1,0.95])
    for j in tqdm(range(nstart_gif_tru,n_train+1,nstep_gif_tru)):
        gspace_2 = np.linspace(1,j,j)
        for i in range(nrs):
            ax[i].plot(gspace_2,at[:j,i],'k',label=r'True Values')
            ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
            ax[-1].set_xlim([gspace_out[0],gspace_out[-1]])
        
        ax[-2].set_xlabel(r'$t$',fontsize=14)    
        ax[-1].set_xlabel(r'$t$',fontsize=14)    
        
        
        fig.subplots_adjust(bottom=0.1)
        line_labels = ["True"]
        plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
        plt.suptitle("True Modes 1-"+str(nrs))
         
       
        fig.savefig('plots/giftru/true_modes.'+str(int((j-nstart_gif_tru)/nstep_gif_tru))+'.png', dpi=300)
        plt.show()
        #plt.close()
    del fig,ax,gspace_out,gspace_2
#%%
"""
Function to plot and save gif for ml and true modes
"""
def plot_gifmlorg(n_pages,n_perpage,nstep_mlorg,nrow_mlmodes_plot,ncol_mlmodes_plot,n_train,n_pred,at2,ytest_ml):
    os.system('rm -rf plots/gifmlorg')
    os.system('mkdir plots/gifmlorg')
    for k in tqdm(range(n_train,n_pred+1,nstep_mlorg)):
        for j in range(n_pages):
            
            plt.figure()
            fig, ax = plt.subplots(nrows=nrow_mlmodes_plot,ncols=ncol_mlmodes_plot,figsize=(12,8),sharex=True)
            ax = ax.flat
            nrs = at2.shape[1]
            t_out = np.linspace(1,n_pred,n_pred)
        
            for i in range(n_perpage):
                t = np.linspace(1,k,k)
                ax[i].plot(t,at2[:k,i+j*8],'k',label=r'True Values')
                ax[i].plot(t,ytest_ml[:k,i+j*8],'b-',label=r'ML ')
                ax[i].set_xlabel(r'$t$',fontsize=14)
                ax[i].set_ylabel(r'$a_{'+str(i+1+j*8) +'}(t)$',fontsize=14)    
                ax[-1].set_xlim([t_out[0],t_out[-1]])
                ax[i].axvspan(0, t[n_train-1], alpha=0.2, color='darkorange')
        
            ax[-2].set_xlabel(r'$t$',fontsize=14)    
            ax[-1].set_xlabel(r'$t$',fontsize=14)    
            fig.tight_layout()
        
            fig.subplots_adjust(bottom=0.1)
            line_labels = ["True", "ML"]#, "ML-Test"]
            plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
             
            
        
            fig.savefig('plots/gifmlorg/true_ml'+str(j+1)+'.'+ str(int((k-n_train)/nstep_mlorg)) +'.png', dpi=300)
            plt.show()
            #plt.close()

    del fig,ax,t,t_out 
#plt.close('all')
#%%
"""
Function to calculate and plot Hurst Coefficients
"""    
def calc_hurst(nr,n_pred,at):
    H = np.zeros(nr)
    for i in range(nr):
        H[i],c,data =  compute_Hc(at[:n_pred,i])
    H = abs(H)    
    plt.figure()
    plt.tight_layout()
    plt.title("Hurst exponent for various modes")
    plt.plot(H,'bo-')
    plt.legend("Modes")
     
    
    plt.savefig("plots/hurst.png",dpi=300)
    plt.show()
    
    return H
#%%
"""
Function to calculate specific modes in physical space
Input   :   n- Number of the mode needed
        ntime- Number of the time step snapshot 
            a- matrix with the modes of form (a[:,n] for nth mode)
          phi- the basis
        iflag- =0 Just one mode numbered n
"""
def mode_phy(a,phi,nb,ne):
    
    a_slice = a[:,nb:ne+1]
    a_mod   = np.zeros((a.shape[0],a.shape[1])) 
    a_mod[:,nb:ne+1] = a_slice 
    
    #Find the values in physical space
    u_var = np.dot(phi,a_mod.T) 

    return u_var


