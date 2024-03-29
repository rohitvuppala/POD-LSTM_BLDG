#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 20:28:56 2021

@author: Rohit K S S Vuppala
@email : rvuppal@okstate.edu
         Graduate Student
         Mechanical and Aerospace Engineering
         Oklahoma State University


"""

#%%


exec(open("imp_lib.py").read())

from functions import *
from misc_plotting import *

exec(open("init.py").read())
    

#%%
"""
Choose backend for plotting
"""
if imp_backend == 1:
    mp.use(mp_backend)
#%%
"""
Write the .mat file if required (NC2MAT)
"""
if (inc2mat==1): 
    nc2mat(nc2mat_fname_nc,nc2mat_fname_mat,nc2mat_ntot,nc2mat_start)
    
#%%
"""
Load from data from file
"""
(u,x,y,z,nx,ny,nz) = load_data(idata_in,data_ntot,n_offset,data_fname)


#%%
"""
Number of training data 
n_train : from input.yaml file
"""

u_org= np.where(u<-9998.0,np.nan,u)
u  = np.where(np.isnan(u_org),0.0,u)

u_test     = u[:,:n_train]
#u_test = np.where(u_test<-9998.0,0.0,u_test)
u_test_avg = np.sum(u_test,axis=1,keepdims=True)/(n_train) #keepdims to ensure it is vector
u_variation_test = u_test - u_test_avg

"""
v_test     = v[:,:n_train]
v_test_avg = np.sum(v_test,axis=1,keepdims=True)/(n_train) #keepdims to ensure it is vector
v_variation_test = v_test - v_test_avg

w_test     = w[:,:n_train]
w_test_avg = np.sum(w_test,axis=1,keepdims=True)/(n_train) #keepdims to ensure it is vector
w_variation_test = w_test - w_test_avg
"""

#dat_variation_test = np.concatenate((u_variation_test,v_variation_test,w_variation_test))
#%%
"""
Calculating and Plotting the Relative Importance Index
For chosen number of modes    
"""

#Calculate the Square of singular values for RIC
S2_nr = calc_S2(u_variation_test)

#Calculate RIC
S2_nr_ric = np.zeros(S2_nr.shape[0])
for n in range(S2_nr.shape[0]):
    S2_nr_ric[n] = np.sum(S2_nr[:n],axis=0,keepdims=True)/np.sum(S2_nr,axis=0,keepdims=True)*100

if (iric_lim==0):
    nr = n_modes
elif (iric_lim==1):
    idx = (np.abs(S2_nr_ric - ric_lim)).argmin()
    nr = idx + 1
else:
    print_status("E: Error incorrect iric_lim chosen")

    
#Construct the basis for 'nr' modes
PHIw, S2_nr, RIC  = con_basis(u_variation_test, nr)     
    
#Plotting the RIC 
if (iric_plot==1):    
    plot_ric(n_train,nr,S2_nr_ric,RIC)
    
#%%

"""
Calc and Plot the true modes for all the training data
"""
#Calculating modes
at = get_coeff(u_variation_test, PHIw)

#Plot modes if necessary
if (itrumodes_plot==1):
    plot_trumodes(n_train,n_trumodes_plot,nrow_trumodes_plot,ncol_trumodes_plot,at)
    
#%%
"""
Detrend the data
"""
if(i_detrend==1):
    p_detrend = np.zeros((nr,nord_detrend+1))
    (at_detrend,p_detrend) = detrend_train(at,nr,n_train,nord_detrend)
            
    plot_dtmodes(n_train,n_trumodes_plot,nrow_trumodes_plot,ncol_trumodes_plot,at_detrend)
        
#%%
"""
Reconstruct True fluc and True Temp from modes
"""
u_true_fluc = recon_data(at,PHIw)
u_true = u_true_fluc + u_test_avg
#%%
#Plot first snapshot 
idx = 20
plot_nthsnapshot_org(u,u_true,x,y,z,nx,ny,nz,0,cm,idx,'building-span-ratio=0.5')
idx = 15
plot_nthsnapshot_org(u,u_true,x,y,z,nx,ny,nz,0,cm,idx,'building-span-ratio=0.0')
idx = 25
plot_nthsnapshot_org(u,u_true,x,y,z,nx,ny,nz,0,cm,idx,'building-span-ratio=1.0')


    
#%%
"""
Pre-process data for training: 1. Detrend the data if needed
                               2. Scale the data from (-1,1)
                               3. Create xtrain and ytrain for LSTM in its format

"""
lookback = n_lookback

if (i_detrend==1) :
    atrain = at_detrend[:n_train,:]
else:
    atrain = at[:n_train,:]
   
#n_train,nr = atrain.shape
m,n = n_train,nr
       
#Scaling the data from -1 to 1
sc = MinMaxScaler(feature_range=(-1,1))
training_set_scaled = sc.fit_transform(atrain)
training_set = training_set_scaled

#%%
#Creating the training data for the LSTM

data_sc, labels_sc = create_training_data_lstm(training_set, m, n, lookback)
xtrain, xvalid, ytrain, yvalid = train_test_split(data_sc, labels_sc, test_size=0.3 , shuffle= True)

#%%
"""
Init and compile the model
"""
training_time_init = tm.time()
nl = n_hidlayers
nn = n_neurons
rdrp= rec_dropout
drp = dropout
model = model_lstm(nl,nn,lookback,nr,drp,rdrp)
model.summary()
    
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[coeff_metric])
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


#%%
"""
Start the training
"""
if(itrain==1):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=50, min_lr=0.0005)
    history = model.fit(xtrain, ytrain, epochs=n_epochs, batch_size=n_batchsize, 
                        validation_data= (xvalid,yvalid),callbacks=[reduce_lr,tensorboard_callback])
    
    
    
    # Training time per each step and total time 
    total_training_time = tm.time() - training_time_init
    print('Total training time=', total_training_time)
    cpu = open("a_cpu.txt", "w+")
    cpu.write('training time in seconds =')
    cpu.write(str(total_training_time))
    cpu.write('\n')


#%%
"""
Save the model if itrain = 1
"""

if(itrain==1):
    plot_training(model_fname,model,history)
    
#%%
"""
Scale the testing data and prepare for prediction
"""

u_pred     = u[:,:n_pred]
u_pred_avg = np.sum(u_pred,axis=1,keepdims=True)/(n_pred) #keepdims to ensure it is vector
u_variation_pred = u_pred - u_pred_avg

if (i_detrend==1) :
    at2 = get_coeff(u_variation_pred, PHIw)
    at2_detrend = detrend_pred(at2,p_detrend,nr,1,n_pred,nord_detrend) 
    testing_set = np.copy(at2_detrend)
else:
    at2 = get_coeff(u_variation_pred, PHIw)
    testing_set = np.copy(at2)

testing_set_scaled = sc.fit_transform(testing_set)

m,n = testing_set_scaled.shape
ytest = np.zeros((1,lookback,n))
ytest_ml = np.zeros((m,n))

# create input at t = 0 for the model testing
for i in range(lookback):
    ytest[0,i,:] = testing_set_scaled[i]
    ytest_ml[i] = testing_set_scaled[i]
#%%
"""
Start the predictions with the trained model for test data
"""
testing_time_init = tm.time()

#Load model
if (itrain==0):
    model = tf.keras.models.load_model(model_fname,custom_objects={'coeff_metric':coeff_metric })

print_status("Predicting Results")

# predict results recursively using the model
for i in tqdm(range(lookback,m)):
    slope_ml = model.predict(ytest)
    ytest_ml[i] = slope_ml
    e = ytest
    for i in range(lookback-1):
        e[0,i,:] = e[0,i+1,:]
    e[0,lookback-1,:] = slope_ml
    ytest = e 

#%%
"""
Post process the predicted data: 1. Unscale the data
                                 2. Re-trend the data
                                 2. Add to mean to predicted fluct to get the plots
"""

ytest_ml_unscaled = sc.inverse_transform(ytest_ml)
ytest_ml_detrend= np.copy(ytest_ml_unscaled)
#%%
#Retrend the data
if (i_detrend==1):
    ytest_ml = retrend(ytest_ml_detrend,p_detrend,nr,1,n_pred,nord_detrend) 
else:
    ytest_ml = ytest_ml_detrend
#%%    
"""
Plot the ml modes with original
"""

plot_orgmlmodes(n_pages,n_perpage,nrow_mlmodes_plot,ncol_mlmodes_plot,n_pred,n_train,at2,ytest_ml,'org_ml','True and ML predicted modes')
if (i_detrend==1):
    plot_orgmlmodes(n_pages,n_perpage,nrow_mlmodes_plot,ncol_mlmodes_plot,n_pred,n_train,at2_detrend,ytest_ml_detrend,'org_ml_detrend','Detrended-Original and ML modes')


#%%
"""
Reconstruct True fluc and Temp from ml
"""
u_ml_fluc = recon_data(ytest_ml,PHIw)
u_ml = u_ml_fluc + u_test_avg
#%%
"""
Reconstruction from modes
"""
u_recon_fluc = recon_data(at2, PHIw)
u_recon = u_recon_fluc + u_pred_avg
#%%
"""
Write the ml values to 3D .nc
"""
if irecon_nc==1:
    write_recon(u_ml,u_recon,x,y,z,nx,ny,nz,t_start,t_end,n_pred)  


#%%
"""
Plotting nth snapshot
"""
for i in range(n_plot_num):
    idx = 20
    plot_nthsnapshot(u_recon,u_ml,x,y,z,nx,ny,nz,n_plot_start+i*n_plot_step,cm,idx,'building-span-ratio=0.5')
    idx = 15
    plot_nthsnapshot(u_recon,u_ml,x,y,z,nx,ny,nz,n_plot_start+i*n_plot_step,cm,idx,'building-span-ratio=0.0')
    idx = 25
    plot_nthsnapshot(u_recon,u_ml,x,y,z,nx,ny,nz,n_plot_start+i*n_plot_step,cm,idx,'building-span-ratio=1.0')
    idz = 5
    plot_nthsnapshot_xy(u_recon,u_ml,x,y,z,nx,ny,nz,n_plot_start+i*n_plot_step,cm,idz,'mid-height')
#%%
"""
Calculate L2 norm of the difference in the Recon and Org Data
"""

if (il2norm == 1):
    l2_norm_diff,l2_rel_diff = calc_l2norm_rom(u_ml,u_recon,n_pred,n_train,nx,ny,nz)
#%%
"""
Calculate Hurst component
"""
if (ihurst==1):
    H = calc_hurst(nr,n_pred,at)

"""
Plot the true modes gif all training data and comp. with org.
"""

if (igif_tru==1):    
    plot_giftru(nstart_gif_tru,nstep_gif_tru,nrow_trumodes_plot,ncol_trumodes_plot,n_train,u_variation_test,PHIw)
 
if (igif_mlorg==1):
    plot_gifmlorg(n_pages,n_perpage,nstep_mlorg,nrow_mlmodes_plot,ncol_mlmodes_plot,n_train,n_pred,at2,ytest_ml)
    
#%%
"""
    Find the mean, min and max for u'
"""
u_recon_mean=np.mean(u_recon_fluc[:,:],axis=0)
u_recon_std=np.std(u_recon_fluc[:,:],axis=0)


u_ml_mean=np.mean(u_ml_fluc[:,:],axis=0) 
u_ml_std =np.std(u_ml_fluc[:,:],axis=0)

plt.figure(figsize=(10,5))
#plt.title("Mean of the u-velocity fluctutation",fontsize=18)
plt.xlabel("Snapshot number",fontsize=14)
plt.ylabel("Mean value",fontsize=14)
plt.xlim(0,n_pred)
plt.tight_layout()
plt.axvspan(0, n_train, alpha=0.2, color='green')
plt.axvspan(n_train,n_pred,alpha=0.2, color='red')
plt.plot(u_ml_mean,'b')
plt.plot(u_recon_mean,'k')
line_labels = ["ML-ROM modes reconstructed", "Exact POD modes reconstructed"]#, "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.4, ncol=3, labelspacing=0., fontsize=14)
plt.savefig('plots/comp_mean.png',dpi=300)
plt.show()


# plt.figure(figsize=(10,5))
# #plt.title("Variance of the u-velocity fluctutation",fontsize=18)
# plt.xlabel("Snapshot number",fontsize=14)
# plt.ylabel("Variance",fontsize=14)
# plt.xlim(0,n_pred)
# plt.tight_layout()
# plt.axvspan(0, n_train, alpha=0.2, color='green')
# plt.axvspan(n_train,n_pred,alpha=0.2, color='red')
# plt.plot(u_ml_var,'b')
# plt.plot(u_recon_var,'k')
# line_labels = ["ML-ROM modes reconstructed", "Exact POD modes reconstructed"]#, "ML-Test"]
# plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.4, ncol=3, labelspacing=0., fontsize=14)
# plt.savefig('plots/comp_var.png',dpi=300)
# plt.show()

#%%
plt.figure(figsize=(10,5))
plt.title("U-velocity field characteristics",fontsize=18)
plt.xlabel("Snapshot number",fontsize=14)
plt.ylabel("($\mu-\sigma,\mu+\sigma)$",fontsize=14)
plt.xlim(0,n_pred)
plt.tight_layout()

plt.plot(u_ml_mean,'r')
plt.plot(u_recon_mean,'b')
plt.fill_between(range(n_pred),u_ml_mean+u_ml_std,u_ml_mean-u_ml_std,alpha=0.4,color='r')
plt.fill_between(range(n_pred),u_recon_mean+u_recon_std,u_recon_mean-u_recon_std,alpha=0.4,color='b')
plt.axvspan(0, n_train, alpha=0.1, color='green')
#plt.axvspan(n_train,n_pred,alpha=0.1, color='red')


line_labels = ["From ML-ROM ($\mu$)", " From exact POD modes ($\mu$)"]#, "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.4, ncol=3, labelspacing=0., fontsize=14)
plt.savefig('plots/comp_var.png',dpi=300)
plt.show()



#%%    
"""
Investigating the modes
"""
"""
for i in range(6):
    
    atmode = at2[:,i]
    u_mode_fluc = recon_data(atmode.reshape((-1,1)), PHIw[:,i].reshape((-1,1)))
    u_mode = u_mode_fluc + u_pred_avg
    
    # ytest_mode = ytest_ml[:,i]
    # u_mode_ml_fluc = recon_data(ytest_mode,PHIw)
    # u_mode_ml = u_mode_ml_fluc + u_test_avg


    n_plot= 0
    #Plot them
    plt.figure() 
    X,Y = np.meshgrid(x,z)
    
    #n_plot = 359
    fig,axs = plt.subplots(1,1, figsize=(6,5))
    
    current_cmap = plt.cm.get_cmap('hot')
    current_cmap.set_bad(color='white',alpha=1.0)
    
    
    u_3d_org = np.copy(u_mode[:,n_plot-1].reshape(nz,ny,nx))
    
    #levels = np.linspace(np.min(u_3d_org),np.max(u_3d_org),120)
    mid    = int(ny/2)
    
    #Nan values for building
    u_3d_org[0:10,15:25,20:30] = np.nan
    
    #Plot from the data 
    axs.set_title("Mode number-"+str(i+1))
    axs.set_xlabel("x location in metres",fontsize=12)
    axs.set_ylabel("y location in metres",fontsize=12)
    cs = axs.contourf(X,Y,u_3d_org[:,mid,:],cmap=cm,extend='both')
    #cs1 = axs[0].contour(X,Y,u_3d_org[mid,:,:],levels=[-2,-1.5,-1,-0.5,0,2,4,6,8,9],colors='k',extend='both')
    cbar=fig.colorbar(cs, ax=axs, orientation='vertical',shrink=1.0)
    cbar.ax.set_title('u-vel',fontsize=12)
    plt.clabel(cs1,inline=True)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Mode"+str(i)+" for first snapshot at mid-height")
    plt.savefig("plots/mode"+str(i)+"_snapshotxz.png",dpi=300)
    plt.show() 
"""

#%%
"""
nb = 0
ne = 10
nstep = 100
u_modes_var = mode_phy(at,PHIw,nb,ne)
u_modes     = u_modes_var + u_test_avg 
u_modes_var = u_modes_var.reshape((nz,ny,nx,n_train))
u_modes     = u_modes.reshape((nz,ny,nx,n_train))  

vmin_var = np.amin(u_modes_var[:,0,:,:])
vmax_var = np.amax(u_modes_var[:,0,:,:])

vmin = np.amin(u_modes[:,0,:,:])
vmax = np.amax(u_modes[:,0,:,:])

os.system('rm -rf plots/modes')
os.system('mkdir plots/modes')  
    
X,Y = np.meshgrid(x,z)
for k in range(0,n_train,nstep):
    plt.figure()
    fig,axs = plt.subplots(1,2, figsize=(12,5))
    plt.suptitle('Velocity Variation and total for modes '+str(nb)+' to '+str(ne))
    
    levels_var = np.linspace(vmin_var,vmax_var,120)
    axs[0].set_title("Variation")
    cs = axs[0].contourf(X,Y,u_modes_var[:,0,:,k],levels_var,cmap='jet')
    cs.set_clim([vmin_var, vmax_var])
    fig.colorbar(cs, ax=axs[0], orientation='vertical',shrink=1.0)
    
    #Plot from ML
    levels = np.linspace(vmin,vmax,120)
    axs[1].set_title("Total Value")
    cs = axs[1].contourf(X,Y,u_modes[:,0,:,k],levels,cmap='jet',vmin = vmin,vmax = vmax)
    cs.set_clim([vmin, vmax])
    fig.colorbar(cs, ax=axs[1], orientation='vertical',shrink=1.0)
    
    fig.savefig("plots/modes/modes_"+str(int(k/nstep)+1)+".png")
    plt.show()
""" 
 
    
    
    
    
