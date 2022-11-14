# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:00:41 2022

@author: s163761
"""

#%% Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter("ignore")

import sys
import os

# Import time-keeping
import time

# Create distance matrix faster
from scipy.spatial import distance_matrix

# For GPy
#import GPy
#GPy.plotting.change_plotting_library('matplotlib')

import pylab as pb

import DamageTools


#%% Folder structure

folder_accs = r'output_files\ACCS'

folder_structure = r'output_files'

folder_figure_save = r'output_files\Entropy'

#%% Load Structure
Structure = pd.read_pickle( os.path.join(folder_structure, '00_Structure.pkl') )
Index_Results = pd.read_pickle( os.path.join(folder_structure, '00_Index_Results.pkl') )
#[10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]
struc_nodes = Structure.Nodes[0]

struc_periods = list(Structure.Periods[0])

#%% Estimate Entropy
df_SampEn = pd.DataFrame(0,columns = struc_nodes, index = Index_Results.index)

# r=root, d=directories, f = files
for rdirs, dirs, files in os.walk(folder_accs):
    for file in files:
        
        # Load Ground Motions for X/Y
        if rdirs == folder_accs and file.endswith("Accs.out"):
            #print(os.path.join(rdirs, file))
            #print(idx)
            #print('Loading file: ',file)
            
            time_Accs = np.loadtxt( os.path.join(folder_accs, file) )
            
            if file[3:6][0] != str(0):
                idx = int(file[3:6])
            elif file[3:6][1] != str(0):
                idx = int(file[4:6])
            else:
                idx = int(file[5:6])
                    
            # GM = Index_Results['Ground motion'][idx]
            # LF = Index_Results['Load factor'][idx]
            # print('GM: ',GM ,'Loadfactor: ', LF)
            print('EarthQuake ID: ', idx)
            
            
            
            # Load Accelerations in nodes X
            for j in range(1,len(time_Accs[0])):
                #time = time_Accs[:,0]
                accs = time_Accs[:,j].tolist()
                
                SampEn = DamageTools.SampEn(accs, 2, 0.2*np.std(accs))
                
                df_SampEn[struc_nodes[j-1]][idx] = SampEn
                
            
df_SampEn.to_pickle(folder_structure + "/00_SampEn.pkl")
#%% Load
df_SampEn = pd.read_pickle( os.path.join(folder_structure, '00_SampEn.pkl') ) 

#%% 
x = struc_periods
cm = 1/2.54  # centimeters in inches
for i in df_SampEn.index:
    
    GM = Index_Results['Ground motion'][i]
    
        
    fig = plt.figure(figsize=(20*cm, 15*cm)); ax = fig.add_subplot(111)
    x = df_SampEn.columns.tolist()
    plt.plot(x,df_SampEn.loc[i].tolist())
    plt.grid()
    
    plt.xticks(x[0:len(x):4] + x[3:len(x):4])
    
    rel_height = 0.35
    plt.text(x=0.09, y=rel_height, s='Ground',    rotation=90, va='bottom', ha='center', transform = ax.transAxes)
    plt.text(x=0.37, y=rel_height, s='1st Floor', rotation=90, va='bottom', ha='center', transform = ax.transAxes)
    plt.text(x=0.65, y=rel_height, s='2nd Floor', rotation=90, va='bottom', ha='center', transform = ax.transAxes)
    plt.text(x=0.92, y=rel_height, s='3rd Floor', rotation=90, va='bottom', ha='center', transform = ax.transAxes)
    
    for j in [0, 1, 2, 3]:
        plt.axvspan(x[4*j], x[(4*j)+3], alpha=0.4, color='tab:blue')
    
    plt.xlabel('Nodes')
    plt.ylabel('SampEn')
    
    plt.title(f' SampEn estimations of Accelerations \n {GM}')
    
    
    plt.savefig(os.path.join(folder_figure_save, f'ACC_Entropy_{GM}.png'))
    plt.close()
    

#%%

df = df_SampEn
# Plot bloxplot
cm = 1/2.54  # centimeters in inches
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25*cm, 15*cm))
#plt.figure(figsize =(10, 10)); ax = fig.add_subplot(111)


len_sensor = len(df.columns.tolist())

 #for error in df.index.tolist():
     #cur_error = list(df.index)[error_id]

sensor_id = 1
for sensor in df.columns.tolist():
    cur_sensor = list(df.columns)[sensor_id-1]
    
    data = np.array(df[sensor].tolist()).reshape(-1,1)
    
    ax.boxplot(data,widths=0.5, positions=[sensor_id], labels=[f'{cur_sensor}'])  
    ax.text(x=(sensor_id-.5)/(len_sensor), y=1, s=f'({round(data.mean(),2)})', va='bottom', ha='center', transform = ax.transAxes)
        
    if sensor_id in [5, 9, 13]:
        ax.axvline(x = sensor_id-0.5, color = 'black', linestyle='dashed', linewidth=1.3, label = 'axvline - full height')
    
    sensor_id+=1


rel_height = 0.99
plt.text(x=0.12, y=rel_height, s='Ground',    rotation=0, va='top', ha='center', transform = ax.transAxes)
plt.text(x=0.37, y=rel_height, s='1st Floor', rotation=0, va='top', ha='center', transform = ax.transAxes)
plt.text(x=0.63, y=rel_height, s='2nd Floor', rotation=0, va='top', ha='center', transform = ax.transAxes)
plt.text(x=0.87, y=rel_height, s='3rd Floor', rotation=0, va='top', ha='center', transform = ax.transAxes)

plt.grid(axis='y')
plt.title('SampEn of Acceleration \n All Earthquake loadings (mean) \n')
plt.xlabel('Nodes')
plt.ylabel('SampEn')


plt.savefig(os.path.join(folder_figure_save, 'All_ACC_Entropy.png'))
#plt.close()
    
#%%

def errors(y_true, y_pred):
    RMSE = ((y_pred - y_true)**2).mean() **.5
    
    SMSE = ((y_pred - y_true)**2).mean() / y_true.var()
    
    MAE = (abs(y_pred - y_true)).mean()
    MAPE = (abs((y_pred - y_true)/y_true)).mean()*100
    
    TRAC = np.dot(y_pred.T,y_true)**2 / (np.dot(y_true.T,y_true)*np.dot(y_pred.T,y_pred))
    # Dustance    
    #DIST = ((y_pred - y_true)**2).sum() **.5
    #DISTN = ((y_pred/y_true - 1)**2).sum() **.5
    return RMSE, SMSE, MAE, MAPE, TRAC
#%%

Errors = ['RMSE', 'SMSE', 'MAE', 'TRAC']
df_Close_Error = pd.DataFrame( columns = Errors, index = Index_Results.index)

for EQ in Index_Results.index:
    for Error in Errors:
    
        df_Close_Error[Error][EQ] = pd.DataFrame(0, columns = struc_nodes, index = struc_nodes)
        
#%%      
# r=root, d=directories, f = files
for rdirs, dirs, files in os.walk(folder_accs):
    for file in files:
        
        # Load Ground Motions for X/Y
        if rdirs == folder_accs and file.endswith("Accs.out"):
            #print(os.path.join(rdirs, file))
            #print(idx)
            #print('Loading file: ',file)
            
            time_Accs = np.loadtxt( os.path.join(folder_accs, file) )
            
            if file[3:6][0] != str(0):
                idx = int(file[3:6])
            elif file[3:6][1] != str(0):
                idx = int(file[4:6])
            else:
                idx = int(file[5:6])
                    
            # GM = Index_Results['Ground motion'][idx]
            # LF = Index_Results['Load factor'][idx]
            # print('GM: ',GM ,'Loadfactor: ', LF)
            print('EarthQuake ID: ', idx)
            
            
            
            # Load Accelerations in nodes X
            for j in range(1,len(time_Accs[0])):
                #time = time_Accs[:,0]
                accs_j = np.array(time_Accs[:,j].tolist())
                
                # Load Accelerations in nodes X
                for i in range(1,len(time_Accs[0])):
                    #time = time_Accs[:,0]
                    accs_i = np.array(time_Accs[:,i].tolist())
                    
                    
                    
                    RMSE, SMSE, MAE, MAPE, TRAC = errors(accs_j, accs_i)
                    
                    df_Close_Error['RMSE'][idx][struc_nodes[j-1]][struc_nodes[i-1]] = RMSE
                    df_Close_Error['SMSE'][idx][struc_nodes[j-1]][struc_nodes[i-1]] = SMSE
                    df_Close_Error['MAE'][idx][struc_nodes[j-1]][struc_nodes[i-1]] = MAE
                    df_Close_Error['TRAC'][idx][struc_nodes[j-1]][struc_nodes[i-1]] = TRAC
                    
#%%         Save DataFrame        
df_Close_Error.to_pickle(folder_structure + "/00_df_Close_Error.pkl")
       
#%% Load DaraFRame
df_Close_Error = pd.read_pickle( os.path.join(folder_structure, '00_df_Close_Error.pkl') ) 

#%% Add DataFrame

Errors = ['RMSE', 'SMSE', 'MAE', 'TRAC']
df_Error_mean = pd.DataFrame( columns = Errors, index = [0])

for error in Errors:
    #error = 'TRAC'
    
    df_sum_error = df_Close_Error[error][0]
    len_sum_error = len(df_Close_Error.index)
    
    for i in df_Close_Error.index[1:]:
        df_sum_error = df_sum_error.add(df_Close_Error[error][i])
    df_sum_error = df_sum_error / len_sum_error

    df_Error_mean[error][0] = df_sum_error

#%% Save DataFrame
df_Error_mean.to_pickle(folder_structure + "/00_df_Close_Error_Mean.pkl")

#%% Load DataFrame
df_Error_mean = pd.read_pickle( os.path.join(folder_structure, '00_df_Close_Error_Mean.pkl') ) 

#%% Heat Map
Errors = ['RMSE', 'SMSE', 'MAE', 'TRAC']
for error in Errors:
    #error = 'RMSE'
    fig = plt.figure(figsize =(8, 7))#; ax = fig.add_subplot(111)
    
    df = df_Error_mean[error][0]
    plt.pcolor(df)
    
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    plt.colorbar(label=f'{error} Mean')
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            if round(df.iloc[j,i],2) > 0.9:
                text = plt.text(j+0.5, i+0.5, round(df.iloc[j,i],2),
                               ha="center", va="center", color="k", fontsize='small')#, transform = ax.transAxes)
            else:
                text = plt.text(j+0.5, i+0.5, round(df.iloc[j,i],2),
                               ha="center", va="center", color="w", fontsize='small')#, transform = ax.transAxes)
                    
    
    plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
    
    plt.suptitle(f'Error Heat Map - OpenSees \n {error} Error' )
    plt.xlabel('Testing Nodes')
    plt.ylabel('Training Nodes')
    #plt.show()
    
    plt.savefig(os.path.join(folder_structure, f'ErrorMap_OpenSees_{error}.png'))
    #plt.close()