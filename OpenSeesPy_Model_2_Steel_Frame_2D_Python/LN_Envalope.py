# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:21:43 2022

@author: s163761
"""


#%% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

#%%

folder_structure = r'output_linear_non'


#%%

Index_Results = pd.read_pickle(os.path.join(folder_structure, '00_Index_Results.pkl'))
Structure = pd.read_pickle(os.path.join(folder_structure, '00_Structure.pkl'))

#%%

LN_nodes = Index_Results['LN_Node']
LN_energy = Index_Results['LN_Energy']
LN_res_def = Index_Results['LN_Res_Def']

struc_nodes = Structure['Nodes'][0]

df_LN = pd.DataFrame(0, columns=struc_nodes, index = Index_Results.index)
#df_LN = pd.DataFrame(0, columns=struc_nodes, index = Index_Results.index)

energy_thr = 0.01 # Larger values are non-linear

plt.figure()

nonlinear_elements = 0; linear_elements = 0
for i in range(LN_nodes.shape[0]): # EQ
    for j in range(len(LN_nodes[0])): # 2 in each element (21 elements)
        node_0 = LN_nodes[i][j][0]
        node_1 = LN_nodes[i][j][1]
        
        energy_0 = LN_energy[i][j][0]
        energy_1 = LN_energy[i][j][1]
        
        if energy_0 > energy_thr or energy_1 > energy_thr:
            nonlinear_elements += 1
        else:
            linear_elements += 1
        
        res_dif_0 = np.abs(LN_res_def[i][j][0])
        res_dif_1 = np.abs(LN_res_def[i][j][1])
        
        # End point 0
        if energy_0 > energy_thr:
            #df_LN[node_0][i] = 'N'
            if df_LN[node_0][i] == 0: #isinstance(df_LN[node_0][i], str):
                df_LN[node_0][i] = energy_0
            else:
                df_LN[node_0][i] = max(df_LN[node_0][i], energy_0)
                
            plt.scatter(energy_0, res_dif_0, c='tab:blue')
            
        else:
            plt.scatter(energy_0, res_dif_0, c='tab:orange')
                
        # End point 1
        if energy_1 > energy_thr:
            #df_LN[node_1][i] = 'N'
            if df_LN[node_1][i] == 0: #isinstance(df_LN[node_1][i], str):
                df_LN[node_1][i] = energy_1
            else:
                df_LN[node_1][i] = max(df_LN[node_1][i], energy_1)
                
            plt.scatter(energy_1, res_dif_1, c='tab:blue')
            
        else:
            plt.scatter(energy_1, res_dif_1, c='tab:orange')
        
plt.grid()
plt.xlabel('Energy')
plt.ylabel('Elasic deformation (residual rotation)')
plt.axvline(energy_thr, linewidth=1, linestyle='--', c='k')

#%% Save
df_LN.to_pickle(folder_structure + "/00_LN_Envalope.pkl")

#%% Load
df_LN = pd.read_pickle( os.path.join(folder_structure + "/00_LN_Envalope.pkl") )
#%% Counting

# Row rount
row_Ns = (df_LN != 0).sum(axis=1)
plt.figure()
plt.plot(row_Ns)
plt.grid()


num_L = (df_LN.values == 0).sum()
num_N = (df_LN.values != 0).sum()
num_T = num_L + num_N

num_T_el = nonlinear_elements + linear_elements

print(f'L/N: {num_L}/{num_N} -- ({round(num_L/num_T,4)}/{round(num_N/num_T,4)})')
print(f'L/N EL: {linear_elements}/{nonlinear_elements} -- ({round(linear_elements/num_T_el,4)}/{round(nonlinear_elements/num_T_el,4)})')


#%% Find higest energy in nide 23

df_test = df_LN.replace('L', 0)
df_test = df_test[[23, 22, 32, 42]]

df_test.sort_values(by=[23], inplace=True)

print(df_test.tail(10).index)  

#%% Getting Lin/NonLin Nodes
df_list = pd.DataFrame([], columns = df_test.columns, index = ['L', 'N'])

N_threshold = 4

for i in df_test.columns:
    L_val = df_test[i][df_test[i] == 0].shape[0]
    N_val = df_test[i][df_test[i] < N_threshold].shape[0] - L_val
    
    df_list[i]['L'] = df_test[i][df_test[i] == 0].index.tolist()
    N_list = df_test[i][df_test[i] < N_threshold].index.tolist()
    for k in df_list[i]['L']:
        N_list.remove(k)
        
    df_list[i]['N'] = N_list
    
    print(f'Node {i}:', L_val, N_val)
    #print('Lin EQs \n', df_list[i]['L'])
    #print('NLin EQs \n', df_list[i]['N'])
    #print()

#%% Remove from 23 - L

for i in df_list[22]['L'] + df_list[32]['L']:
    
    if i in df_list[23]['L']:
        df_list[23]['L'].remove(i)
        
for j in df_list.columns:
    
    print(f'Node {j}: ', len(df_list[j]['L']), len(df_list[j]['N']))
    
#%% Remove from 42 - L

for i in df_list[23]['L']:
    
    if i in df_list[42]['L']:
        df_list[42]['L'].remove(i)
        
for j in df_list.columns:
    
    print(f'Node {j}: ', len(df_list[j]['L']), len(df_list[j]['N']))
    
    
#%% Remove from 23 - N

for i in df_list[22]['N']:# + df_list[32]['N']:
    
    if i in df_list[23]['N']:
        df_list[23]['N'].remove(i)
        
for j in df_list.columns:
    
    print(f'Node {j}: ', len(df_list[j]['L']), len(df_list[j]['N']))
    
#%% Remove from 22 + 32 + 42 - N

for i in df_list[23]['N']:
    
    if i in df_list[42]['N'] :
        df_list[42]['N'].remove(i)
        
    if i in df_list[32]['N'] :
        df_list[32]['N'].remove(i)
        
    if i in df_list[22]['N'] :
        df_list[22]['N'].remove(i)
        
for j in df_list.columns:
    
    print(f'Node {j}: ', len(df_list[j]['L']), len(df_list[j]['N']))


#%% Function
def common_member(a, b):
    result = [i for i in a if i in b]
    return result




common_temp = common_member(df_list[42]['N'], df_list[32]['N'])
common_temp = common_member(common_temp, df_list[22]['N'])

for i in [22, 32, 42]:
    df_list[i]['N'] = common_temp
#%% Save
df_list.to_pickle(folder_structure + "/00_EQ_List.pkl")

sys.exit()
# Results from Damage Index
#df.to_csv(output_directory + r'/00_Index_Results.csv')  # export dataframe to cvs
#df.to_pickle(output_directory + "/00_Index_Results.pkl") 
#unpickled_df = pd.read_pickle("./dummy.pkl")  