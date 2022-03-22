import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
import yaml
from rdkit import Chem
import pickle
import h5py

from commons.process_mols import *

def pairwise_distance_matrix(array):
    
    first = True
    for entry in array:
        
        if first:
            pairwise_dist = pairwise_distances(entry.numpy()).flatten().reshape(1,-1)
        else:
            pairwise_dist = np.vstack( (pairwise_dist, pairwise_distances(entry.numpy()).flatten().reshape(1,-1)) ) 

        if first:
            first = False
            
    return pairwise_dist

def store_data(data_names, data, path):
  hf = h5py.File(path, 'w')
  for i in range(len(data_names)):
    hf.create_dataset(data_names[i], data=data[i])
  hf.close()

def load_data(data_names, path):
  hf = h5py.File(path, 'r')
  data = []
  for i in range(len(data_names)):
    d = np.array(hf.get(data_names[i]))
    data.append(d)
  hf.close()
  return data

if __name__ == "__main__":
    results = torch.load('./runs/flexible_self_docking/predictions_RDKitTrue.pt')
    names, lig_keypts, rec_keypts = results['names'], results['lig_keypts'], results['rec_keypts']
    print('Results Loaded')
    
    complex_to_rec_lig = {}
    rec_to_complex_lig = {}
    rec_ids = {}
    lig_ids = {}

    rec_id, lig_id = 0, 0
    for name in names:

        rec_smiles = Chem.MolToSmiles(read_molecule(f'./data/PDBBind/{name}/{name}_protein_processed.pdb'))
        lig_smiles = Chem.MolToSmiles(read_molecule(f'./data/PDBBind/{name}/{name}_ligand.sdf'))

        complex_to_rec_lig[name] = (rec_smiles, lig_smiles)

        if rec_smiles not in rec_to_complex_lig:
            rec_to_complex_lig[rec_smiles] = [(name, lig_smiles)]
        else:
            rec_to_complex_lig[rec_smiles].append( (name, lig_smiles) )

        if rec_smiles not in rec_ids:
            rec_ids[rec_smiles] = rec_id
            rec_id += 1
        if lig_smiles not in lig_ids:
            lig_ids[lig_smiles] = lig_id
            lig_id += 1

    complex_mappings = {'complex_to_rec_lig': complex_to_rec_lig, 'rec_to_complex_lig': rec_to_complex_lig, 'rec_ids': rec_ids, 'lig_ids': lig_ids }
    with open('complex_mappings.pickle', 'wb') as outfile:
        pickle.dump(complex_mappings, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('Complex Mappings Finished')
    
    lig_matrix = pairwise_distance_matrix(lig_keypts)
    rec_matrix = pairwise_distance_matrix(rec_keypts)
    print('Pairwise Distances Matrices Obtained')
    
    pca = PCA(n_components=2)
    lig_reduced = pca.fit_transform(lig_matrix)
    rec_reduced = pca.fit_transform(rec_matrix)
    print('PCA Reduction Done')
    
    store_data(['stacked_ligand', 'stacked_receptor', 'reduced_ligand', 'reduced_receptor'], [lig_matrix, rec_matrix, lig_reduced, rec_reduced], './pairwise_dist_data.h5')
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
    ax1.scatter(lig_reduced[:,0], lig_reduced[:,1], color='blue', alpha=0.5)
    ax2.scatter(rec_reduced[:,0], rec_reduced[:,1], color='red', alpha=0.5)

    fig.savefig('keypts_PCA.png')
    fig.show()
    print('Keypoints Visualized')
    
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
    colors = np.linspace(0, 1, num = len(lig_reduced))
    ax1.scatter(lig_reduced[:,0], lig_reduced[:,1], c=colors, cmap='plasma', alpha=0.5)
    ax2.scatter(rec_reduced[:,0], rec_reduced[:,1], c=colors, cmap='plasma', alpha=0.5)

    fig.savefig('pairs_keypts_PCA.png')
    fig.show()
    print('Keypoint Pairs Visualized')
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))

    rec_colors = []
    lig_colors = []
    for name in names:

        rec, lig = complex_to_rec_lig[name]
        rec_colors.append( rec_ids[rec]/len(rec_ids) )
        lig_colors.append( lig_ids[lig]/len(lig_ids) )


    colors = np.linspace(0, 1, num = len(lig_reduced))
    ax1.scatter(lig_reduced[:,0], lig_reduced[:,1], c=rec_colors, cmap='plasma', alpha=0.5)
    ax2.scatter(rec_reduced[:,0], rec_reduced[:,1], c=lig_colors, cmap='plasma', alpha=0.5)

    fig.savefig('lig_rec_keypts_PCA.png')
    fig.show()
    print('DONE!')