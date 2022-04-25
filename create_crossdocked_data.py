import torch
import shutil
import os
from rdkit import Chem


def get_folder_pairs(folder_dir):
    
    folder_files = os.listdir(folder_dir)
    
    pairs = []
    for file in folder_files:
        if file[-3:] == 'pdb':
            pairs.append(file[0:-13])
        else:
            pairs.append(file[0:-4])
    
    return list(set(pairs))

if __name__=="__main__":
    
    print("Beginning!")
    os.mkdir('data/CrossDocked')
    data_dir = '/data/rsg/nlp/xiangfu/sbdd_data/crossdocked_pocket10'
    folders = os.listdir(data_dir)
    for folder in folders:
        
        try:
            print(f"FOLDER: {folder}")
            current_dir = f"{data_dir}/{folder}"
            pairs = get_folder_pairs(current_dir)
            for pair in pairs:

                #print(f"PAIR: {pair}")
                protein_dir = f'{current_dir}/{pair}_pocket10.pdb'
                ligand_dir = f'{current_dir}/{pair}.sdf'
                #print(protein_dir)
                #print(ligand_dir)

                try:
                    #print("protein: ", Chem.MolFromPDBFile(protein_dir, sanitize=False, removeHs=False))
                    pdb_mol = Chem.MolFromPDBFile(protein_dir, sanitize=False, removeHs=False)
                    #print("we good?")
                    #print("ok")
                    
                    #print("ligand: ", Chem.SDMolSupplier(ligand_dir, sanitize=False, removeHs=False))
                    supplier = Chem.SDMolSupplier(ligand_dir, sanitize=False, removeHs=False)
                    mol = supplier[0]
                    #print("did we make it?")

                    #print("Good Pair")
                    new_dir = f'data/CrossDocked/{pair}'
                    os.mkdir(new_dir)

                    shutil.copyfile(protein_dir, f'{new_dir}/{pair}_protein.pdb')
                    shutil.copyfile(protein_dir, f'{new_dir}/{pair}_protein_processed.pdb')
                    shutil.copyfile(ligand_dir, f'{new_dir}/{pair}_ligand.sdf')
                    
                except:
                    #print("")
                    print("Bad Pair")
                
        except:
            print("Bad Folder")
            
        
