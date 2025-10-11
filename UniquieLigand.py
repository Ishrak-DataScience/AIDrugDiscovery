from Bio.PDB.MMCIF2Dict import MMCIF2Dict

cif_dict = MMCIF2Dict("components.cif")
ligand_ids = cif_dict['_chem_comp.id']
print(f"Number of unique ligand IDs: {len(ligand_ids)}")
print(ligand_ids[:50]) 