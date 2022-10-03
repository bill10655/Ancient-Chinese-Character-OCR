#%%
from pathlib import Path

#%%
folder = Path('data2')
fnames = [fname.stem for fname in folder.glob('*.jpg')]

# %%
import pickle

with open('data2_list.pickle', 'wb') as infile:
    pickle.dump(fnames, infile)
