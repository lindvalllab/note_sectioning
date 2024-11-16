import os
import pandas as pd

def load_data(studies_folder):
    type = studies_folder.strip('/').split('/')[-2]
    files_in_folder = os.listdir(studies_folder)
    batches = set()
    for file_name in files_in_folder:
        if "JB.csv" in file_name:
            batch_num = file_name.split('_')[2]
            corresponding_ks_file = f'{type}_batch_{batch_num}_KS.csv'
            if corresponding_ks_file in files_in_folder:
                batches.add(batch_num)

    # Load all batches
    j_ls, k_ls = [], []
    for batch_num in sorted(batches):
        jb_file = os.path.join(studies_folder, f'{type}_batch_{batch_num}_JB.csv')
        ks_file = os.path.join(studies_folder, f'{type}_batch_{batch_num}_KS.csv')
        j_ls.append(pd.read_csv(jb_file))
        k_ls.append(pd.read_csv(ks_file))

    j_df = pd.concat(j_ls, ignore_index=True)
    k_df = pd.concat(k_ls, ignore_index=True)

    return j_df, k_df