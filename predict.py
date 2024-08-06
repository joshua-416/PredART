# General imports
import argparse
import os
import numpy as np
import pandas as pd
import joblib
import glob
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors


# Do bootstrap aggregating for all .csv results in a result folder and generates model performance and ensemble result file.
def bootstrap_aggregating(dir):
    split_results = glob.glob(os.path.join(dir, '*.csv'))
    predval_dict = {}
    print("Do bootstrap aggregating for %d models.............." % (len(split_results)))
    for result in split_results:
        df = pd.read_csv(result)
        smiles_list = df.iloc[:,0].tolist()
        pred_list = df.iloc[:,1].tolist()
        for idx, smi in enumerate(smiles_list):
            if smi in predval_dict:
                predval_dict[smi].append(float(pred_list[idx]))
            else:
                predval_dict[smi] = [float(pred_list[idx])]

    print("Writing prediction result file....")

    df_out = pd.DataFrame()
    smiles = []
    pred_ensemble = []
    std_ensemble = []

    for key, values in predval_dict.items():
        avg_pred = np.mean(np.array(values))
        std_pred = np.std(np.array(values))
        smiles.append(key)
        pred_ensemble.append(avg_pred)
        std_ensemble.append(std_pred)

    df_out['smiles'] = smiles
    df_out['pred'] = pred_ensemble
    df_out['std'] = std_ensemble

    print("Done")
    return df_out


def smiles_to_feature(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.array(fp, dtype=float)

ROOT_DIR = './'

# Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--test', type=str,
                    help='Test data .csv file')
parser.add_argument('--models', type=str,
                    help='Path for trained model')


args = parser.parse_args().__dict__
#{'dataset': 'uspto_50k', 'epochs': 150, ...}


# Read test .csv file and featurization for test set
df_test = pd.read_csv(args['test'])
test_filename = args["test"].split("/")[-1].split('.')[0]
modelname = args['models'].split('/')[-1].split('-')[-1]

test_smiles = df_test['smiles'].tolist()
test_features = np.stack([smiles_to_feature(s) for s in tqdm(test_smiles)])

bagging_size = len(glob.glob(os.path.join(args['models'], '*.pkl')))
print(f'Total {bagging_size} models are loaded for the test of {test_filename + ".csv"}')


for idx in tqdm(range(bagging_size)):
    loaded_model = joblib.load(os.path.join(args['models'], 'model_split_'+str(idx+1)+'.pkl'))
    test_preds = loaded_model.predict(test_features)

    test_result = pd.DataFrame()
    test_result['smiles'] = test_smiles
    test_result['pred'] = test_preds

    if not os.path.exists(os.path.join(args['models'], test_filename)):
        os.makedirs(os.path.join(args['models'], test_filename))

    test_result.to_csv(os.path.join(args['models'], test_filename, f'test_results_split_{idx+1}.csv'), index=False)


ensemble_result = bootstrap_aggregating(os.path.join(args['models'], test_filename))
ensemble_result.to_csv(os.path.join(ROOT_DIR, test_filename+'_ensemble_' + str(bagging_size)+ '-' + modelname + '.csv'), index=False)










