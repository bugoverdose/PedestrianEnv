import pandas as pd

from util import data_dir

def load_behavioral_task_results(subjIds):
    dfs = []
    for subjId in subjIds:
        DDT_mean_log_k, CRA_mean_alpha = load_behavioral_task_results_ind(subjId)
        dfs.append(pd.DataFrame({
            "subj_id": [subjId],
            "DDT_mean_log_k": [DDT_mean_log_k],
            "CRA_mean_alpha": [CRA_mean_alpha],
        }))
    return pd.concat(dfs, ignore_index=True)

def load_behavioral_task_results_ind(subjId):
    DDT_mean_log_k, CRA_mean_alpha = None, None
    subj_data_dir = data_dir(subjId)
    for file in subj_data_dir.iterdir():
        if file.is_dir(): continue
        if file.name.startswith("DDT"):
            if DDT_mean_log_k is not None:
                raise Exception("multiple DDT results found")
            df = pd.read_csv(file, sep="\t")
            DDT_mean_log_k = df.iloc[-1]["mean_log_k"]
        if file.name.startswith("CRA"):
            if CRA_mean_alpha is not None:
                raise Exception("multiple CRA results found")
            df = pd.read_csv(file, sep="\t")
            CRA_mean_alpha = df.iloc[-1]["mean_alpha"]
    return DDT_mean_log_k, CRA_mean_alpha
