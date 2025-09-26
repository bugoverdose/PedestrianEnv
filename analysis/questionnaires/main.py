import pandas as pd

RAW_SUBJECT_ID_COLNAME = "SubjID_4"

def preprocess(test = False):
    # NOTE: download from Qualtrics (https://snuss1.qualtrics.com/responses/#/surveys/SV_a9RzRm8sDPWWh8y)
    df = pd.read_csv("qualtrics.csv")
    df[RAW_SUBJECT_ID_COLNAME] = pd.to_numeric(df[RAW_SUBJECT_ID_COLNAME], errors="coerce")
    if test:
        df = df[(df[RAW_SUBJECT_ID_COLNAME] < 1000)]
    else:
        df = df[(df[RAW_SUBJECT_ID_COLNAME] >= 1000)]

    base = {
        "subj_id":     df[RAW_SUBJECT_ID_COLNAME].astype(int),
        "gender":      df["demo_1"].astype(int), # NOTE: let's not debate about anything regarding this wording
        "birth_year":  df["demo_2_1"].astype(int),
        "birth_month": df["demo_2_2"].astype(int),
        "school_year": df["demo_3_1"].astype(int),
        "game_play_time": df["playtime_5"].astype(int),
    }
    bis     = {f"BIS_{i}":     df[f"BIS_{i+1}"].astype(int)     for i in range(30)}
    stai_s  = {f"STAI-S_{i}":  df[f"STAI-S_{i+1}"].astype(int)  for i in range(20)}
    stai_t  = {f"STAI-T_{i}":  df[f"STAI-T_{i+1}"].astype(int)  for i in range(20)}
    dospert = {f"DOSPERT_{i}": df[f"DOSPERT_{i+1}"].astype(int) for i in range(30)}
    cesd    = {f"CES-D_{i}":   df[f"CES-D_{i+1}"].astype(int)   for i in range(20)}
    kigds   = {f"K-IGDS_{i}":  df[f"K-IGDS_{i+1}"].astype(int)  for i in range(27)}
    kiat    = {f"K-IAT_{i}":   df[f"K-IAT_{i+1}"].astype(int)   for i in range(20)}
    df = pd.DataFrame({**base, **bis, **stai_s, **stai_t, **dospert, **cesd, **kigds, **kiat})
    print(df)
    print(f"Total Subjects: {len(df)} ({df['subj_id'].tolist()})")
    return df

if __name__ == "__main__":
    # preprocess()
    preprocess(test = True)
