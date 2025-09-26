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

    def expand(questionnaire_name, question_count):
        return {f"raw_{questionnaire_name}_{i+1}": df[f"{questionnaire_name}_{i+1}"].astype(int) for i in range(question_count)}
    df = pd.DataFrame({
        "subj_id":         df[RAW_SUBJECT_ID_COLNAME].astype(int),
        "age":             _calculate_age(df),
        "gender":          df["demo_1"].astype(int), # NOTE: let's not debate about anything regarding this wording
        "school_year":     df["demo_3_1"].astype(int),

        "game_play_time":  df["playtime_5"].astype(int),
        **expand("BIS", 30),
        **expand("STAI-S", 20),
        **expand("STAI-T", 20),
        **expand("DOSPERT", 30),
        **expand("CES-D", 20),
        **expand("K-IGDS", 27),
        **expand("K-IAT", 20),
    })
    print(df)
    return df

def _calculate_age(df):
    response_date = pd.to_datetime(df["EndDate"], errors='coerce') # format: 2025-09-26 14:38:19
    response_year = response_date.dt.year.astype(int)
    response_month = response_date.dt.month.astype(int)
    birth_year = df["demo_2_1"].astype(int)
    birth_month = df["demo_2_2"].astype(int)
    age = response_year - birth_year
    age[response_month < birth_month] -= 1
    return age.astype(int)

def demographics(df):
    print(f"Total Subjects: {len(df)} ({df['subj_id'].tolist()})")
    age = df["age"]
    print(f"Age: mean = {age.mean()} ({age.min()} ~ {age.max()})")
    gender = df["gender"]
    print(f"Gender: {len(df[gender == 2])} males & {len(df[gender == 1])} females")

if __name__ == "__main__":
    # df = preprocess()
    df = preprocess(test = True)
    demographics(df)
