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

        **_calculate_BIS(df),
        "game_play_time":  df["playtime_5"].astype(int),
        # **expand("STAI-S", 20),
        # **expand("STAI-T", 20),
        # **expand("DOSPERT", 30),
        # **expand("CES-D", 20),
        # **expand("K-IGDS", 27),
        # **expand("K-IAT", 20),
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

def _calculate_BIS(df):
    all_questions = [f"BIS_{i+1}" for i in range(30)]
    df[all_questions] = df[all_questions].astype(int)

    # 이소라, 이원혜, 박정수, 김설민, 김종우, 심재현 (2012). 한국판 Barratt Impulsiveness Scale-11-Revised의 신뢰도 및 타당도 연구 ; 일반성인집단을 중심으로
    reverse_scored_questions = [1, 7, 8, 9, 10, 12, 13, 15, 20, 29, 30]
    for q in reverse_scored_questions:
        apply_reverse_scoring(df, f"BIS_{q}", min_score = 1, max_score = 4)

    # 허심양, 오주용, & 김지혜. (2012). 한국판 Barratt 충동성 검사-11 의 신뢰도 및 타당도 연구. 한국심리학회지: 일반, 31(3), 769-782.
    motor_impulsivity_questions = [2, 3, 4, 16, 17, 19, 21, 22, 23, 25, 30] # 운동 충동성 (11)
    nonplanning_impulsivity_questions = [1, 7, 8, 10, 12, 13, 14, 15, 18, 27, 29] # 무계획성 충동성 (11)
    attentional_impulsivity_questions = [5, 6, 9, 11, 20, 24, 26, 28] # 인지 충동성 (8)

    BIS_total = df[all_questions].sum(axis=1)
    BIS_motor_impulsivity = df[[f"BIS_{i}" for i in motor_impulsivity_questions]].sum(axis=1)
    BIS_nonplanning_impulsivity = df[[f"BIS_{i}" for i in nonplanning_impulsivity_questions]].sum(axis=1)
    BIS_attentional_impulsivity = df[[f"BIS_{i}" for i in attentional_impulsivity_questions]].sum(axis=1)
    return {
        "BIS_total":                   BIS_total, # 30 ~ 120
        "BIS_motor_impulsivity":       BIS_motor_impulsivity, # 11 ~ 44
        "BIS_nonplanning_impulsivity": BIS_nonplanning_impulsivity, # 11 ~ 44
        "BIS_attentional_impulsivity": BIS_attentional_impulsivity, # 8 ~ 32
    }

def apply_reverse_scoring(df, col, min_score = 1, max_score = 4):
    df[col] = max_score + min_score - df[col]

if __name__ == "__main__":
    # df = preprocess()
    df = preprocess(test = True)
    demographics(df)
