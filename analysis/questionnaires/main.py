import pandas as pd

DATAFILE = "qualtrics.csv"
RAW_SUBJECT_ID_COLNAME = "SubjID_4"

def preprocess(test = False):
    # NOTE: download from Qualtrics (https://snuss1.qualtrics.com/responses/#/surveys/SV_a9RzRm8sDPWWh8y)
    df = pd.read_csv(DATAFILE)
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
        **_calculate_STAI(df),
        "game_play_time":  df["playtime_5"].astype(int),
        # **expand("DOSPERT", 30),
        # **expand("CES-D", 20),
        # **expand("K-IGDS", 27),
        # **expand("K-IAT", 20),
    })
    # print(df)
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

def _calculate_BIS(df):
    all_questions = [f"BIS_{i+1}" for i in range(30)]
    df[all_questions] = df[all_questions].astype(int)

    # 이소라, 이원혜, 박정수, 김설민, 김종우, 심재현 (2012). 한국판 Barratt Impulsiveness Scale-11-Revised의 신뢰도 및 타당도 연구 ; 일반성인집단을 중심으로
    prev_BIS_1 = df["BIS_1"].to_list()
    reverse_scored_questions = [1, 7, 8, 9, 10, 12, 13, 15, 20, 29, 30]
    for q in reverse_scored_questions:
        _apply_reverse_scoring(df, f"BIS_{q}", min_score = 1, max_score = 4)
    print(f"BIS_1: {prev_BIS_1} => {df['BIS_1'].to_list()}")
    # 허심양, 오주용, & 김지혜. (2012). 한국판 Barratt 충동성 검사-11 의 신뢰도 및 타당도 연구. 한국심리학회지: 일반, 31(3), 769-782.
    motor_impulsivity_questions = [2, 3, 4, 16, 17, 19, 21, 22, 23, 25, 30] # 운동 충동성 (11)
    nonplanning_impulsivity_questions = [1, 7, 8, 10, 12, 13, 14, 15, 18, 27, 29] # 무계획성 충동성 (11)
    attentional_impulsivity_questions = [5, 6, 9, 11, 20, 24, 26, 28] # 인지 충동성 (8)

    BIS_total = df[all_questions].sum(axis=1)
    BIS_motor_impulsivity = df[[f"BIS_{i}" for i in motor_impulsivity_questions]].sum(axis=1)
    BIS_nonplanning_impulsivity = df[[f"BIS_{i}" for i in nonplanning_impulsivity_questions]].sum(axis=1)
    BIS_attentional_impulsivity = df[[f"BIS_{i}" for i in attentional_impulsivity_questions]].sum(axis=1)
    return {
        "BIS":                         BIS_total, # 30 ~ 120
        "BIS_motor_impulsivity":       BIS_motor_impulsivity, # 11 ~ 44
        "BIS_nonplanning_impulsivity": BIS_nonplanning_impulsivity, # 11 ~ 44
        "BIS_attentional_impulsivity": BIS_attentional_impulsivity, # 8 ~ 32
    }

def _calculate_STAI(df):
    stai_s = [f"STAI-S_{i+1}" for i in range(20)]
    stai_t = [f"STAI-T_{i+1}" for i in range(20)]
    all_questions = [f"STAI_{i+1}" for i in range(40)]
    df[all_questions] = df[stai_s + stai_t].astype(int)

    # Seok, Hamid, Mutang, and Ismail (2018). Psychometric Properties of the State-Trait Anxiety Inventory (Form Y) among Malaysian University Students
    # The reversed score items
    # - 1, 2, 5, 8, 10, 11, 15, 16, 19, 20 (State Anxiety Scale)
    # - 21, 23, 26, 27, 30, 33, 34, 36, 39 (Trait Anxiety Scale)
    prev_STAI_1 =  df["STAI_1"].to_list()
    reverse_scored_questions = [1, 2, 5, 8, 10, 11, 15, 16, 19, 20, 21, 23, 26, 27, 30, 33, 34, 36, 39]
    for q in reverse_scored_questions:
        _apply_reverse_scoring(df, f"STAI_{q}", min_score = 1, max_score = 4)
    print(f"STAI_1: {prev_STAI_1} => {df['STAI_1'].to_list()}")

    STAI_total = df[all_questions].sum(axis=1)
    STAI_S = df[[f"STAI_{i + 1}" for i in range(20)]].sum(axis=1)
    STAI_T = df[[f"STAI_{i + 21}" for i in range(20)]].sum(axis=1)
    return {
        "STAI":   STAI_total, # 40 ~ 160
        "STAI_S": STAI_S, # 20 ~ 80
        "STAI_T": STAI_T, # 20 ~ 80
    }

def group_analysis(df):
    print(f"\nTotal Subjects: {len(df)} ({df['subj_id'].tolist()})")

    print("\n################################ Demographics ################################")
    age = df["age"]
    print(f"Age: mean = {age.mean()}, std = {age.std()} (min={age.min()} ~ max={age.max()})")
    gender = df["gender"]
    print(f"Gender Ratio: {len(df[gender == 2])} males & {len(df[gender == 1])} females")

    print("\n#################################### BIS #####################################")
    for col in ["BIS", "BIS_motor_impulsivity", "BIS_nonplanning_impulsivity", "BIS_attentional_impulsivity"]:
        print(f"{col}: mean = {df[col].mean()}, std = {df[col].std()} (min={df[col].min()} ~ max={df[col].max()})")

    print("\n#################################### STAI ####################################")
    for col in ["STAI", "STAI_S", "STAI_T"]:
        print(f"{col}: mean = {df[col].mean()}, std = {df[col].std()} (min={df[col].min()} ~ max={df[col].max()})")

def _apply_reverse_scoring(df, col, min_score = 1, max_score = 4):
    df[col] = max_score + min_score - df[col]

if __name__ == "__main__":
    # df = preprocess()
    df = preprocess(test = True)
    group_analysis(df)
