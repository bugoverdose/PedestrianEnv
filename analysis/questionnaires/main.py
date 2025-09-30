import os

import pandas as pd

DATAFILE = "qualtrics.csv"
RAW_SUBJECT_ID_COLNAME = "SubjID_4"

def load_survey_results(pilot = False):
    # NOTE: download from Qualtrics (https://snuss1.qualtrics.com/responses/#/surveys/SV_a9RzRm8sDPWWh8y)
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), DATAFILE))
    df[RAW_SUBJECT_ID_COLNAME] = pd.to_numeric(df[RAW_SUBJECT_ID_COLNAME], errors="coerce")
    if pilot:
        df = df[(900 <= df[RAW_SUBJECT_ID_COLNAME]) & (df[RAW_SUBJECT_ID_COLNAME] < 1000)]
    else:
        df = df[(df[RAW_SUBJECT_ID_COLNAME] >= 1000)]

    print(f"\nTotal Subjects: {len(df)} ({df[RAW_SUBJECT_ID_COLNAME].astype(int).tolist()})")
    return pd.DataFrame({
        "subj_id": df[RAW_SUBJECT_ID_COLNAME].astype(int),
        "age": _calculate_age(df),
        "gender": df["demo_1"].astype(int), # NOTE: let's not debate about anything regarding this wording
        "school_year": df["demo_3_1"].astype(int),
        **_calculate_BIS(df),
        **_calculate_STAI(df),
        **_calculate_DOSPERT(df),
        **_calculate_CES_D(df),
        "game_play_time": df["playtime_5"].astype(int),
    })

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
    print(f"reversing BIS_1: {prev_BIS_1} => {df['BIS_1'].to_list()}")

    # 허심양, 오주용, & 김지혜. (2012). 한국판 Barratt 충동성 검사-11 의 신뢰도 및 타당도 연구. 한국심리학회지: 일반, 31(3), 769-782.
    motor_impulsivity_questions = [2, 3, 4, 16, 17, 19, 21, 22, 23, 25, 30] # 운동 충동성 (11)
    nonplanning_impulsivity_questions = [1, 7, 8, 10, 12, 13, 14, 15, 18, 27, 29] # 무계획성 충동성 (11)
    attentional_impulsivity_questions = [5, 6, 9, 11, 20, 24, 26, 28] # 인지 충동성 (8)

    BIS_total = df[all_questions].sum(axis=1)
    BIS_motor = df[[f"BIS_{i}" for i in motor_impulsivity_questions]].sum(axis=1)
    BIS_nonplanning = df[[f"BIS_{i}" for i in nonplanning_impulsivity_questions]].sum(axis=1)
    BIS_attentional = df[[f"BIS_{i}" for i in attentional_impulsivity_questions]].sum(axis=1)
    return {
        "BIS":                         BIS_total, # 30 ~ 120
        "BIS_motor":       BIS_motor, # 11 ~ 44
        "BIS_nonplanning": BIS_nonplanning, # 11 ~ 44
        "BIS_attentional": BIS_attentional, # 8 ~ 32
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
    print(f"reversing STAI_1: {prev_STAI_1} => {df['STAI_1'].to_list()}")

    STAI_total = df[all_questions].sum(axis=1)
    STAI_S = df[[f"STAI_{i + 1}" for i in range(20)]].sum(axis=1)
    STAI_T = df[[f"STAI_{i + 21}" for i in range(20)]].sum(axis=1)
    return {
        "STAI":   STAI_total, # 40 ~ 160
        "STAI_S": STAI_S, # 20 ~ 80
        "STAI_T": STAI_T, # 20 ~ 80
    }

def _calculate_DOSPERT(df):
    all_questions = [f"DOSPERT_{i+1}" for i in range(30)]
    df[all_questions] = df[all_questions].astype(int)

    # Hong et al (2010). BART와 K-DOSPERT로 알아본 한국인의 위험감수경향
    ethical_questions = [6, 9, 10, 16, 29, 30] # 윤리, ETH (6)
    financial_questions = [3, 4, 8, 12, 14, 18] # 경제, FIN (6)
    health_safety_questions = [5, 15, 17, 20, 23, 26] # 건강/안전, HEA (6)
    recreational_questions = [2, 11, 13, 19, 24, 25] # 오락, REC (6)
    social_questions = [1, 7, 21, 22, 27, 28] # 사회성, SOC (6)
    assert len(set(ethical_questions + financial_questions + health_safety_questions + recreational_questions + social_questions)) == 30

    DOSPERT_total = df[all_questions].sum(axis=1)
    DOSPERT_ethical = df[[f"DOSPERT_{i}" for i in ethical_questions]].sum(axis=1)
    DOSPERT_financial = df[[f"DOSPERT_{i}" for i in financial_questions]].sum(axis=1)
    DOSPERT_health_safety = df[[f"DOSPERT_{i}" for i in health_safety_questions]].sum(axis=1)
    DOSPERT_recreational = df[[f"DOSPERT_{i}" for i in recreational_questions]].sum(axis=1)
    DOSPERT_social = df[[f"DOSPERT_{i}" for i in social_questions]].sum(axis=1)
    return {
        "DOSPERT":     DOSPERT_total, # 30 ~ 150
        "DOSPERT_ETH": DOSPERT_ethical, # 6 ~ 30
        "DOSPERT_FIN": DOSPERT_financial, # 6 ~ 30
        "DOSPERT_HEA": DOSPERT_health_safety, # 6 ~ 30
        "DOSPERT_REC": DOSPERT_recreational, # 6 ~ 30
        "DOSPERT_SOC": DOSPERT_social, # 6 ~ 30
    }

def _calculate_CES_D(df):
    all_questions = [f"CES-D_{i+1}" for i in range(20)]
    df[all_questions] = df[all_questions].astype(int)

    prev_CES_D_1 = df["CES-D_4"].to_list()
    reverse_scored_questions = [4, 8, 12, 16]
    for q in reverse_scored_questions:
        _apply_reverse_scoring(df, f"CES-D_{q}", min_score = 1, max_score = 4)
    print(f"reversing CES-D_4: {prev_CES_D_1} => {df['CES-D_4'].to_list()}")

    CES_D_total = df[all_questions].sum(axis=1)
    return {
        "CES_D": CES_D_total, # 20 ~ 80
        # NOTE: only use the total score since it's used only for screening purposes.
        # should also consider K-IGDS_1 ~ K-IGDS_27, K-IAT_1 ~ K-IAT_20 for screening.
    }

def _apply_reverse_scoring(df, col, min_score = 1, max_score = 4):
    df[col] = max_score + min_score - df[col]
