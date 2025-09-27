from questionnaires import load_survey_results
from pedestrian_stats import build_pedestrian_task_results
from behavior_tasks import build_behavioral_task_results

DATAFILE = "data.csv"

if __name__ == "__main__":
    df = load_survey_results(test = True)
    # TODO: 데이터 부족으로 인한 임시방편. 향후 현실적인 데이터로 대체.
    df.loc[3, "subj_id"] = 502
    df.loc[4, "subj_id"] = 503
    subj_ids = df["subj_id"].to_list()

    df = build_pedestrian_task_results(df, subj_ids)
    df = build_behavioral_task_results(df, subj_ids)

    df.to_csv(DATAFILE, index=False)
