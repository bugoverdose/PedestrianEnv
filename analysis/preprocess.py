from questionnaires import load_survey_results
from pedestrian_stats import build_pedestrian_task_results
from behavior_tasks import build_behavioral_task_results

def load_and_save(pilot):
    df = load_survey_results(pilot = pilot)
    subj_ids = df["subj_id"].to_list()
    if len(subj_ids) == 0: return

    df = build_pedestrian_task_results(df, subj_ids)
    df = build_behavioral_task_results(df, subj_ids)
    if pilot:
        df.to_csv("data_pilot.csv", index=False)
    else:
        df.to_csv("data.csv", index=False)

if __name__ == "__main__":
    load_and_save(pilot=True)
    load_and_save(pilot=False)
