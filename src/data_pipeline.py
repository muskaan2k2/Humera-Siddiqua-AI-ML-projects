import os
import json
import pandas as pd
from tqdm import tqdm


def get_json_files(raw_dir="data/raw"):
    files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".json")]
    print(f"Total matches found: {len(files)}")
    return files


def parse_match(filepath):
    with open(filepath, "r") as f:
        match = json.load(f)

    rows = []
    meta = match.get("info", {})
    match_id = os.path.basename(filepath).replace(".json", "")
    match_type = meta.get("match_type", "unknown")
    venue = meta.get("venue", "unknown")
    teams = meta.get("teams", ["team1", "team2"])

    if match_type not in ["T20", "ODI"]:
        return []

    outcome = meta.get("outcome", {})
    winner = outcome.get("winner", None)
    if winner is None:
        return []

    max_overs = 20 if match_type == "T20" else 50
    total_balls = max_overs * 6

    innings = match.get("innings", [])
    if len(innings) < 2:
        return []

    # Calculate 1st innings total
    first_innings_total = 0
    for over in innings[0].get("overs", []):
        for delivery in over.get("deliveries", []):
            first_innings_total += delivery.get("runs", {}).get("total", 0)

    target = first_innings_total + 1

    # 2nd innings
    inning_data = innings[1]
    batting_team = inning_data.get("team", "unknown")
    label = 1 if batting_team == winner else 0

    runs_so_far = 0
    wickets_fallen = 0
    balls_bowled = 0
    recent_runs = []

    for over_data in inning_data.get("overs", []):
        over_number = over_data.get("over", 0)
        for delivery in over_data.get("deliveries", []):
            balls_bowled += 1
            runs_this_ball = delivery.get("runs", {}).get("total", 0)
            runs_so_far += runs_this_ball

            if "wickets" in delivery:
                wickets_fallen += 1

            recent_runs.append(runs_this_ball)
            if len(recent_runs) > 12:
                recent_runs.pop(0)

            runs_remaining = target - runs_so_far
            balls_remaining = total_balls - balls_bowled
            wickets_remaining = 10 - wickets_fallen

            current_rr = round((runs_so_far / balls_bowled * 6), 4) if balls_bowled > 0 else 0.0
            required_rr = round((runs_remaining / balls_remaining * 6), 4) if balls_remaining > 0 else 999.0

            rows.append({
                "match_id": match_id,
                "match_type": match_type,
                "venue": venue,
                "batting_team": batting_team,
                "over": over_number,
                "balls_bowled": balls_bowled,
                "balls_remaining": balls_remaining,
                "runs_scored": runs_so_far,
                "runs_remaining": runs_remaining,
                "wickets_fallen": wickets_fallen,
                "wickets_remaining": wickets_remaining,
                "target": target,
                "current_run_rate": current_rr,
                "required_run_rate": required_rr,
                "recent_12_balls_runs": sum(recent_runs),
                "label": label,
            })

    return rows


def build_dataset(raw_dir="data/raw", output_path="data/processed/ball_by_ball.csv"):
    files = get_json_files(raw_dir)
    all_rows = []
    skipped = 0

    print("Loading JSON matches...")
    for filepath in tqdm(files):
        rows = parse_match(filepath)
        if rows:
            all_rows.extend(rows)
        else:
            skipped += 1

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved at: {output_path}")
    print(f"Total rows: {len(df)}")


if __name__ == "__main__":
    build_dataset()