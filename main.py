import os
import time
from datetime import datetime

import argparse

from run_pedestrian import play_game

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run for experiment")
    arg_parser.add_argument('--subjId', type=int, default=1, help='subject ID')
    arg_parser.add_argument('--sessionId', type=int, default=0, help='session ID')
    args = arg_parser.parse_args()

    base_dir = f"data/{args.subjId}"
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, f"README_{args.sessionId}.md"), 'w') as f:
        timestamp = time.time()
        dt = datetime.fromtimestamp(timestamp)
        f.write(f"Subject ID: {args.subjId}\n"
                + f"Session ID: {args.sessionId}\n"
                + f"Play start time: {dt.year}.{dt.month}.{dt.day} {dt.hour}:{dt.minute}:{dt.second}")

    # practice session: 5 minutes
    # main session: 20 minutes
    max_seconds = 300 if args.sessionId == 0 else 1200
    play_game(base_dir=base_dir, session_id=args.sessionId, max_seconds=max_seconds)
