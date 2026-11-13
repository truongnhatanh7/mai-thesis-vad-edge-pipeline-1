#!/bin/bash
# Extract all non-test members (resumable: verified files are skipped), then verify and update README status.
set -u
cd /Users/atruong/thesis-data/ucf-crime
PY=/Users/atruong/Documents/thesis/experiments/.venv/bin/python
echo "start $(date -u +%FT%TZ)"
for attempt in 1 2 3; do
  $PY tools/extract_members.py raw/remaining_members.txt UCF_Crimes/Videos raw/extract_rest.tsv 6
  missing=$(while read -r r; do [ -f "UCF_Crimes/Videos/$r" ] || echo "$r"; done < raw/remaining_members.txt | wc -l | tr -d ' ')
  echo "attempt $attempt done $(date -u +%FT%TZ) missing=$missing"
  [ "$missing" = 0 ] && break
done
n_mp4=$(find UCF_Crimes/Videos -name '*.mp4' | wc -l | tr -d ' ')
n_train=$(while read -r r; do [ -f "UCF_Crimes/Videos/$r" ] && echo x; done < UCF_Crimes/Anomaly_Detection_splits/Anomaly_Train.txt | wc -l | tr -d ' ')
bytes=$(du -sk UCF_Crimes/Videos | cut -f1)
echo "n_mp4=$n_mp4 n_train=$n_train kB=$bytes end $(date -u +%FT%TZ)"
if [ "$n_mp4" = 1950 ] && [ "$n_train" = 1610 ]; then
  status="COMPLETE"
else
  status="PARTIAL"
fi
sed -i '' "s/^STATUS: .*/STATUS: $status (test set COMPLETE; all videos: $n_mp4\/1950 present, train $n_train\/1610; auto-updated $(date -u +%FT%TZ))/" README.md
echo "README status -> $status"
