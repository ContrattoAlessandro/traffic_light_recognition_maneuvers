#!/usr/bin/env bash
# Download OpenLane-V2 subset_A train/val image tars one at a time, keep only the
# front camera (ring_front_center), then delete each tar. Google Drive often
# rate-limits a file ("Too many users have viewed or downloaded this file"), so
# blocked parts are skipped and retried in later rounds. Re-running skips parts
# already marked done.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GDOWN="$ROOT/.venv/Scripts/gdown"
DL="$ROOT/data/download"
OUT="$ROOT/data/OpenLane-V2/subset_A"
ROUNDS=${ROUNDS:-48}
WAIT_S=${WAIT_S:-1800}
mkdir -p "$DL" "$OUT"

# Google Drive ids from the official data/README.md (image_0..6 train, image_7 val)
IDS=(
  1jio4Gj3dNlXmSzebO6D7Uy5oz4EaTNTq
  1IgnvZ2UljL49AzNV6CGNGFLQo6tjNFJq
  1ViEsK5hukjMGfOm_HrCiQPkGArWrT91o
  1r3NYauV0JIghSmEihTxto0MMoyoh4waK
  1aBe5yxNBew11YRRu-srQNwc5OloyKP4r
  1Or-Nmsq4SU24KNe-cn9twVYVprYPUd_y
  1mSWU-2nMzCO5PGF7yF9scoPntWl7ItfZ
  19N5q-zbjE2QWngAT9xfqgOR3DROTAln0
)

fetch_part() {
  local k=$1 tarf="$DL/OpenLane-V2_subset_A_image_$1.tar"
  echo "[image_$k] download $(date +%T)"
  "$GDOWN" "${IDS[$k]}" -O "$tarf" 2>&1 | tr '\r' '\n' | grep -v '%|' | grep -E 'Too many|Failed|To:' | head -2
  if ! { [ -s "$tarf" ] && tar tf "$tarf" >/dev/null 2>&1; }; then
    echo "[image_$k] not available now"; rm -f "$tarf"; return 1
  fi
  echo "[image_$k] extracting ring_front_center $(date +%T)"
  tar xf "$tarf" -C "$OUT" --wildcards '*/ring_front_center/*' || return 1
  rm -f "$tarf" && touch "$DL/.image_$k.done"
  echo "[image_$k] done: $(find "$OUT" -path '*ring_front_center*' -name '*.jpg' | wc -l) front images total"
}

for round in $(seq 1 "$ROUNDS"); do
  pending=0
  for k in "${!IDS[@]}"; do
    [ -f "$DL/.image_$k.done" ] && continue
    fetch_part "$k" || pending=$((pending + 1))
  done
  if [ "$pending" -eq 0 ]; then echo "ALL DONE $(date +%T)"; exit 0; fi
  echo "round $round: $pending part(s) pending, next try in ${WAIT_S}s"
  sleep "$WAIT_S"
done
echo "GAVE UP with parts pending $(date +%T)"; exit 1
