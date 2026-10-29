#!/usr/bin/env bash
# Extract OpenLane-V2 train/val info and front-camera images from the official tars.
# Usage: tools/extract_front.sh <tar_dir> <subset: A|B>
# The tars are left in place; parts already extracted are skipped via marker files.
set -eu
SRC="$1"; SUBSET="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/OpenLane-V2/subset_$SUBSET"
case "$SUBSET" in
  A) CAM=ring_front_center ;;
  B) CAM=CAM_FRONT ;;
  *) echo "subset must be A or B"; exit 1 ;;
esac
mkdir -p "$OUT"

if [ ! -f "$OUT/.info.done" ]; then
  echo "[$SUBSET info] $(date +%T)"
  tar xf "$SRC/OpenLane-V2_subset_${SUBSET}_info.tar" -C "$OUT" --wildcards 'train/*' 'val/*' './train/*' './val/*' 2>/dev/null \
    || tar xf "$SRC/OpenLane-V2_subset_${SUBSET}_info.tar" -C "$OUT" --wildcards '*train/*' '*val/*'
  touch "$OUT/.info.done"
fi
for k in 0 1 2 3 4 5 6 7; do  # image_8 is the test split, not needed
  [ -f "$OUT/.image_$k.done" ] && continue
  echo "[$SUBSET image_$k] $(date +%T)"
  tar xf "$SRC/OpenLane-V2_subset_${SUBSET}_image_$k.tar" -C "$OUT" --wildcards "*/$CAM/*"
  touch "$OUT/.image_$k.done"
done
echo "[$SUBSET] DONE $(date +%T)"
