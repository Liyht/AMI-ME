#!/bin/bash
#
# Batch meta-evaluation of effectiveness scores across every run under ./results.
# For each run folder it computes the segment-level Spearman / Kendall correlation
# against the AMI-ME ground truth (see analysis_scripts/meta_evaluate.py).
#
# Set SUBSET to "all", "scenario", or "nonscenario".

TARGET_DIR="./results"
SUBSET="all"

for dir in ${TARGET_DIR}/*/; do
  results_subfolder=$(basename "${dir}")

  # Skip intermediate stage folders (segmentation / objective only).
  case "$results_subfolder" in
    *_obj | *_seg)
      continue
      ;;
    *)
      echo "Processing folder: ${results_subfolder}"
      python ./analysis_scripts/meta_evaluate.py --prediction_dir "$results_subfolder" --save_dir "$results_subfolder/evaluation_$SUBSET" --subset $SUBSET
      echo ""
      ;;
  esac
done
