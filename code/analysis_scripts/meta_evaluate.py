# Meta-evaluation: align predicted segment scores to the AMI-ME ground-truth
# segmentation and report segment-level Spearman (ρ) and Kendall-Tau (τ).
import json
import numpy as np
from fire import Fire
import os
from scipy.stats import spearmanr, kendalltau
import sys
import copy

script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_path)

sys.path.append("../")
from post_process import get_time_range_of_transcripts


def merge_scores(scores):
    return np.mean(scores)


def score_to_score_distribution(score):
    score_distribution = np.zeros(5, dtype=float)
    for s in score:
        score_distribution[int(s) - 1] += 1
    score_distribution = score_distribution / len(score)
    return score_distribution


def load_gt_from_ami_me(ami_me_path):
    """Build the ground-truth ``pure_score`` structure directly from the released
    AMI-ME dataset (``AMI_ME.json``), so the meta-evaluation reproduces from the
    public dataset alone.

    pure_score format:
        {meeting_id: [overall_score, [[start, end], ...], [segment_score, ...], [score_distribution, ...]]}

    The per-segment score is the mean of the three annotator scores; segment
    timestamps are taken from the first/last utterance (seconds from meeting
    start). AMI-ME has no meeting-level overall annotation, so overall_score is None.
    """
    ami_me = json.load(open(ami_me_path, "r"))
    pure_score = {}
    for meeting_id, meeting in ami_me.items():
        timestamps, segment_scores, segment_score_distribution = [], [], []
        for segment in meeting["segments"]:
            utterances = segment["utterances"]
            timestamps.append([utterances[0]["start"], utterances[-1]["end"]])
            scores = segment["scores"]
            segment_scores.append(merge_scores(scores))
            segment_score_distribution.append(score_to_score_distribution(scores))
        pure_score[meeting_id] = [None, timestamps, segment_scores, segment_score_distribution]
    return pure_score


def prediction_to_pure_score_format(prediction_pure_score, meetingeval_LLM):
    # pure_score format: {meeting_id: [overall_score, [[start, end], ...], [segment_score, ...], segment_score_distribution]}
    new_prediction_pure_score = copy.deepcopy(prediction_pure_score)
    for meeting_id, pure_score_data in prediction_pure_score.items():
        assert meeting_id in meetingeval_LLM
        segments = meetingeval_LLM[meeting_id]["segments"]
        assert len(segments) == len(pure_score_data[1])
        for i, segment in enumerate(segments):
            start, end = get_time_range_of_transcripts(segment)
            new_prediction_pure_score[meeting_id][1][i] = [start, end]
    return new_prediction_pure_score


def align_segmentation(prediction_pure_score, GT_pure_score, truncation=True):
    """
    Align predicted scores onto the GT segmentation using absolute timestamps.
    Both prediction and GT data are FILTERED, keeping only the segments where a
    successful alignment (overlap > 0) occurred.

    The aligned score for a GT segment is the duration-weighted average of the
    scores of all predicted segments that overlap it (weight = overlap duration).
    """
    new_prediction_pure_score = {}
    new_GT_pure_score = {}

    for meeting_id in GT_pure_score:
        if meeting_id not in prediction_pure_score:
            continue

        pred_overall_score, pred_timestamps, pred_scores, pred_distributions = prediction_pure_score[meeting_id]
        gt_overall_score, gt_timestamps, gt_scores, gt_distributions = GT_pure_score[meeting_id]

        # buffers for the filtered results
        filtered_aligned_scores = []
        filtered_aligned_distributions = []

        filtered_gt_timestamps = []
        filtered_gt_scores = []
        filtered_gt_distributions = []

        # iterate over each GT segment (index needed below)
        for i, (gt_start, gt_end) in enumerate(gt_timestamps):
            weighted_score_sum = 0.0
            weighted_dist_sum = np.zeros(len(pred_distributions[0]) if pred_distributions else 5, dtype=float)
            total_overlap_duration = 0.0

            # find all overlapping predicted segments
            for j, (pred_start, pred_end) in enumerate(pred_timestamps):
                overlap_start = max(gt_start, pred_start)
                overlap_end = min(gt_end, pred_end)
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > 0:
                    weighted_score_sum += pred_scores[j] * overlap_duration
                    weighted_dist_sum += np.array(pred_distributions[j]) * overlap_duration
                    total_overlap_duration += overlap_duration

            # keep the segment only when there is a valid overlap
            if total_overlap_duration > 0:
                final_score = weighted_score_sum / total_overlap_duration
                final_dist = (weighted_dist_sum / total_overlap_duration).tolist()

                # 1. the computed aligned prediction score
                filtered_aligned_scores.append(final_score)
                filtered_aligned_distributions.append(final_dist)

                # 2. the GT segment's own data
                filtered_gt_timestamps.append(gt_timestamps[i])
                filtered_gt_scores.append(gt_scores[i])
                filtered_gt_distributions.append(gt_distributions[i])

        # build the new pure_score dict for this meeting from the filtered lists;
        # prediction and GT share the same filtered timestamps
        new_prediction_pure_score[meeting_id] = [pred_overall_score, filtered_gt_timestamps, filtered_aligned_scores, filtered_aligned_distributions]
        new_GT_pure_score[meeting_id] = [gt_overall_score, filtered_gt_timestamps, filtered_gt_scores, filtered_gt_distributions]

    return new_prediction_pure_score, new_GT_pure_score


def meta_evaluation(prediction_pure_score, GT_pure_score, need_align=True, save_path=None):
    """Compute segment-level Spearman (ρ) and Kendall-Tau (τ) between the predicted
    and ground-truth segment scores."""
    meeting_ids = set(prediction_pure_score.keys()) & set(GT_pure_score.keys())
    print(f"Num common meeting: {len(meeting_ids)}, lost {set(GT_pure_score.keys()) - set(prediction_pure_score.keys())}")
    prediction_pure_score = {mid: mdata for mid, mdata in prediction_pure_score.items() if mid in meeting_ids}
    GT_pure_score = {mid: mdata for mid, mdata in GT_pure_score.items() if mid in meeting_ids}

    if need_align:
        prediction_pure_score, GT_pure_score = align_segmentation(prediction_pure_score, GT_pure_score)

    all_pred_segment_scores = []
    all_gt_segment_scores = []
    save_summary = {}

    for mid in sorted(meeting_ids):
        # Segment scores are at index 2 of the pure_score entry.
        all_pred_segment_scores.extend(prediction_pure_score[mid][2])
        all_gt_segment_scores.extend(GT_pure_score[mid][2])
        save_summary[mid] = {"pred": prediction_pure_score[mid][2], "gt": GT_pure_score[mid][2]}

    spearman_segments, _ = spearmanr(all_pred_segment_scores, all_gt_segment_scores)
    kendall_segments, _ = kendalltau(all_pred_segment_scores, all_gt_segment_scores)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(save_summary, f, indent=4)

    return {"segments": {"Spearman": spearman_segments, "Kendall-Tau": kendall_segments}}


def calculate_segmentation_ceiling(prediction_pure_score, GT_pure_score):
    """Estimate the theoretical upper bound on correlation imposed by the predicted
    segmentation alone (paper Table 2 / Appendix H).

    The GT scores are aligned onto the predicted segmentation (the best a perfect
    scorer could do given those boundaries) and then realigned back to the GT
    segmentation; correlating the result against GT isolates the information lost to
    boundary mismatch, independent of scoring quality.
    """
    # Step 1: Create the "ideal prediction" by aligning GT scores to the prediction's segmentation.
    # No deepcopy is needed as align_segmentation does not modify its inputs.
    ideal_prediction_pure_score, _ = align_segmentation(GT_pure_score, prediction_pure_score)

    # Step 2: Now, align the "ideal prediction" back to the original GT segmentation.
    realigned_ideal_prediction, final_GT = align_segmentation(ideal_prediction_pure_score, GT_pure_score)

    # Step 3: Calculate the correlation between the realigned ideal scores and the final GT scores.
    all_pred_segment_scores = []
    all_gt_segment_scores = []

    meeting_ids = set(realigned_ideal_prediction.keys()) & set(final_GT.keys())

    for mid in meeting_ids:
        all_pred_segment_scores.extend(realigned_ideal_prediction[mid][2])
        all_gt_segment_scores.extend(final_GT[mid][2])

    if not all_pred_segment_scores or not all_gt_segment_scores:
        return {"Spearman": 0.0, "Kendall-Tau": 0.0}

    spearman_segments, _ = spearmanr(all_pred_segment_scores, all_gt_segment_scores)
    kendall_segments, _ = kendalltau(all_pred_segment_scores, all_gt_segment_scores)

    return {"Spearman": spearman_segments, "Kendall-Tau": kendall_segments}


def run_evaluation(prediction_dir, prediction_root, ami_me_path, save_dir, subset):
    # Make sure the save directory exists
    if save_dir:
        full_save_dir = os.path.join(prediction_root, save_dir)
        os.makedirs(full_save_dir, exist_ok=True)

    # Ground truth scores/timestamps come straight from the released AMI-ME dataset.
    GT_pure_score = load_gt_from_ami_me(ami_me_path)

    prediction_path = os.path.join(prediction_root, prediction_dir, "pure_score.json")
    meetingeval_LLM_path = os.path.join(prediction_root, prediction_dir, "meetingeval_LLM.json")

    if not os.path.exists(prediction_path):
        print(f"  - Error: File not found, skipping: {prediction_path}")
        return

    prediction_pure_score = json.load(open(prediction_path, "r"))
    meetingeval_LLM = json.load(open(meetingeval_LLM_path, "r"))
    prediction_pure_score = prediction_to_pure_score_format(prediction_pure_score, meetingeval_LLM)

    if subset == "nonscenario":
        # For Non-scenario meetings
        prediction_pure_score = {meeting_id: prediction_pure_score[meeting_id] for meeting_id in prediction_pure_score if meeting_id.startswith("IB")}
        GT_pure_score = {meeting_id: GT_pure_score[meeting_id] for meeting_id in GT_pure_score if meeting_id.startswith("IB")}
    elif subset == "scenario":
        # For Scenario meetings
        prediction_pure_score = {
            meeting_id: prediction_pure_score[meeting_id] for meeting_id in prediction_pure_score if not meeting_id.startswith("IB")
        }
        GT_pure_score = {meeting_id: GT_pure_score[meeting_id] for meeting_id in GT_pure_score if not meeting_id.startswith("IB")}

    need_align = "GTseg" not in prediction_dir

    meta_evaluation_result = meta_evaluation(
        prediction_pure_score, GT_pure_score, need_align=need_align, save_path=os.path.join(full_save_dir, "meta_intermediate_scores.json")
    )

    seg_res = meta_evaluation_result["segments"]
    print(f"Segments Spearman: {seg_res['Spearman']:.4f}")
    print(f"Segments Kendall: {seg_res['Kendall-Tau']:.4f}")

    # Theoretical upper bound given the predicted segmentation (paper Table 2 / Appendix H).
    print("\n--- Segmentation Ceiling Evaluation ---")
    ceiling_result = calculate_segmentation_ceiling(prediction_pure_score, GT_pure_score) if need_align else {"Spearman": 1.0, "Kendall-Tau": 1.0}
    print("This is the theoretical maximum correlation achievable with the given segmentation.")
    print(
        "(Spearman, Kendall-Tau)",
        round(ceiling_result["Spearman"], 4),
        round(ceiling_result["Kendall-Tau"], 4),
    )

    if save_dir:
        json.dump(
            {"segments": meta_evaluation_result["segments"], "ceiling": ceiling_result},
            open(os.path.join(full_save_dir, "score_meta_evaluation.json"), "w"),
        )


def main(prediction_dir, save_dir="evaluation", subset="all", prediction_root="../results", ami_me_path="../../AMI_ME.json"):
    """Meta-evaluate the predicted segment effectiveness scores against AMI-ME.

    Args:
        prediction_dir: name of the run folder under ``prediction_root`` that
            holds ``pure_score.json`` and ``meetingeval_LLM.json``.
        save_dir: sub-directory (inside the run folder) to write metrics into.
        subset: ``all``, ``scenario``, or ``nonscenario`` (non-scenario = the 4
            ``IB``-prefixed film/office meetings).
        prediction_root: root that contains the run folders (default ``../results``).
        ami_me_path: path to the released AMI-ME dataset used as ground truth.
            Default points at ``AMI_ME.json`` at the repository root.
    """
    run_evaluation(
        prediction_dir=prediction_dir,
        prediction_root=prediction_root,
        ami_me_path=ami_me_path,
        save_dir=save_dir,
        subset=subset,
    )


if __name__ == "__main__":
    Fire(main)
