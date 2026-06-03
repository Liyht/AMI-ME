"""Rebuild the pipeline inputs from the released AMI-ME dataset (`AMI_ME.json`).

Writes two files:
- `data/AMI/meetingeval.json`: the flat transcript (`meeting`), used by the topic-segmentation and objective-classification stages.
- `data/AMI/meetingeval_GTseg.json`: the ground-truth segmentation in the `meetingeval_LLM.json` structure (`{"meeting", "segments", "topics"}`), feed it to the `eval` task to score against the gold segmentation (`seg_source="gt"` in `vllm_eval.sh`).

Usage:
    python data/build_meetingeval.py
    python data/build_meetingeval.py --ami_me_path PATH --out PATH --gtseg_out PATH
"""

import argparse
import datetime
import json
import os


def s2time(seconds):
    return str(datetime.timedelta(seconds=round(float(seconds))))


def utterance_to_text(utterance):
    return f"[{s2time(utterance['start'])} - {s2time(utterance['end'])}] [{utterance['speaker']}] {utterance['text'].strip()}"


def segment_to_text(segment):
    return "\n".join(utterance_to_text(u) for u in segment["utterances"])


def build_meetingeval(ami_me_path):
    ami_me = json.load(open(ami_me_path, "r"))
    meetingeval = {}
    for meeting_id, meeting in ami_me.items():
        lines = [utterance_to_text(u) for seg in meeting["segments"] for u in seg["utterances"]]
        meetingeval[meeting_id] = {"meeting": "\n".join(lines)}
    return meetingeval


def build_meetingeval_gtseg(ami_me_path):
    """Build the ground-truth-segmented input, matching the `meetingeval_LLM.json`
    structure produced by `post_process.py --task segmentation`:
        {meeting_id: {"meeting": str, "segments": [str, ...], "topics": [str, ...]}}
    """
    ami_me = json.load(open(ami_me_path, "r"))
    meetingeval = {}
    for meeting_id, meeting in ami_me.items():
        segments = [segment_to_text(seg) for seg in meeting["segments"]]
        topics = [seg["topic"] for seg in meeting["segments"]]
        meetingeval[meeting_id] = {"meeting": "\n".join(segments), "segments": segments, "topics": topics}
    return meetingeval


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--ami_me_path", default=os.path.join(here, "..", "..", "AMI_ME.json"))
    parser.add_argument("--out", default=os.path.join(here, "AMI", "meetingeval.json"))
    parser.add_argument("--gtseg_out", default=os.path.join(here, "AMI", "meetingeval_GTseg.json"))
    args = parser.parse_args()

    meetingeval = build_meetingeval(args.ami_me_path)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(meetingeval, f)
    print(f"Wrote {len(meetingeval)} meetings to {args.out}")

    meetingeval_gtseg = build_meetingeval_gtseg(args.ami_me_path)
    os.makedirs(os.path.dirname(args.gtseg_out), exist_ok=True)
    with open(args.gtseg_out, "w") as f:
        json.dump(meetingeval_gtseg, f)
    print(f"Wrote {len(meetingeval_gtseg)} meetings to {args.gtseg_out}")
