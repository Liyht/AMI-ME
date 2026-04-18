# AMI-ME

Dataset and code for the ACL 2026 paper
**"Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation"**.

The paper proposes a new paradigm for evaluating meeting effectiveness:
effectiveness is defined as the rate of objective achievement over time,
and is scored per topical segment rather than per whole meeting. **AMI-ME**
is the meta-evaluation dataset built to support this paradigm.

## Release status

- [x] **Dataset** — released in this repository (`AMI_ME.json`).
- [ ] **Code** — the LLM-based automatic evaluation framework
  (topic segmentation + objective classification + segment-level scoring +
  segmentation alignment) is being cleaned up and will be released here in a
  follow-up update.

## Dataset summary

AMI-ME is derived from the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
(Carletta et al., 2005). We refined the original coarse, discontinuous topic
segmentation into continuous fine-grained units with an LLM-assisted
reference-based method, and then collected human effectiveness annotations
on every resulting segment.


Each segment is independently rated by three annotators on a 5-point
effectiveness scale, and each annotator also identifies the meeting
objectives that the segment addresses (multi-label). See the paper for the
full annotation protocol.

## Schema

`AMI_ME.json` is a single JSON object keyed by AMI meeting ID:

```json
{
  "ES2002a": {
    "meeting_type": "scenario_a",
    "predefined_objectives": [
      "Effectively share information about the project",
      "Get acquainted to team members",
      "Learn to use drawing tools",
      "Generate good ideas on remote control"
    ],
    "segments": [
      {
        "segment_id": "ES2002a_000",
        "topic": "Participant Introductions",
        "utterances": [
          {"start": 50, "end": 77, "speaker": "B", "text": "Okay. Right. Um well this is the kick-off meeting ..."},
          {"start": 67, "end": 75, "speaker": "D", "text": "Mm-hmm. Great."}
        ],
        "scores": [4, 3, 4],
        "objectives": [
          ["2. Get acquainted to team members"],
          ["2. Get acquainted to team members"],
          ["2. Get acquainted to team members"]
        ]
      }
    ]
  }
}
```

**Meeting-level fields**

- `meeting_type` — one of `scenario_a`, `scenario_b`, `scenario_c`,
  `scenario_d` (the four sub-types of AMI scenario business meetings),
  `non_scenario_film` (movie-club film selection), or
  `non_scenario_office` (office relocation).
- `predefined_objectives` — the manually designed meeting objectives shown
  to annotators for this meeting type. Used both as context during
  annotation and as the reference label space for the per-segment
  `objectives` field.
- `segments` — ordered list of fine-grained topical segments.

**Segment-level fields**

- `segment_id` — `{meeting_id}_{index:03d}`.
- `topic` — short topic label generated during reference-based
  segmentation.
- `utterances` — ordered list of utterances, each with `start` / `end`
  (seconds from meeting start), anonymized `speaker` letter, and `text`.
  Segment start/end and duration can be derived from these.
- `scores` — list of three integer effectiveness scores (1–5), one per
  annotator.
- `objectives` — list of three sub-lists, one per annotator. Each
  sub-list contains the `predefined_objectives` the annotator judged this
  segment to serve, or `"None of them"` if none applied. The leading number
  in each label (e.g. `"2. Get acquainted..."`) matches the index in
  `predefined_objectives` (1-based).

## License

The dataset is released under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license, consistent with the original AMI Meeting Corpus. See
[`LICENSE`](LICENSE) for the full text.

The original audio and transcripts are the property of the AMI Consortium
and are subject to the AMI Corpus license; users who need the raw media
should obtain it from the [AMI Corpus website](https://groups.inf.ed.ac.uk/ami/corpus/).

## Citation

If you use AMI-ME, please cite our paper:

```bibtex
@inproceedings{li2026ami-me,
  title     = {Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation},
  author    = {Li, Yihang and Chu, Chenhui},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
```
