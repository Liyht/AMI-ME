"""Parsers and metrics for the pipeline outputs: turn raw LLM responses into
objectives (pure_obj.json), LLM segmentation (meetingeval_LLM.json), and
duration-weighted effectiveness scores (pure_score.json). Also the CLI entry
point for the post-processing tasks invoked by vllm_eval.sh."""

import json
import numpy as np
import re
import argparse
import os
from utils import remove_think, s2time, count_utterance
import ast
import copy


def extract_obj(class_response_path, objectives_path="./data/AMI/objectives.txt", output_type="str"):
    """extract objective class information from objective classification response json file

    Args:
        class_response_path (str): path to objective classification response json file

    Returns:
        dict: {"meeting_id":[class]}
    """
    with open(class_response_path, "r") as f:
        content = json.load(f)
    objectives = open(objectives_path).read().split("\n")
    for i, entry in enumerate(objectives):
        objectives[i] = entry[entry.find(".") + 2 :]

    result = {}
    for instance in content:
        try:
            meeting_id = instance["meeting_id"]
            response = instance["all_responses"][0]
            response = remove_think(response)
            prefix = "Round 3 - Final Selection"
            start_ind = response.find(prefix) + len(prefix)
            start_ind = response.find("\n", start_ind) + 1
            end_ind = response.find("Justification:", start_ind)
            lines = response[start_ind:end_ind].strip().split("\n")
            classes = []
            for line in lines:
                for obj in objectives:
                    if obj in line:
                        classes.append(obj)
                        break
            # classes = [i[2:] for i in classes]
            if output_type == "int":
                classes = [objectives.index(i) + 1 for i in classes]
            result[meeting_id] = classes
        except Exception as e:
            with open("./logs/parse_LLM_response_error_logs.txt", "a") as f:
                f.write("#" * 20 + f"\nMet error when processing {meeting_id}, task is extract_obj post processing, response is as following:\n")
                f.write(response)
            print(e)
    return result


def align_segjson_with_transcripts(segments: list, meeting_transcript: list) -> list:
    """Map the LLM's segment boundaries (start_id/end_id) back onto the transcript,
    attaching each segment's utterance text and reporting any uncovered gaps.

    Args:
        segments (list): LLM-produced segments [{"start_id", "end_id", "topic", "description"}, ...].
        meeting_transcript (list): ["[start - end] [speaker] utterance", ...].

    Returns:
        new_segments (list): [{"start_id", "end_id", "topic", "description", "text"}, ...]
        lost_idx (list): [(start_id, end_id), ...] utterance ranges left uncovered.
    """
    # For checking completion of segments
    num_total_transcripts = len(meeting_transcript)
    last_seg_end_idx = -1
    lost_idx = []

    new_segments = []
    new_segments_id = []
    for i, segment in enumerate(segments):
        new_segment = {}
        seg_start_idx = int(segment["start_id"]) - 1
        seg_end_idx = int(segment["end_id"]) - 1

        topic = segment["topic"]
        description = segment["description"]
        if seg_start_idx != last_seg_end_idx + 1:
            lost_idx.append([last_seg_end_idx + 1, seg_start_idx - 1])

        new_segments_id.append([seg_start_idx, seg_end_idx])

        new_segment["topic"] = topic
        new_segment["description"] = description
        new_segments.append(new_segment)

        last_seg_end_idx = seg_end_idx

    if seg_end_idx != num_total_transcripts - 1:
        lost_idx.append([seg_end_idx + 1, num_total_transcripts - 1])

    new_segments = refine_segments_id_to_text(new_segments_id, new_segments, meeting_transcript)

    return new_segments, lost_idx


def valid_segments_id(segments_id):
    last_end = -1
    for segment_id in segments_id:
        start, end = segment_id
        if start > end or start != last_end + 1:
            return False
        last_end = end
    return True


def refine_segments_id_to_text(segments_id, segments, meeting_transcript):
    """Repair overlapping/gapped segment id ranges into a contiguous cover of the
    transcript, then attach the corresponding utterance text to each segment.

    Args:
        segments_id (list): [[start_idx, end_idx], ...] candidate utterance ranges.
        segments (list): [{"start_id", "end_id", "topic", "description"}, ...] (no "text" yet).
        meeting_transcript (list): the transcript lines indexed by the ids above.

    Returns:
        list: segments with a "text" field added for each.
    """
    original_segments_id = copy.deepcopy(segments_id)
    # The start_id of first segment should be 0
    segments_id[0][0] = 0
    # The end_id of last segment should be len(segments_id) - 1
    segments_id[len(segments_id) - 1][-1] = len(meeting_transcript) - 1

    while not valid_segments_id(segments_id):
        # Split segments according to start_id
        for i in range(len(segments_id) - 1, 0, -1):
            if segments_id[i][0] > segments_id[i - 1][1] + 1:
                # e.g., [1, 2], [10, 11] -> [1, 9], [10, 11]
                segments_id[i - 1][1] = segments_id[i][0] - 1

            elif segments_id[i][0] < segments_id[i - 1][1] + 1:
                # How to process overlap?
                # e.g., [1, 10], [2, 11]
                segments_id[i - 1][1] = segments_id[i][0] - 1

        # Search and remove invalid segments
        invalid = []
        for i in range(len(segments_id)):
            if segments_id[i][0] > segments_id[i][1]:
                invalid.append(i)
        segments_id = [segments_id[i] for i in range(len(segments_id)) if i not in invalid]
        segments = [segments[i] for i in range(len(segments)) if i not in invalid]

    if segments_id != original_segments_id:
        pass

    assert len(segments_id) == len(segments)
    for i, segment_id in enumerate(segments_id):
        start, end = segment_id
        utterance_text = [utterance for utterance in meeting_transcript[start : end + 1]]
        utterance_text = "\n".join(utterance_text)
        segments[i]["text"] = utterance_text

    return segments


def extract_seg(response_path, meetingeval_path, remove_bad_responses=False):
    """Parse the LLM segmentation responses and align them to the transcripts.

    Args:
        response_path (str): raw LLM segmentation responses (response_segmentation.json).
        meetingeval_path (str): meetingeval.json providing each meeting's transcript.
        remove_bad_responses (bool): drop (and rewrite) meetings whose parsed
            segmentation leaves too many utterances uncovered.

    Returns:
        dict: {meeting_id: [{"start_id", "end_id", "topic", "description", "text"}, ...]}
    """
    with open(response_path, "r") as f:
        content = json.load(f)
        valid_content = []
    meetingeval_data = json.load(open(meetingeval_path))

    all_segments = {}
    total_lost = []
    for i, meeting_instance in enumerate(content):
        try:
            meeting_id = meeting_instance["meeting_id"]
            response = meeting_instance["all_responses"][0]
            response = remove_think(response)
            if "```json" in response:
                response = response[response.find("```json") :]
            response = response[response.find("[") : response.rfind("]") + 1]
            try:
                response_json = json.loads(response)
            except json.decoder.JSONDecodeError:
                try:
                    response_json = ast.literal_eval(response)
                except Exception as e:
                    print(response)
                    print(e)
            meeting_transcript = meetingeval_data[meeting_id]["meeting"].strip().split("\n")

            segments, lost_idx = align_segjson_with_transcripts(response_json, meeting_transcript)
            total_lost += [abs(end - start + 1) for start, end in lost_idx]
            all_segments[meeting_id] = segments
            if remove_bad_responses:
                if sum([abs(end - start) for start, end in lost_idx]) > 10:
                    print(f"WARNING! When doing segmentation for {meeting_id}, lost idx: {lost_idx}")
                    raise
            valid_content.append(meeting_instance)
        except Exception as e:
            print(f"Failed to process {meeting_id} due to error: {e}")
    print(f"Average lost: each meeting lost {sum(total_lost) / len(content)} utterances")
    if len(valid_content) < len(content):
        with open(response_path, "w") as f:
            json.dump(valid_content, f)
        content_meeting_id = set([i["meeting_id"] for i in content])
        valid_content_meeting_id = set([i["meeting_id"] for i in valid_content])

        print(
            f"ERROR! We deleted invalid meetings ({content_meeting_id - valid_content_meeting_id}) from {response_path}, please generate those meetings again!"
        )
    return all_segments


def extract_and_add_segment_score(response_json_list, exist_meeting_id, score_mode, temperature=None):
    """For each instance in response_json_list, extract segment scores, save them,
    and remove "logprobs_content" so the result can be serialized as JSON.

    Returns:
        result (dict): {meeting_id: (scores, all_score_probs)}
    """
    result = []
    for i, instance in enumerate(response_json_list):
        try:
            # Each instance is response for a meeting
            # instance={"meeting_id": , "prompt": [], "all_responses": []}
            meeting_id = instance["meeting_id"]
            if meeting_id not in exist_meeting_id:
                all_score_probs = []
                all_score_tokens = []

                if score_mode == "prob":
                    for content_object in instance["logprobs_content"]:
                        probs, score_token_str = process_logprobs_content(content_object, temperature)
                        all_score_probs.append(probs)
                        all_score_tokens.append(int(score_token_str))
                elif score_mode == "sampling":
                    for i, segment_responses in enumerate(instance["all_responses"]):
                        # Parse all scores for the current segment
                        scores = [parse_score_from_text(resp) for resp in segment_responses]
                        # Filter out any responses where parsing failed
                        valid_scores = [s for s in scores if s is not None]

                        if not valid_scores:
                            # If no scores could be parsed, assume a zero distribution
                            raise Exception(f"Meeting {meeting_id} segment {i}: No scores could be parsed")
                        else:
                            counts = np.bincount(valid_scores, minlength=6)[1:]  # Index 0 is unused, we want 1-5
                            # Normalize counts to get probability distribution
                            probs = counts / len(valid_scores)
                        all_score_probs.append(probs)
                all_score_probs = np.stack(all_score_probs)
                scores = probs_to_weighted_score(all_score_probs)

                new_instance = instance.copy()
                new_instance["scores"] = (scores.tolist(), all_score_probs.tolist())
                if all_score_tokens:
                    new_instance["score_tokens"] = all_score_tokens
                new_instance.pop("logprobs_content")

            else:
                new_instance = instance.copy()
        except Exception as e:
            # raise e
            print("Error!", e)
            continue

        result.append(new_instance)
        # result[meeting_id]=(scores.tolist(), all_score_probs.tolist())
    return result


def convert_to_seconds(time_str):
    # Parse the time string in format H:MM:SS
    time_parts = time_str.split(":")

    # Convert to integers
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = int(time_parts[2])

    # Calculate total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds

    return total_seconds


def parse_timestamp_to_seconds(text):
    # regex for the "[start_time - end_time]" format
    pattern = r"\[(\d+:\d+:\d+)\s*-\s*(\d+:\d+:\d+)\]"

    # search for a match
    match = re.search(pattern, text)

    if match:
        start_time = match.group(1)
        end_time = match.group(2)
        return convert_to_seconds(start_time), convert_to_seconds(end_time)
    else:
        print(f"Found mistake utterance: {text}")
        return None, None


def get_time_duration_of_transcripts(transcripts, min_duration=1):
    start_time, end_time = get_time_range_of_transcripts(transcripts)
    duration = max(min_duration, end_time - start_time)
    return duration


def get_time_range_of_transcripts(transcripts):
    utterances = transcripts.rstrip().split("\n")
    start_and_end_time = [parse_timestamp_to_seconds(utterance) for utterance in utterances]
    start_and_end_time = [times for times in start_and_end_time if times[0] is not None and times[1] is not None]
    start_time = min([times[0] for times in start_and_end_time])
    end_time = max([times[1] for times in start_and_end_time])
    return (start_time, end_time)


def cal_overall_meeting_score(segment_result, duration_path):
    """Calculate overall meeting score according to segment scores and duration

    Args:
        segment_result (dict): {meeting_id:(scores, all_score_probs)}

    Returns:
        segment_result: {meeting_id:(score, scores, all_score_probs)}
    """
    # Note: to calculate weighted time according to duration, ignore time duration between segments
    meeting_eval = json.load(open(duration_path))
    new_segment_result = {}
    for meeting_id in segment_result:
        try:
            scores = segment_result[meeting_id][0]
            durations = []
            for segment in meeting_eval[meeting_id]["segments"]:
                try:
                    duration = get_time_duration_of_transcripts(segment)
                    durations.append(duration)
                except Exception as e:
                    print(f"{meeting_id=}")
                    print(f"{segment=}")
                    raise e
            durations = np.array(durations)
            time_weight = durations / durations.sum()
            weighted_score = (scores * time_weight).sum()
            new_segment_result[meeting_id] = [
                weighted_score,
                durations.tolist(),
                *segment_result[meeting_id],
            ]
        except Exception as e:
            print(f"Error! Can not generate pure_score data for {meeting_id}")
            print(f"Error: {e}")
    return new_segment_result


def parse_score_from_text(response_text):
    """Extracts a numerical score from the LLM's response text."""
    # Regex for: "- Meeting Segment Effective (1-5): [SCORE]"
    response_text = remove_think(response_text)
    # match = re.search(r"Meeting Segment Effective \(1-5\):.*?(\d+)", response_text)
    match = re.search(r":.*?(\d+)", response_text)
    if match:
        return int(match.group(1))
    print(f"Warning: Could not parse score from response: {response_text}")
    return None


def search_score_from_content(content_object):
    # - Meeting Segment Effective (1-5): [SCORE]
    if not content_object:
        raise ValueError("Cannot search for score in empty logprobs content.")

    # UNIVERSAL FIX: Check if we have dicts (from Gemini) or objects (from OpenAI)
    is_dict = isinstance(content_object[0], dict)

    # Use the correct accessor (.token or ['token']) based on the type
    response_tokens = [token_obj["token"] if is_dict else token_obj.token for token_obj in content_object]

    think_exist = False
    for i, token in enumerate(response_tokens):
        # skip think
        if token == "<think>":
            think_exist = True
        if think_exist and token != "</think>":
            continue
        else:
            think_exist = False

        # get score index
        if token == "1" and "".join(response_tokens[i : i + 4]) == "1-5):":
            candidate = i + 5
            if response_tokens[candidate] not in [str(i) for i in range(1, 6)]:
                candidate += 1
            if candidate >= len(response_tokens):
                raise Exception(f"Can not parse response to get score.\n Response tokens: {response_tokens}")
            return candidate


def prob_sample_to_recover(prob_sample, temperature, top_p=1, hidden_vocab_num=None):
    prob_sample = prob_sample * top_p
    top_logprobs = len(prob_sample)

    if hidden_vocab_num is None:
        hidden_vocab_num = 1 if (1 - prob_sample.sum() > 0.001) else 0

    if hidden_vocab_num > 0:
        hidden_probs = [max(1 - prob_sample.sum(), 0) / hidden_vocab_num] * hidden_vocab_num
        prob_sample = np.concatenate([prob_sample, hidden_probs])

    prob_recover_unnorm = prob_sample**temperature
    prob_recover = prob_recover_unnorm / prob_recover_unnorm.sum()

    return prob_recover[:top_logprobs]


def process_logprobs_content(content_object, temperature):
    # Note: if score 1-5 in top 20, use the probability; otherwise, set probability to 0
    idx = search_score_from_content(content_object)
    score_token = content_object[idx]

    # UNIVERSAL FIX: Check the type to use the correct accessor
    is_dict = isinstance(score_token, dict)

    num_scores = set([str(i) for i in range(1, 6)])
    probs = np.zeros(5)

    # Use the correct accessor for top_logprobs
    score_token_str = score_token["token"] if is_dict else score_token.token
    top_logprobs = score_token["top_logprobs"] if is_dict else score_token.top_logprobs

    for candidate_token in top_logprobs:
        # Use the correct accessor for token and logprob
        token = candidate_token["token"] if is_dict else candidate_token.token
        logprob = candidate_token["logprob"] if is_dict else candidate_token.logprob

        if token in num_scores:
            probs[int(token) - 1] = np.exp(logprob)

    if temperature:
        probs = prob_sample_to_recover(probs, temperature, top_p=1, hidden_vocab_num=None)
    return probs, score_token_str


def probs_to_weighted_score(probs, precision=2):
    weighted_score = (np.arange(1, 6) * probs).sum(axis=-1)
    weighted_score = np.round(weighted_score, precision)
    return weighted_score


def gen_segmentation_context_for_checking(all_segments):
    """Render the parsed segments into a human-readable text block (one per
    meeting) for manual inspection of the segmentation output."""
    context = []
    for meeting_id, segments in all_segments.items():
        meeting_context = []
        for i, segment in enumerate(segments):
            segment_context = f"## Topic {i + 1}: " + segment["topic"] + "\n"
            segment_context += "## Description: " + segment["description"] + "\n"
            segment_context += segment["text"]
            meeting_context.append(segment_context)
        context.append(meeting_id + "\n" + "\n\n".join(meeting_context))
    return "\n\n".join(context)


def convert_segments_to_meetingeval(all_segments):
    """Convert parsed LLM segments into the meetingeval_LLM.json structure consumed
    by the effectiveness-scoring stage.

    Args:
        all_segments (dict): {meeting_id: [{"start_id", "end_id", "topic", "description", "text"}, ...]}

    Returns:
        dict: {meeting_id: {"meeting": str, "segments": [str, ...], "topics": [str, ...]}}
    """
    meetingeval_results = {}
    for meeting_id, segments in all_segments.items():
        meetingeval_results[meeting_id] = {}
        meetingeval_results[meeting_id]["segments"] = []
        meetingeval_results[meeting_id]["topics"] = []
        for segment in segments:
            if not segment["text"]:
                continue
            meetingeval_results[meeting_id]["segments"].append(segment["text"])
            meetingeval_results[meeting_id]["topics"].append(segment["topic"])
        meetingeval_results[meeting_id]["meeting"] = "\n".join(meetingeval_results[meeting_id]["segments"])
    return meetingeval_results


def count_segments(meetingeval_data):
    # meetingeval_data: {meeting_id: {"meeting": "", "segments": [], "topics": []}}
    count = 0
    for meeting_data in meetingeval_data.values():
        count += len(meeting_data["segments"])
    return count


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--task", type=str, required=True)
    argparser.add_argument("--input_path", type=str, required=True)
    argparser.add_argument("--save_dir", type=str)
    argparser.add_argument("--duration_path", type=str)
    argparser.add_argument("--meetingeval_path", type=str)
    argparser.add_argument("--flat_transcripts_path", type=str)
    argparser.add_argument("--remove_bad_response", action="store_true")
    args = argparser.parse_args()

    if args.task == "process_classify":
        response_path = args.input_path
        result = extract_obj(response_path)
        with open(os.path.join(args.save_dir, "pure_obj.json"), "w") as f:
            json.dump(result, f)
        print(result)

    elif args.task == "process_scores":
        response_path = args.input_path
        response_json_list = json.load(open(response_path))
        segment_result = {}
        for instance in response_json_list:
            segment_result[instance["meeting_id"]] = instance["scores"]
        pure_score_dict = cal_overall_meeting_score(segment_result, args.duration_path)
        with open(os.path.join(args.save_dir, "pure_score.json"), "w") as f:
            json.dump(pure_score_dict, f)

    elif args.task == "segmentation":
        response_path = args.input_path
        all_segments = extract_seg(response_path, args.meetingeval_path, args.remove_bad_response)
        meetingeval_results = convert_segments_to_meetingeval(all_segments)
        print(f"Number of segments: {count_segments(meetingeval_results)}")
        print(f"Average number of segments per meeting: {count_segments(meetingeval_results) / len(meetingeval_results)}")
        context_for_checking = gen_segmentation_context_for_checking(all_segments)

        with open(os.path.join(args.save_dir, "pure_seg.json"), "w") as f:
            json.dump(all_segments, f)
        with open(os.path.join(args.save_dir, "response_segmentation.txt"), "w") as f:
            f.write(context_for_checking)
        with open(os.path.join(args.save_dir, "meetingeval_LLM.json"), "w") as f:
            json.dump(meetingeval_results, f)

    elif args.task == "gen_eval_steps":
        response_path = args.input_path
        response_data = json.load(open(response_path))
        steps = remove_think(response_data[0]["all_responses"][0])
        with open(os.path.join("./prompts/meetingeval", "eval_steps.txt"), "w") as f:
            f.write(steps)

        before = open("./prompts/meetingeval/before_eval_steps.txt", "r").read()
        after = open("./prompts/meetingeval/after_eval_steps.txt", "r").read()

        combine = before + "\n\n" + steps + "\n\n" + after
        with open(os.path.join("./prompts/meetingeval", "segment_score.txt"), "w") as f:
            f.write(combine)
