"""Shared helpers: transcript-line parsing, time formatting, reasoning-model
<think> stripping, and utterance-ID injection."""

import datetime
import re


def s2time(seconds):
    if isinstance(seconds, str):
        seconds = float(seconds)
    seconds = round(seconds)
    convert = str(datetime.timedelta(seconds=seconds))
    return convert


def remove_think(response):
    # deepseekR1 respons doesn't contain <think>
    cleaned_content = re.sub(r".*?</think>\n?", "", response, flags=re.DOTALL)
    return cleaned_content


def parse_time_and_speaker(line: str) -> dict:
    """
    Extract the timestamp, speaker ID and text from a single transcript line.
    If only one timestamp is present, it is treated as the end_time.

    Args:
        line (str): a single line containing timestamp, speaker and text, e.g.
                    "[0:00:33 - 0:01:02] [A] Okay. Good morning everybody."
                    or "[0:01:02] [A] ...since we only have twenty five minutes."

    Returns:
        dict: with 'start_time', 'end_time', 'speaker_id', 'text', and 'error'.
              'error' is None on success. 'start_time' may be "N/A".
              'text' is what the speaker said.
    """
    # Regex breakdown:
    # ^\[                               # start of line and opening '[' of the time
    # (?:                               # optional non-capturing group (for the "start_time - " part)
    #   (\d+:\d{2}:\d{2})               #   group 1: start_time (H:MM:SS)
    #   \s*-\s*                         #   matches " - " (any spaces around the dash)
    # )?                                # this group is optional
    # (\d+:\d{2}:\d{2})                 # group 2: end_time (the only timestamp if start_time is absent)
    # \]                               # closing ']' of the time
    # \s*                              # spaces between time and speaker ID
    # \[                               # opening '[' of the speaker ID
    # ([^\]]+)                           # group 3: speaker_id (one or more non-']' chars)
    # \]                               # closing ']' of the speaker ID
    # \s*                              # optional spaces after the speaker ID block
    # (.*)                               # group 4: the actual text content
    # $                                  # end of line

    pattern = r"^\[(?:(\d+:\d{2}:\d{2})\s*-\s*)?(\d+:\d{2}:\d{2})\]\s*\[([^\]]+)\]\s*(.*)$"

    match = re.match(pattern, line)

    if match:
        # group 1: start_time (if present)
        # group 2: end_time
        # group 3: speaker_id
        # group 4: text

        start_time_val = match.group(1)
        end_time_val = match.group(2)
        speaker_id_val = match.group(3)
        text_val = match.group(4).strip()  # strip whitespace around the text

        return {
            "start_time": start_time_val if start_time_val else "N/A",
            "end_time": end_time_val,
            "speaker_id": speaker_id_val,
            "text": text_val,
            "error": None,
        }
    else:
        return {"start_time": None, "end_time": None, "speaker_id": None, "text": None, "error": "Pattern did not match the line."}


def utterance_to_text(utterance):
    return f"[{s2time(utterance['start'])} - {s2time(utterance['end'])}] [{utterance['speaker']}] {utterance['text'].strip()}"


def add_id_to_utterance(transcripts, start=1):
    utterance = transcripts.split("\n")
    for i in range(len(utterance)):
        utterance[i] = f"ID:{start + i} " + utterance[i]
    return "\n".join(utterance)


def count_utterance(transcript):
    return len(transcript.strip().split("\n"))
