"""LLM driver for the framework. Builds a flat task list (per meeting / segment /
sample) and fans the requests out through a thread pool, against either a local
vLLM OpenAI-compatible server or a hosted API (Gemini, GPT). Handles the
segmentation, objective-classification, effectiveness-scoring, and eval-step
generation tasks."""

import openai
import json
import argparse
import tqdm
import time
from openai import OpenAI
from pprint import pprint
import numpy as np
from post_process import extract_and_add_segment_score
import os
from utils import add_id_to_utterance
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
from google.genai import types


# Per-model sampling settings (recommended hyperparameters for each model).
setting = {
    "llama3.3": {"temperature": 0.6, "top_p": 0.9, "extra_body": {}},
    "deepseekR1": {"temperature": 0.6, "top_p": 0.95, "extra_body": {}},
    "qwen3_thinking": {"temperature": 0.6, "top_p": 0.95, "extra_body": {"top_k": 20}},
    "qwen3_nonthinking": {"temperature": 0.7, "top_p": 0.8, "extra_body": {"top_k": 20}},
}
PRINT_SETTING = True
print_lock = threading.Lock()


class Segment(BaseModel):
    start_id: str
    end_id: str
    topic: str
    description: str


def gen_model_config(model: str, args):
    """Constructs the settings dictionary for an API call."""
    if "Qwen3" in model:
        if args.disable_thinking:
            config = setting["qwen3_nonthinking"]
            config["extra_body"] |= {
                "chat_template_kwargs": {"enable_thinking": False},
            }
        else:
            config = setting["qwen3_thinking"]
    elif "llama" in model:
        config = setting["llama3.3"]
        if hasattr(args, "guided_regex"):
            escaped_prefix = re.escape(args.guided_regex)
            final_regex = f"{escaped_prefix}[\\s\\S]*"
            config["extra_body"] |= {"guided_regex": final_regex}
    elif "deepseek" in model:
        config = setting["deepseekR1"]
    elif "gpt" in model:
        config = {}
    else:
        raise Exception("Unknown model")

    if args.temperature_zero:
        config["temperature"] = 0

    if args.require_logprob:
        config["logprobs"] = True
        config["top_logprobs"] = 20

    # Return a copy to avoid modification issues in threads
    return config.copy()


def communicate_and_save(model: str, messages: list, args):
    try:
        if isinstance(client, OpenAI):
            config = gen_model_config(model, args)

            global PRINT_SETTING
            # Use lock to ensure settings are printed only once in a multithreaded environment
            with print_lock:
                if PRINT_SETTING:
                    print("Model Settings:", config)
                    PRINT_SETTING = False

            response = client.chat.completions.create(model=model, messages=messages, **config)
            # A small sleep might not be necessary with vLLM but kept for safety
            time.sleep(0.01)
            all_responses = [choice.message.content for choice in response.choices]

            if not (args.require_logprob or args.require_entire_response):
                return all_responses

            return_entries = [all_responses]

            if args.require_logprob:
                response.model_dump_json()
                content_object = response.choices[0].logprobs.content
                return_entries.append(content_object)

            if args.require_entire_response:
                original_responses = response.model_dump_json()
                return_entries.append(original_responses)

        elif isinstance(client, genai.Client):
            config_dict = {}
            if args.disable_thinking:
                config_dict.update({"thinking_config": types.ThinkingConfig(thinking_budget=0)})
            if "segmentation" in args.task:
                config_dict.update({"response_mime_type": "application/json", "response_schema": list[Segment]})
            if args.require_logprob:
                # Use standard SDK parameter names for enabling logprobs
                config_dict.update({"response_logprobs": True, "logprobs": 20})

            # Create the generation configuration object
            config = types.GenerateContentConfig(**config_dict)
            response = client.models.generate_content(model=model, contents=messages[0]["content"], config=config)
            return_entries = [response.text]

            # Parse logprobs if requested, mimicking the structure of the OpenAI response data
            content_object = []
            if args.require_logprob and response.candidates and response.candidates[0].logprobs_result:
                logprobs_result = response.candidates[0].logprobs_result
                for i, chosen_candidate in enumerate(logprobs_result.chosen_candidates):
                    top_logprobs = []
                    if i < len(logprobs_result.top_candidates):
                        alternatives = logprobs_result.top_candidates[i].candidates
                        for alt_token_info in alternatives:
                            top_logprobs.append({"token": alt_token_info.token, "logprob": alt_token_info.log_probability})
                    content_object.append({"token": chosen_candidate.token, "top_logprobs": top_logprobs})
                return_entries.append(content_object)

            # Append the entire raw response object if requested
            if args.require_entire_response:
                original_response = content_object
                return_entries.append(original_response)
        else:
            raise Exception("Unknown client!")

        if random.random() > 0.9:
            if "gemini" in model:
                print(response.text)
            elif "gpt" in model:
                # For OpenAI/GPT, the content is in the 'choices' attribute
                print(response.choices[0].message.content)

        return tuple(return_entries)
    except Exception as e:
        # Raise the exception to be caught by the thread pool executor
        print(f"Error during API call: {e}")
        raise


def complete_task_on_meetings(eval_target_json, args):
    if args.continue_task and os.path.exists(args.save_fp):
        new_json = json.load(open(args.save_fp))
        exist_meeting_id = set([entry["meeting_id"] for entry in new_json])
        if args.save_fp1 and os.path.exists(args.save_fp1):
            all_original_response = json.load(open(args.save_fp1))
        new_json = [entry for entry in new_json if entry["meeting_id"] in eval_target_json]
    else:
        new_json = []
        all_original_response = []
        exist_meeting_id = set()

    prompt = open(args.prompt_fp).read()
    if args.task == "eval_segments":
        if args.use_GTobj:
            print("Using GT objectives")
            meeting_obj = json.load(open("./data/AMI/objective.json"))
        elif "noobj" in os.path.basename(args.prompt_fp):
            print("Using no objectives")
            meeting_obj = None
        else:
            print("Using predicted objectives")
            meeting_obj = json.load(open(args.input_fp1))

    # --- 1. Prepare all tasks for parallel execution ---
    tasks = []
    for meeting_id, meeting in tqdm.tqdm(eval_target_json.items(), desc="Preparing tasks"):
        if args.continue_task and meeting_id in exist_meeting_id:
            continue

        meeting_transcripts = meeting if isinstance(meeting, str) else meeting["meeting"]
        segments = meeting.get("segments")

        # This is a simplified wrapper for args to handle guided_regex properly per task
        task_args = argparse.Namespace(**vars(args))

        if args.task == "classify_obj":
            cur_prompt = prompt.replace("{{MEETING_TRANSCRIPTS}}", meeting_transcripts)
            messages = [{"role": args.role, "content": cur_prompt}]
            task_args.guided_regex = "Meeting Objectives Analysis\n\nRound 1 - Initial Candidates:"
            tasks.append({"meeting_id": meeting_id, "prompt": cur_prompt, "messages": messages, "args": task_args})

        elif args.task == "segmentation":
            cur_prompt = prompt.replace("{{MEETING_TRANSCRIPTS}}", add_id_to_utterance(meeting_transcripts, start=1))
            cur_prompt = cur_prompt.replace("{{LAST_UTTERANCE_ID_PLACEHOLDER}}", str(len(meeting_transcripts.split("\n"))))
            messages = [{"role": args.role, "content": cur_prompt}]
            tasks.append({"meeting_id": meeting_id, "prompt": cur_prompt, "messages": messages, "args": task_args})

        elif args.task == "eval_segments":
            if args.use_GTobj:
                if meeting_id.startswith("IB40"):
                    obj_key = meeting_id[:5]
                else:
                    obj_key = meeting_id[-1]
            else:
                obj_key = meeting_id

            if meeting_obj is None:
                obj_str = ""
            elif obj_key in meeting_obj:
                obj_str = str(meeting_obj[obj_key])
            else:
                continue

            task_args.guided_regex = "Evaluation Form (scores ONLY):\n- Meeting Segment Effective (1-5):"

            for i, segment in enumerate(segments):
                if args.winsize == 1:
                    cur_prompt = prompt.replace("\nMeeting transcripts:\n{{MEETING_TRANSCRIPTS}}\n", "")
                else:
                    window_context_start = max(0, i - (args.winsize - 1) // 2)
                    window_context_end = min(len(segments), i + (args.winsize - 1) // 2 + 1)
                    window_context = "\n".join(segments[window_context_start:window_context_end])
                    cur_prompt = prompt.replace("{{MEETING_TRANSCRIPTS}}", window_context)

                cur_prompt = cur_prompt.replace("{{SEGMENT_TRANSCRIPTS}}", segment).replace("{{OVERALL_MEETING_OBJECTIVES}}", obj_str)
                messages = [{"role": args.role, "content": cur_prompt}]

                num_samples = args.sampling_n if args.score_mode == "sampling" else 1
                for j in range(num_samples):
                    current_task_args = task_args
                    if args.score_mode == "prob":
                        current_task_args.require_logprob = True
                        current_task_args.require_entire_response = True

                    tasks.append(
                        {
                            "meeting_id": meeting_id,
                            "prompt": cur_prompt,
                            "messages": messages,
                            "args": current_task_args,
                            "sub_task_id": i,
                            "sample_id": j,
                        }
                    )

    # --- 2. Execute tasks in parallel ---
    results = []
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        future_to_task = {executor.submit(communicate_and_save, args.fullname_model, task["messages"], task["args"]): task for task in tasks}

        for future in tqdm.tqdm(as_completed(future_to_task), total=len(tasks), desc="Executing requests"):
            task = future_to_task[future]
            try:
                response_data = future.result()
                task_result = task.copy()
                task_result["response_data"] = response_data
                results.append(task_result)
            except Exception as exc:
                print(f"Task for meeting {task['meeting_id']} generated an exception: {exc}")

    # --- 3. Aggregate results ---
    processed_meetings = defaultdict(lambda: {"meeting_id": None, "prompt": [], "all_responses": [], "logprobs_content": []})

    # Sort results to ensure order is maintained, especially for segments
    results.sort(key=lambda r: (r["meeting_id"], r.get("sub_task_id", 0), r.get("sample_id", 0)))

    for result in results:
        meeting_id = result["meeting_id"]
        data = processed_meetings[meeting_id]
        data["meeting_id"] = meeting_id

        response_data = result["response_data"]

        if args.task == "eval_segments":
            # eval_segments has one sub-task per segment; initialize/populate the per-segment lists
            sub_task_id = result.get("sub_task_id", 0)
            while len(data["prompt"]) <= sub_task_id:
                data["prompt"].append(None)
                data["all_responses"].append([])
                if args.score_mode == "prob":
                    data["logprobs_content"].append(None)

            data["prompt"][sub_task_id] = result["prompt"]

            if args.score_mode == "prob":
                responses, content_object, original_response = response_data
                data["all_responses"][sub_task_id] = responses
                data["logprobs_content"][sub_task_id] = content_object
                all_original_response.append(original_response)
            elif args.score_mode == "sampling":
                data["all_responses"][sub_task_id].append(response_data[0])
        else:
            # For simpler tasks
            data["prompt"] = result["prompt"]
            data["all_responses"] = response_data

    # Combine existing data with new results
    final_json = new_json + list(processed_meetings.values())

    if args.task == "eval_segments":
        if args.score_mode == "prob":
            with open(args.save_fp1, "w") as f:
                json.dump(all_original_response, f)
        final_json = extract_and_add_segment_score(final_json, exist_meeting_id, args.score_mode)

    return final_json


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--prompt_fp", type=str, default="prompts/meetingeval/objective_classification.txt")
    argparser.add_argument("--save_fp", type=str)
    argparser.add_argument("--input_fp", type=str, default="data/meetingeval.json", help="evaluation target")
    argparser.add_argument("--key", type=str)
    argparser.add_argument("--model", type=str, default="llama3.3")
    # task: classify_obj | segmentation | eval_segments | gen_eval_steps
    argparser.add_argument("--task", type=str, default="classify_obj", required=True)
    argparser.add_argument("--port", type=str, default="4060")
    argparser.add_argument("--input_fp1", type=str)
    argparser.add_argument("--save_fp1", type=str)
    argparser.add_argument("--disable_thinking", action="store_true")
    argparser.add_argument("--continue_task", action="store_true")
    argparser.add_argument("--temperature_zero", action="store_true")
    argparser.add_argument("--winsize", type=int, default=1000)
    argparser.add_argument("--role", type=str, default="system")
    argparser.add_argument("--use_GTobj", action="store_true")
    argparser.add_argument(
        "--score_mode",
        type=str,
        default="prob",
        choices=["prob", "sampling"],
        help="Method for score distribution: 'prob' uses token probabilities, 'sampling' uses multiple requests.",
    )
    argparser.add_argument("--sampling_n", type=int, default=5, help="Number of samples to request when --score_mode=sampling.")
    argparser.add_argument("--num_threads", type=int, default=4, help="Number of parallel request threads.")

    args = argparser.parse_args()
    args.require_logprob = False
    args.require_entire_response = False

    if "gemini" in args.model or "gpt" in args.model:
        print(f"For model {args.model}, force single thread.")
        args.num_threads = 1

    if "gemini" in args.model:
        load_dotenv()
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        if "flash" in args.model:
            args.fullname_model = "gemini-2.5-flash"
        elif "pro" in args.model:
            args.fullname_model = "gemini-2.5-pro"
        client = genai.Client(api_key=GEMINI_API_KEY)
    elif "gpt" in args.model:
        load_dotenv()
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        args.fullname_model = "gpt-4o-2024-08-06"
        client = OpenAI()
    else:
        with open("./start_server/model_path.json", "r") as f:
            models = json.load(f)
        args.fullname_model = models[args.model]
        client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY")

    eval_target = json.load(open(args.input_fp))

    if args.task == "gen_eval_steps":
        prompt = open(args.prompt_fp).read()
        new_json = []
        instance = {}
        cur_prompt = prompt
        instance["prompt"] = cur_prompt
        messages = [
            {"role": args.role, "content": cur_prompt},
        ]
        responses = communicate_and_save(args.fullname_model, messages, args)
        instance["all_responses"] = responses
        new_json.append(instance)
    else:
        new_json = complete_task_on_meetings(eval_target, args)

    os.makedirs(os.path.dirname(args.save_fp), exist_ok=True)
    with open(args.save_fp, "w") as f:
        print(f"Task {args.task} save to {args.save_fp}")
        json.dump(new_json, f, indent=4)
