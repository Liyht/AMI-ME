set -e

model=$1

# Enable to stop the vLLM server automatically when this script exits.
# trap cleanup EXIT

task="eval" # "seg" or "obj" or "eval"
data_subfolder="AMI"

script="./vllm_eval_parallel.py"

# Single switch for reasoning vs non-reasoning models, driving both thinking mode and the scoring method.
# non-reasoning disables thinking and uses token-probability-weighted scoring; 
# reasoning keeps thinking on and averages over sampling_n samples.
reasoning="false"  # "true" or "false"
if [ "$reasoning" = "true" ]; then
    disable_thinking=""
    score_mode="--score_mode sampling"
    sampling_n="--sampling_n 5"
else
    disable_thinking="--disable_thinking"
    score_mode=""
    sampling_n=""
fi

# Settings only for task "eval" 
winsize=1
seg_source="pred"  # "pred" (LLM segmentation from the _seg run) or "gt" (ground-truth segmentation)
use_GTobj=""    # "--use_GTobj" to score against ground-truth objectives, else ""

# Distinguishes experiment variants under results/. 
# Each task writes to its own folder: seg -> _seg, obj -> _obj, eval -> _{GT,pred}obj_winsize<N>. 
# The eval run pulls its inputs from the matching _seg / _obj folders (see below).
if [[ "$task" == "seg" ]]; then
    postfix="_seg"
elif [[ "$task" == "obj" ]]; then
    postfix="_obj"
elif [[ "$task" == "eval" ]]; then
    if [ -n "$use_GTobj" ]; then
        postfix="_GTobj_winsize${winsize}"
    else
        postfix="_predobj_winsize${winsize}"
    fi
    # GT segmentation runs go to their own folder so they don't clobber pred-seg runs.
    if [ "$seg_source" = "gt" ]; then
        postfix="_GTseg${postfix}"
    fi
fi

data_postfix=""
other=""
if [ "${model}" = "deepseekR1" ] || [ "${model}" = "gpt" ] || [[ "${model}" = "gemini"* ]]; then
    other="--role user"
fi

segment_score_prompt="segment_score.txt"

port="4060"
results_subfolder="${data_subfolder}${data_postfix}_${model}${postfix}"

mkdir -p ./results/$results_subfolder/evaluation/prompts
cp vllm_eval.sh ./results/$results_subfolder
cp -r ./prompts/meetingeval/* ./results/$results_subfolder/evaluation/prompts

if [[ "$task" == "seg" ]]; then
    # Segmentation from Scratch
    # Considering there are broken responses, we need to do segmentation for several times
    for i in {1..4}
    do
      # Do segmentation with --continue_task
      python ${script} --task segmentation --prompt ./prompts/meetingeval/segmentation.txt \
          --save_fp ./results/$results_subfolder/response_segmentation.json --input_fp ./data/$data_subfolder/meetingeval${data_postfix}.json \
          --model $model --port $port --continue_task ${disable_thinking} ${other} |& tee -a ./results/$results_subfolder/segmentation.log
      # Postprocess with --remove_bad_response to remove broken response from checkpoint
      python ./post_process.py --task segmentation --input_path ./results/$results_subfolder/response_segmentation.json \
          --meetingeval_path ./data/$data_subfolder/meetingeval${data_postfix}.json  --save_dir ./results/$results_subfolder --remove_bad_response \
        |& tee -a ./results/$results_subfolder/evaluation/segmentation.log
    done
    python ${script} --task segmentation --prompt ./prompts/meetingeval/segmentation.txt \
        --save_fp ./results/$results_subfolder/response_segmentation.json --input_fp ./data/$data_subfolder/meetingeval${data_postfix}.json \
        --model $model --port $port --continue_task ${disable_thinking} ${other} |& tee -a ./results/$results_subfolder/segmentation.log
    python ./post_process.py --task segmentation --input_path ./results/$results_subfolder/response_segmentation.json \
        --meetingeval_path ./data/$data_subfolder/meetingeval${data_postfix}.json  --save_dir ./results/$results_subfolder \
        |& tee -a ./results/$results_subfolder/evaluation/segmentation.log

elif [[ "$task" == "obj" ]]; then
    # Objective Classification
    python ${script} --task classify_obj --prompt ./prompts/meetingeval/objective_classification.txt \
        --save_fp ./results/$results_subfolder/response_classify_obj.json --input_fp ./data/$data_subfolder/meetingeval${data_postfix}.json \
        --model $model --port $port ${disable_thinking} --continue_task ${other} |& tee -a ./results/$results_subfolder/classify_obj.log
    python ./post_process.py --task process_classify --input_path ./results/$results_subfolder/response_classify_obj.json \
        --save_dir ./results/$results_subfolder |& tee -a ./results/$results_subfolder/evaluation/classify_obj.log

elif [[ "$task" == "eval" ]]; then
    data_postfix="_LLM"

    # Segmentation source: predicted (from the earlier `seg` run) or ground truth (built by data/build_meetingeval.py). 
    # Both land as meetingeval_LLM.json.
    if [ "$seg_source" = "gt" ]; then
        cp ./data/$data_subfolder/meetingeval_GTseg.json ./results/$results_subfolder/meetingeval_LLM.json
    else
        cp ./results/${data_subfolder}_${model}_seg/meetingeval_LLM.json ./results/$results_subfolder/
    fi
    # Objective output from the earlier `obj` run (run it first on the same data/model).
    cp ./results/${data_subfolder}_${model}_obj/pure_obj.json ./results/$results_subfolder/

    # Effectiveness Evaluation
    python ${script}  --task eval_segments --prompt ./prompts/meetingeval/${segment_score_prompt} \
        --input_fp ./results/$results_subfolder/meetingeval${data_postfix}.json --input_fp1 ./results/$results_subfolder/pure_obj.json \
        --save_fp ./results/$results_subfolder/response_score_segments.json \
        --save_fp1 ./results/$results_subfolder/original_responses_of_segments_score.json \
        --model $model --port $port --winsize $winsize --continue_task ${score_mode} ${use_GTobj} ${disable_thinking} ${other} ${sampling_n}\
        |& tee -a ./results/$results_subfolder/eval_segments.log
    python ./post_process.py --task process_scores --input_path ./results/$results_subfolder/response_score_segments.json \
          --save_dir ./results/$results_subfolder --duration_path ./results/$results_subfolder/meetingeval${data_postfix}.json \
          |& tee -a ./results/$results_subfolder/evaluation/eval_segments.log
fi
