import os
import json
import datasets
from fire import Fire
from functools import partial
from loguru import logger
from datasets import Dataset
import numpy as np
from copy import deepcopy
from modules import (
    node_sampling_model_card, edge_sampling, message_passing, graph_pooling,
    generate_initial_response_with_confidence, maybe_replace_with_general,
)
from human_eval.evaluation import evaluate_functional_correctness
from utils import (
    seed_everything,
    setup_logger,
    get_alphabet_choice,
    remove_boxed,
    last_boxed_only_string,
    extract_human_eval_completion,
    evaluate,
)
from time import time
from endpoint import model_endpoint_dict

def load_human_eval_dataset_from_json(json_path):
    """
    Load HumanEval dataset from a local JSON file (for debug/dev).
    """
    with open(json_path, 'r') as f:
        samples = json.load(f)

    eval_set = Dataset.from_list(samples)
    eval_set = eval_set.add_column("idx", list(range(len(eval_set))))
    instructions = []
    for data in eval_set:
        prompt = data["prompt"]
        instruction = f"""
            Given the following Python function signature, complete the function to match the expected behavior.

            {prompt}

            Completed Function:
            """
        instructions.append(instruction)

    eval_set = eval_set.add_column("instruction", instructions)
    return eval_set


def load_human_eval_dataset(eval_type):
    """
    Load HumanEval dataset and prepare prompts.
    """
    dataset = datasets.load_dataset("openai_humaneval")
    eval_set = dataset[eval_type]

    eval_set = eval_set.add_column("idx", list(range(len(eval_set))))
    instructions = []
    for data in eval_set:
        prompt = data["prompt"]

        instruction = f"""
        Given the following Python function signature, complete the function to match the expected behavior.

        {prompt}

        Completed Function:
        """

        instructions.append(instruction)

    eval_set = eval_set.add_column("instruction", instructions)

    return eval_set

def remap_keys_and_values(edge_dict, name_map):
    return {
        name_map[key]: [name_map[v] for v in value_list]
        for key, value_list in edge_dict.items()
    }

def make_unique_node_names(sampled_nodes):
    """
    Given a list like ["qwen", "mathstral", "qwen"], returns:
      - unique_names: ["qwen_0", "mathstral", "qwen_1"]
      - unique_to_original: {"qwen_0": "qwen", "mathstral": "mathstral", "qwen_1": "qwen"}
    Only appends index suffix when a name appears more than once.
    """
    from collections import Counter
    counts = Counter(sampled_nodes)
    seen = {}
    unique_names = []
    unique_to_original = {}

    for name in sampled_nodes:
        if counts[name] > 1:
            idx = seen.get(name, 0)
            unique_name = f"{name}_{idx}"
            seen[name] = idx + 1
        else:
            unique_name = name

        unique_names.append(unique_name)
        unique_to_original[unique_name] = name

    return unique_names, unique_to_original

def process_fn(
    item,
    data=None,
    reference_models=[],
    model_info_dict=None,
    temperature=0.7,
    max_tokens=800,
    rounds=1,
    top_k=1,
    threshold=0.1,
    meta_llm=None,
    graph_pooling_method=None,
    output_path=None,
    num_choice=4,
    final_prompt='',
    seed=0,
):
    if data == "human_eval":
        task_id = item["task_id"]

    answer = None
    instruction = item["instruction"]
    input_token_total = 0
    output_token_total = 0

    start_time = time()

    refernce_model_dict = model_endpoint_dict[meta_llm]
    meta_llm_model_name = refernce_model_dict['model_id']
    meta_llm_model_endpoint = refernce_model_dict['url']

    refernce_model_dict = model_endpoint_dict[meta_llm]
    meta_llm_model_name_graph = refernce_model_dict['model_id']
    meta_llm_model_endpoint_graph = refernce_model_dict['url']

    models_dict = {k: v for k, v in model_endpoint_dict.items() if k in reference_models}

    # 1. Node Sampling
    logger.info("Node Sampling")
    fallback_node_sampling = False
    sampled_nodes, fallback_node_sampling, i_t_c, o_t_c = node_sampling_model_card(
            model=meta_llm_model_name,
            endpoint=meta_llm_model_endpoint,
            messages=[instruction],
            model_info_dict=models_dict,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed
        )

    input_token_total += i_t_c
    output_token_total += o_t_c

    logger.info(f"Sampled Nodes: {sampled_nodes}")

    logger.info("Generating initial responses for nodes")

    final_response = None
    general_model_name = meta_llm

    if len(sampled_nodes) == 1:
        ## single model case
        node = sampled_nodes[0]
        node_model_dict = models_dict[node]

        raw_resp, answer_parsed, conf, i_t_c, o_t_c = generate_initial_response_with_confidence(
            model_id=node_model_dict["model_id"],
            endpoint=node_model_dict["url"],
            instruction=instruction,
            data=data,
            num_choice=num_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            debug_txt="[Initial Response] "
        )
        final_response = raw_resp

        input_token_total += i_t_c
        output_token_total += o_t_c

    else:
        ## Initial Responses for Nodes (with confidence)
        initial_responses = {}
        confidences = {}
        answers_parsed = {}

        # Create unique names for duplicate nodes
        unique_names, unique_to_original = make_unique_node_names(sampled_nodes)
        model_name_idx = dict(enumerate(unique_names))
        model_idx_domains = [models_dict[unique_to_original[v]]['domain'] for k,v in model_name_idx.items()]

        for i, uname in enumerate(unique_names):
            orig_name = unique_to_original[uname]
            node_model_dict = models_dict[orig_name]
            raw_resp, answer_parsed, conf, i_t_c, o_t_c = generate_initial_response_with_confidence(
                model_id=node_model_dict["model_id"],
                endpoint=node_model_dict["url"],
                instruction=instruction,
                data=data,
                num_choice=num_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                debug_txt=f"[Initial Response - {uname}] "
            )

            initial_responses[i] = raw_resp
            confidences[i] = conf
            answers_parsed[i] = answer_parsed

            input_token_total += i_t_c
            output_token_total += o_t_c

        logger.info(f"Initial Confidences: {dict(zip(unique_names, [confidences[i] for i in range(len(unique_names))]))}")

        # Confidence-based replacement: replace low-confidence / N/A models with general model
        replacements_log = {}
        for i, uname in enumerate(list(unique_names)):
            orig_name = unique_to_original[uname]
            replaced, i_t_c, o_t_c = maybe_replace_with_general(
                node_idx=i,
                node_name=orig_name,
                confidence=confidences[i],
                models_dict=models_dict,
                general_model_name=general_model_name,
                initial_responses=initial_responses,
                confidences=confidences,
                answers=answers_parsed,
                sampled_nodes=sampled_nodes,
                instruction=instruction,
                data=data,
                num_choice=num_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed
            )
            if replaced:
                replacements_log[i] = {"original": orig_name, "replaced_with": general_model_name,
                                       "original_confidence": confidences.get(i, -1)}
                unique_names[i] = general_model_name if unique_names.count(general_model_name) == 0 else f"{general_model_name}_{i}"
                unique_to_original[unique_names[i]] = general_model_name
                input_token_total += i_t_c
                output_token_total += o_t_c

        # Rebuild model_name_idx after potential replacements
        model_name_idx = dict(enumerate(unique_names))
        model_idx_domains = [models_dict[unique_to_original[v]]['domain'] for k,v in model_name_idx.items()]

        if replacements_log:
            logger.info(f"Replacements made: {replacements_log}")

        # Build a models_dict that maps unique names to model configs
        models_dict_unique = {uname: models_dict[unique_to_original[uname]] for uname in unique_names}

        # 2. Edge Sampling 
        logger.info("Edge Generation")

        output = edge_sampling(
            models_dict=models_dict_unique,
            model_name_idx=model_name_idx,
            initial_responses=initial_responses,
            messages=[instruction],
            threshold=threshold,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            confidences=confidences,
        )

        if not isinstance(output, tuple):
            if output == -1:
                # All models below threshold — fallback to mean pooling of initial responses
                logger.warning("[Edge Sampling] All models pruned by threshold. Falling back to mean pooling of initial responses.")
                final_response, i_t_c, o_t_c = graph_pooling(
                    model=meta_llm_model_name_graph,
                    endpoint=meta_llm_model_endpoint_graph,
                    refined_response=initial_responses,
                    messages=[instruction],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    final_prompt=final_prompt,
                    seed=seed,
                )
                input_token_total += i_t_c
                output_token_total += o_t_c
            else:
                final_response = output  # only top-1 case

        else:
            source_edges, target_edges, score_dict, score_matrix_raw, fallback_used, fallback_edge_sampling, i_t_c, o_t_c = output

            input_token_total += i_t_c
            output_token_total += o_t_c

            logger.info(f"Generated edges for Source Nodes: {source_edges} / Generated edges for Target Nodes: {target_edges}")

            source_with_most_edges = None

            if graph_pooling_method == 'max':
                try:
                    score_value = max(score_dict.values())
                    score_keys = [k for k, v in score_dict.items() if v == score_value]

                    if len(score_keys) == 1:
                        source_with_most_edges = score_keys[0]
                    else:
                        priority_order = ['general_0', 'general_1', 'general_2', 'general', 'code', 'math']

                        def get_priority(key):
                            domain = model_idx_domains[key]
                            return priority_order.index(domain) if domain in priority_order else len(priority_order)

                        source_with_most_edges = min(score_keys, key=get_priority)

                except Exception as e:
                    logger.error(f"Error in edge post-processing: {e}")

            # 3. Message Passing
            logger.info("Starting Message Passing")

            round_responses = deepcopy(initial_responses)
            try:
                for _round in range(rounds):
                    after_s_to_t, after_t_to_s, i_t_c, o_t_c = message_passing(
                        model_endpoint_dict=models_dict_unique,
                        model_name_idx=model_name_idx,
                        initial_responses=round_responses,
                        source_edges=source_edges,
                        target_edges=target_edges,
                        score_dict=score_dict,
                        source_with_most_edges=source_with_most_edges,
                        messages=[instruction],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        round=f"{_round+1}/{rounds}",
                        final_prompt=final_prompt,
                        seed=seed,
                    )

                    input_token_total += i_t_c
                    output_token_total += o_t_c

                    round_responses = deepcopy(after_t_to_s)

                final_response_dict = round_responses

                if final_response is None:
                    # 4: Graph Pooling
                    logger.info("Graph Pooling initiated.")

                    if graph_pooling_method == 'max':
                        final_response = final_response_dict[source_with_most_edges]
                    elif graph_pooling_method == 'mean':
                        final_response, i_t_c, o_t_c = graph_pooling(
                            model=meta_llm_model_name_graph,
                            endpoint=meta_llm_model_endpoint_graph,
                            refined_response=final_response_dict,
                            weights=score_dict,
                            messages=[instruction],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            final_prompt=final_prompt,
                            seed=seed,
                        )

                        input_token_total += i_t_c
                        output_token_total += o_t_c

            except Exception as e:
                logger.error(f"Error in message passing / graph pooling: {e}")

    if final_response is None:
        final_response = ""

    if data in ['MATH', 'AIME24']:
        answer = remove_boxed(last_boxed_only_string(final_response))
        if answer is None or str(answer).strip() in ("", "N/A"):
            answer = "0"
            logger.warning("[Final Parse] Unparseable MATH answer; defaulting to 0.")

    elif data in ['human_eval']:
        completion = extract_human_eval_completion(final_response)

        result = {
            "task_id": task_id,
            "meta_llm": meta_llm,
            "sampled_nodes": sampled_nodes,
            "final_response": final_response,
            "completion": completion,
        }

        return result

    else:
        if answer is None:
            answer = get_alphabet_choice(final_response, num_choice=num_choice)
        if answer is None or str(answer).strip() in ("", "N/A"):
            answer = np.random.choice([chr(65 + i) for i in range(num_choice)])
            
    end_time = time()
    total_time = end_time - start_time

    result = {
        "idx": item["idx"],
        "instruction": instruction,
        "meta_llm": meta_llm,
        "sampled_nodes": sampled_nodes,
        "final_response": final_response,
        "answer": answer if answer is not None else "N/A",
    }

    return result

def main(
    data: str = "GPQA",
    eval: str = "test",  # 'dev' or 'test'
    reference_models: str = "qwen,qwen_coder,mathstral,biomedical_llama,finance_llama,saul",
    meta_llm: str = "qwen",
    graph_pooling_method: str = "max",
    output_file_name: str = None,
    temperature: float = 0.7,
    max_tokens: int = 800,
    top_k: int = 3,
    rounds: int = 1,
    num_proc: int = 1,
    threshold: float = 0.05,
    seed: int = 0,
):

    seed_everything(seed=seed)

    output_path = f"outputs/{data}/{eval}/"

    global_logger = setup_logger('./logs', f'{data}', f'{data}_{eval}_num_proc_{num_proc}.txt')

    log_summary = "======================================================================================\n"

    model_kwargs = {
        "model": 'GoA',
        "data": data,
        "eval": eval,
        "seed": seed,
        "graph_pooling_method": graph_pooling_method,
        "top_k": top_k,
        "threshold": threshold,
        "rounds": rounds,
        "reference_models": reference_models,
        "meta_llm": meta_llm,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    log_summary += f"Model configuration: {model_kwargs}\n"

    os.makedirs(output_path, exist_ok=True)

    if output_file_name is None:
        if isinstance(reference_models, str):
            reference_models = reference_models.split(',')

        reference_model_names = ",".join(reference_models)
        meta_llm_name = meta_llm.split("/")[-1]
        output_file_name = f"goa_r_{reference_model_names}_m_{meta_llm_name}_pooling_{graph_pooling_method}_top_k_{top_k}_t_{threshold}_n_{rounds}_seed_{seed}"
    output_path += f"{output_file_name}.json"

    if data == 'human_eval':
        output_path += "l"

    print("File will be saved:", output_path)

    if reference_models is None:
        reference_models = []
    else:
        if isinstance(reference_models, str):
            reference_models = reference_models.split(',')

    # Load dataset
    is_math = False
    num_choice = 0
    final_prompt = ''

    if data == "human_eval":
        if eval == 'dev':
            eval_set = load_human_eval_dataset_from_json(os.getcwd() + f'/data/dev/human_eval_dev.json')
        else:
            eval_set = load_human_eval_dataset(eval)

    else:
        data_path = os.getcwd() + f'/data/{eval}/{data}_{eval}.json'
        with open(data_path, 'rb') as f:
            samples = json.load(f)

        if data.lower() == 'mmlu' or data.lower() == 'mmlu_sampled':
            num_choice = 4
            is_math = False
        elif data.lower() == 'mmlu_pro' or data.lower() == 'mmlu_pro_sampled':
            num_choice = 10
            is_math = False
        elif data.lower() in ['gpqa', 'medmcqa', 'medmcqa_sampled']:
            num_choice = 4
            is_math = False
        elif data.lower() == 'aime24':
            num_choice = 0
            is_math = True
        elif data.lower() in ['math', 'math_sampled']:
            num_choice = 0
            is_math = True

        eval_dict = {
                'question': [item['question'] for item in samples],
                'gold_answer': [item['gold_answer'] for item in samples]
        }

        eval_set = Dataset.from_dict(eval_dict)

        if data in ['MATH', 'AIME24']:
            instructions = [
                f"Question: {sample['question']}\n"
                f"Provide your step-by-step reasoning first, and then print \"The answer is \\boxed{{X}}\", "
                f"where X is the final answer, at the end of your response."
                for sample in samples
            ]

            final_prompt = f"Please conclude by printing \"The answer is \\boxed{{X}}\", where X is the final answer, at the end of your response."

        else:
            instructions = [
                f"Question: {sample['question']}\n"
                f"Provide your step-by-step reasoning first, and then print \"The answer is (x)\", "
                f"where x is the answer choice (one capital letter), at the end of your response."
                for sample in samples
            ]

            final_prompt = f"Please conclude by printing \"The answer is (x)\", where x is the answer choice (one capital letter), at the end of your response."

        eval_set = eval_set.add_column("idx", list(range(len(eval_set))))
        eval_set = eval_set.add_column(f"instruction", instructions)

    if len(reference_models):
        logger.info(
            f"`reference_models` provided: {reference_models}. Will generate reference responses on-the-fly."
        )

    logger.info(f"Start.")

    try:
        eval_set = eval_set.map(
            partial(
                process_fn,
                data=data,
                reference_models=reference_models,
                meta_llm=meta_llm,
                graph_pooling_method=graph_pooling_method,
                top_k=top_k,
                threshold=threshold,
                temperature=temperature,
                max_tokens=max_tokens,
                rounds=rounds,
                output_path=output_path,
                num_choice=num_choice,
                final_prompt=final_prompt,
                seed=seed,
            ),
            batched=False,
            num_proc=num_proc,
            load_from_cache_file=False,
            remove_columns=["instruction"],
        )

    except Exception as e:
        logger.error(f"Error during eval_set.map: {e}")

    logger.info(f"Saving outputs to {output_path}.")

    if data == "human_eval":
        with open(output_path, "w") as f:
            for r in eval_set:
                f.write(json.dumps(r) + "\n")

        logger.info(f"HumanEval results written to {output_path}. Evaluating...")
        try:
            pass_at_k = evaluate_functional_correctness(sample_file=output_path, k=[1], ignore_incomplete=(eval == 'dev'))
            human_eval_log = f"[Human-Eval] pass@1: {pass_at_k.get('pass@1', 0.0):.2%}"
            logger.info(human_eval_log)
            log_summary += human_eval_log + "\n"
        except Exception as e:
            logger.error(f"Error during HumanEval evaluation: {e}")
            log_summary += "[Human-Eval] Evaluation failed.\n"

    else:

        with open(output_path, "w") as f:
            json_str = json.dumps(list(eval_set), indent=2)
            f.write(json_str)

        acc = evaluate(eval_set, pred_key='answer', is_math=is_math) * 100
        overall_log = f"Overall Accuracy: {acc:.2f}%\n"
        logger.info(overall_log)
        log_summary += overall_log

    global_logger.info(log_summary)

if __name__ == "__main__":
    Fire(main)
