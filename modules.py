import re
import json
import numpy as np
from utils import generate_vllm, extract_numbers_as_ints, remove_boxed, last_boxed_only_string, parse_confidence_response
from copy import deepcopy
from loguru import logger

CONFIDENCE_THRESHOLD = 0.1  # Replace model if confidence < this or N/A

def generate_initial_response_with_confidence(
    model_id, endpoint, instruction, data=None, num_choice=4,
    temperature=0.7, max_tokens=800, seed=0, debug_txt=""
):
    """
    Generate initial response with confidence using JSON-formatted prompt.
    Returns (raw_response, answer, confidence, input_tokens, output_tokens)
    """

    # Build confidence-aware prompt
    if data in ['MATH', 'AIME24']:
        answer_format_hint = 'a mathematical expression or number (e.g., "42" or "3/4")'
    elif data in ['human_eval']:
        answer_format_hint = 'the completed Python function code'
    else:
        choices_str = ', '.join([chr(65 + i) for i in range(num_choice)])
        answer_format_hint = f'one of the answer choices: {choices_str}'

    confidence_instruction = (
        f"\n\nProvide brief reasoning (2-3 key sentences), then output your final answer in JSON format:\n"
        f'{{"reasoning": "<brief reasoning>", '
        f'"answer": "<{answer_format_hint}>", '
        f'"confidence_level": "<a float between 0.0 and 1.0>"}}\n'
        f"Please strictly output in JSON format."
    )

    augmented_instruction = instruction + confidence_instruction

    raw_response, _, _, i_t_c, o_t_c = generate_vllm(
        model=model_id,
        messages=[augmented_instruction],
        temperature=temperature,
        max_tokens=max_tokens,
        endpoint=endpoint,
        debug_txt=debug_txt,
        seed=seed
    )

    # Parse the response
    _, answer, confidence = parse_confidence_response(
        raw_response, data=data, num_choice=num_choice
    )

    return raw_response, answer, confidence, i_t_c, o_t_c


def maybe_replace_with_general(
    node_idx, node_name, confidence, models_dict, general_model_name,
    initial_responses, confidences, answers, sampled_nodes,
    instruction, data, num_choice, temperature, max_tokens, seed
):
    """
    If confidence is N/A (< 0) or below threshold, replace this node with general model.
    Returns (replaced, new_input_tokens, new_output_tokens)
    """
    should_replace = (confidence < 0) or (confidence < CONFIDENCE_THRESHOLD)

    if not should_replace:
        return False, 0, 0

    # Don't replace if this node IS the general model already
    if node_name == general_model_name:
        logger.info(f"[Confidence Replace] Node {node_idx} ({node_name}) has low confidence "
                     f"({confidence}) but is already the general model. Keeping.")
        return False, 0, 0

    logger.info(f"[Confidence Replace] Node {node_idx} ({node_name}) confidence={confidence:.3f} "
                 f"< {CONFIDENCE_THRESHOLD}. Replacing with {general_model_name}.")

    general_dict = models_dict[general_model_name]

    raw_response, answer, new_confidence, i_t_c, o_t_c = generate_initial_response_with_confidence(
        model_id=general_dict["model_id"],
        endpoint=general_dict["url"],
        instruction=instruction,
        data=data,
        num_choice=num_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        debug_txt=f"[Replacement - {general_model_name} for {node_name}] "
    )

    initial_responses[node_idx] = raw_response
    confidences[node_idx] = new_confidence
    answers[node_idx] = answer
    sampled_nodes[node_idx] = general_model_name

    return True, i_t_c, o_t_c

def node_sampling_model_card(model, endpoint, messages, model_info_dict, top_k=4, temperature=0.7, max_tokens=512, seed=0):
    """
    Sample top-K relevant models based on the question and model descriptions.
    Simplified prompt: single combined instruction, no repeated criteria.
    """

    input_token_total = 0
    output_token_total = 0
    node_sampling_fallback = False

    try:
        model_descriptions = [{"index": idx, "description": model_info_dict[info]["model_card"]} for idx, info in enumerate(model_info_dict)]
        max_index = len(model_descriptions) - 1

        invalid_attempts = []

        if top_k == -1:
            # Determine difficulty to set top_k
            difficulty_prompt = {
                "role": "user",
                "content": (
                    f"Rate the difficulty of this question from 1 (very easy) to 6 (very hard).\n\n"
                    f"Question: {messages[0]}\n\n"
                    f"Reply with a single number only."
                )
            }

            difficulty_response, _, _, input_token_count, output_token_count = generate_vllm(
                model=model,
                endpoint=endpoint,
                messages=[difficulty_prompt],
                max_tokens=10,
                temperature=temperature,
                debug_txt="[Node Sampling - Difficulty] ",
                seed=seed
            )

            input_token_total += input_token_count
            output_token_total += output_token_count

            difficulty_level = extract_numbers_as_ints(difficulty_response)
            if difficulty_level and 1 <= difficulty_level[0] <= 6:
                top_k = difficulty_level[0]
            else:
                top_k = 3

            print(f"[Node Sampling] Determined Difficulty: {difficulty_level} → Selecting {top_k} models.")


        max_attempts = 5
        attempts = 0

        # Build example based on top_k
        example_dict = {
            1: "0",
            2: "0,3",
            3: "0,1,5",
            4: "0,0,4,5",
            5: "0,1,2,3,5",
            6: "0,0,2,3,4,5"
        }

        while attempts < max_attempts:
            # --- SIMPLIFIED PROMPT: single user message, concise criteria ---
            prompt_content = (
                f"Select {top_k} models best suited for this question.\n\n"
                f"Question: {messages[0]}\n\n"
                f"Available models:\n{json.dumps(model_descriptions, indent=None)}\n\n"
                f"Selection criteria (in priority order):\n"
                f"1. Domain match — prefer models trained in the question's domain\n"
                f"2. Task specialization — prefer models fine-tuned for the required skill\n"
                f"3. Include at least one generalist model if applicable\n"
                f"4. Prefer larger models only when the size gap is significant\n\n"
                f"Rules:\n"
                f"- Output exactly {top_k} comma-separated indices from [0, {max_index}]\n"
                f"- You may repeat an index if the model is highly relevant\n"
                f"- No explanations\n\n"
                f"Example: {example_dict[top_k]}\n\n"
                f"Answer:"
            )

            if invalid_attempts:
                prompt_content += f"\n(Previous invalid responses: {invalid_attempts}. Fix this.)"

            prompt = [{"role": "user", "content": prompt_content}]

            sampled_indices, _, _, input_token_count, output_token_count = generate_vllm(
                model=model,
                endpoint=endpoint,
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                debug_txt="[Node Sampling] ",
                seed=seed
            )

            input_token_total += input_token_count
            output_token_total += output_token_count

            # Validate extracted indices
            pattern = re.compile(
                r'</think>\s*([\d,]+)'
                r'|(?:Answer:\s*|the answer is:\s*\n?)(\d+)'
                r'|^\s*([\d,]+)\s*$',
                re.IGNORECASE | re.MULTILINE
            )

            match = pattern.search(sampled_indices)

            if match:
                extracted = match.group(1) or match.group(2) or match.group(3)

                if ',' in extracted:
                    sampled_indices = list(map(int, extracted.split(',')))
                else:
                    sampled_indices = [int(extracted)]

                if (0 <= np.array(sampled_indices)).all() and (np.array(sampled_indices) <= max_index).all():
                    break

            # Store invalid response for feedback in the next iteration
            attempts += 1
            invalid_attempts.append(sampled_indices)
            print(f"[Node Sampling] Invalid Answer: {sampled_indices}. Retrying...")


        if attempts == max_attempts:
            sampled_indices = list(np.arange(top_k))
            node_sampling_fallback = True
            print(f"[Node Sampling] Using fallback: selecting first {top_k} models")

        # Convert extracted indices to model names
        models_list = list(model_info_dict.keys())

        try:
            sampled_nodes = list(np.array(models_list)[sampled_indices])
        except Exception as e:
            print(f"Error mapping indices to models: {e}")
            sampled_nodes = [np.array(models_list)[0]]
            node_sampling_fallback = True

        return sampled_nodes, node_sampling_fallback, input_token_total, output_token_total

    except Exception as e:
        print(f"Error in node_sampling: {e}")
        return None


def parse_ranked_scores(text):
    text = text.strip()[:2000]

    # Case 1: Try to extract from a list-style bracketed structure like [0.7, 0.3]
    try:
        match = re.search(r'\[([0-9.,\s]+)\]', text)
        if match:
            tokens = re.split(r'[,\s]+', match.group(1))
            return [float(t) for t in tokens if t]
    except Exception as e:
        print(f"[parse_ranked_scores] List-style parsing error: {e}")

    # Case 2: Dictionary-style parsing like '0': 0.7, '1': 0.3 or * 1: 0.7
    try:
        dict_match = re.findall(r"[\*\-]?\s*'?(\d+)'?\s*:\s*([0-9]*\.?[0-9]+)", text)
        if dict_match:
            sorted_items = sorted([(int(k), float(v)) for k, v in dict_match], key=lambda x: x[0])
            return [v for _, v in sorted_items]
    except Exception as e:
        print(f"[parse_ranked_scores] Dict-style parsing error: {e}")

    # Case 3: Fallback - extract loose floats
    try:
        return [float(t) for t in re.split(r'[,\s]+', text) if t]
    except Exception as e:
        print(f"[parse_ranked_scores] Fallback parsing error: {e}")
        return []

def edge_sampling(models_dict, model_name_idx, initial_responses, messages, temperature=0.7, max_tokens=512, max_retries=3, threshold=0.1, round=None, seed=0, confidences=None):
    """
    Generate edges (relationships) between sampled models and assign scores summing to 1.0.
    Each model scores all other models' responses; scores are aggregated and normalized.
    """

    try:
        models = list(initial_responses.keys())
        total_scores = {model: 0.0 for model in models}
        n_models = len(models)

        # Initialize score matrix (n_models x n_models) with -1 on diagonal
        score_matrix = np.full((n_models, n_models), np.nan)
        np.fill_diagonal(score_matrix, -1)

        # Track which models used fallback scores
        fallback_used = {model: False for model in models}
        judge_count = {model: 0 for model in models}

        for self_model in models:
            other_models = [m for m in models if m != self_model]
            other_responses = {m: initial_responses[m] for m in other_models}

            example_dict = {
                1: [1.0],
                2: [0.7, 0.3],
                3: [0.1, 0.1, 0.8],
                4: [0.3, 0.4, 0.1, 0.2],
                5: [0.2, 0.4, 0.1, 0.3, 0.0]
            }

            example_str = f"{example_dict[len(other_models)]}"
            invalid_attempts = []
            retry_count = 0

            while retry_count < max_retries:
                # --- SIMPLIFIED PROMPT: single user message, clear format ---
                prompt_content = (
                    f"Score the following {len(other_models)} model responses to this question.\n\n"
                    f"Question: {messages[0]}\n\n"
                    f"Responses (in order: {other_models}):\n{json.dumps(other_responses, default=str)}\n\n"
                    f"Assign a score to each response based on correctness, coherence, and relevance.\n"
                    f"Scores must sum to exactly 1.0. Output only a comma-separated list of {len(other_models)} scores.\n\n"
                    f"Example: {example_str}\n\n"
                    f"Answer:"
                )

                if invalid_attempts:
                    prompt_content += (
                        f"\n(Previous invalid responses: {invalid_attempts}. "
                        f"Must have exactly {len(other_models)} scores summing to 1.0.)"
                    )

                prompt = [{"role": "user", "content": prompt_content}]

                model_name = model_name_idx[self_model]

                ranked_scores, _, _, input_token_count, output_token_count = generate_vllm(
                    model=models_dict[model_name]['model_id'],
                    endpoint=models_dict[model_name]['url'],
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    debug_txt=f"[Edge Sampling - {model_name}]",
                    seed=seed
                    )

                try:
                    score_list = parse_ranked_scores(ranked_scores)

                    if len(score_list) == len(other_models) and (0.9 < sum(score_list) <= 1.0):
                        for key, value in zip(other_models, score_list):
                            total_scores[key] += value
                            judge_count[key] += 1

                        # Fill in score matrix
                        self_idx = self_model
                        for other_model, score in zip(other_models, score_list):
                            other_idx = other_model
                            score_matrix[self_idx, other_idx] = score

                        break
                    else:
                        print(f"[Edge Sampling - {model_name}] Invalid score list: {score_list}, Key should be: {other_models}")

                except Exception as e:
                    print(f"[Edge Sampling - {model_name}] Parse error: {e}, Key should be: {other_models}")

                invalid_attempts.append(ranked_scores)
                retry_count += 1
                print(f"[Edge Sampling - {model_name}] Invalid Answer: {ranked_scores}. Key should be: {other_models} Retrying ({retry_count}/{max_retries})...")

            if retry_count == max_retries:
                print(f"[Edge Sampling - {model_name}] Max retries reached. Dropping this model's scores (response kept).")
                fallback_used[self_model] = True

        # Average scores per model (handles dropped judges fairly)
        avg_scores = {k: (total_scores[k] / judge_count[k] if judge_count[k] > 0 else 0.0) for k in total_scores}

        # Normalize to sum to 1.0
        normalization_factor = 1 / sum(avg_scores.values()) if sum(avg_scores.values()) > 0 else 1.0
        final_scores = {k: v * normalization_factor for k, v in avg_scores.items()}

        # Check if any model used fallback scoring
        edge_sampling_fallback_global = any(fallback_used.values())

        # Sort final ranking
        ranked_scores_dict = dict(sorted(final_scores.items(), key=lambda item: float(item[1]), reverse=True))
        models = list(ranked_scores_dict.keys())
        scores = list(ranked_scores_dict.values())

        edges = []
        total_num_edges = len(models) * (len(models)-1)

        filtered_idx = np.array(scores) > threshold
        models = np.array(models)[filtered_idx]
        scores = np.array(scores)[filtered_idx]
        score_dict = dict(zip(models, scores))

        if len(models) <= 1:
            if len(models) == 1:
                return initial_responses[models[0]]
            else:
                return -1

        # Build edges from ranked models
        for s_idx in range(len(models) - 1):
            source_node = models[s_idx]
            for target_node in models[s_idx+1:]:
                edges.append((source_node, target_node))

        print(f'# of Edges: {len(edges)*2} / {total_num_edges}, \t Pruned: {((total_num_edges - len(edges)*2) * 100 / total_num_edges):.2f}%')

        # Initialize edge dictionaries
        source_edges = {}
        target_edges = {}

        for (source, target) in edges:
            if target not in target_edges:
                target_edges[target] = []
            target_edges[target].append(source)

            if source not in source_edges:
                source_edges[source] = []
            source_edges[source].append(target)

        return source_edges, target_edges, score_dict, score_matrix, fallback_used, edge_sampling_fallback_global, input_token_count, output_token_count

    except Exception as e:
        print(f'Error: {e}')


def _build_reference_descriptions(sources_or_targets, weights, initial_responses):
    """
    Build reference descriptions for message passing.
    Annotates each response with a relevance label (high / moderate / low)
    derived from normalized edge scores.
    """
    descriptions = []
    for ref, weight in zip(sources_or_targets, weights):
        if weight > 0.7:
            desc = f"Model {ref} (high relevance):"
        elif weight > 0.4:
            desc = f"Model {ref} (moderate relevance):"
        else:
            desc = f"Model {ref} (low relevance):"
        descriptions.append(f"{desc}\n{initial_responses[ref]}")
    return "\n\n".join(descriptions)


def _run_s_to_t(model_endpoint_dict, model_name_idx, initial_responses, target_edges, score_dict,
                final_response_dict, messages, temperature, max_tokens,
                round, final_prompt, seed):
    """Source -> Target: each target refines its response using source models' responses."""
    input_token_total = 0
    output_token_total = 0

    sorted_targets = sorted(target_edges.keys())
    for target in sorted_targets:

        sources = target_edges[target]
        source_scores = [score_dict[source] for source in sources]
        total_score = sum(source_scores)
        source_weights = [s / total_score for s in source_scores]

        descriptions = _build_reference_descriptions(sources, source_weights, initial_responses)

        # --- SIMPLIFIED PROMPT: single user message ---
        prompt_content = (
            f"Refine your answer by considering other models' responses.\n\n"
            f"Question: {messages[0]}\n\n"
            f"Your initial response: {initial_responses[target]}\n\n"
            f"Other models' responses (ranked by relevance):\n{descriptions}\n\n"
            f"Integrate useful insights from these responses to improve your answer. "
            f"Be critical — some information may be incorrect.\n"
        )

        prompt = [{"role": "user", "content": prompt_content}]

        target_model_name = model_name_idx[target]
        model = model_endpoint_dict[target_model_name]['model_id']
        endpoint = model_endpoint_dict[target_model_name]['url']

        refined_target_response, _, _, input_token_count, output_token_count = generate_vllm(
            model=model,
            endpoint=endpoint,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            debug_txt=f"[Round {round}] [(S -> T): Refining Target] ",
            seed=seed
        )

        input_token_total += input_token_count
        output_token_total += output_token_count

        final_response_dict[target] = refined_target_response

    return final_response_dict, input_token_total, output_token_total


def _run_t_to_s(model_endpoint_dict, model_name_idx, initial_responses, source_edges, score_dict,
                final_response_dict, source_with_most_edges, messages, temperature, max_tokens,
                round, final_prompt, seed):
    """Target -> Source: each source refines its response using target models' refined responses."""
    input_token_total = 0
    output_token_total = 0
    final_user_prompt = final_prompt if source_with_most_edges else ''

    sorted_sources = sorted(source_edges.keys())

    for source in sorted_sources:
        if source_with_most_edges and source != source_with_most_edges:
            continue

        targets = source_edges[source]
        target_scores = [score_dict[target] for target in targets]
        total_score = sum(target_scores)
        target_weights = [t / total_score for t in target_scores]

        descriptions = _build_reference_descriptions(targets, target_weights, final_response_dict)

        # --- SIMPLIFIED PROMPT: single user message ---
        prompt_content = (
            f"Other models refined their answers after seeing yours. Use their improvements to finalize your response.\n\n"
            f"Question: {messages[0]}\n\n"
            f"Your initial response: {initial_responses[source]}\n\n"
            f"Updated responses from other models:\n{descriptions}\n\n"
            f"Write your final response, incorporating valuable refinements. "
            f"Be critical — some information may be incorrect.\n"
            f"{final_user_prompt}"
        )

        prompt = [{"role": "user", "content": prompt_content}]

        source_model_name = model_name_idx[source]
        model = model_endpoint_dict[source_model_name]['model_id']
        endpoint = model_endpoint_dict[source_model_name]['url']

        refined_source_response, _, _, input_token_count, output_token_count = generate_vllm(
            model=model,
            endpoint=endpoint,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            debug_txt=f"[Round {round}] [(T -> S): Updating Source] ",
            seed=seed
        )

        input_token_total += input_token_count
        output_token_total += output_token_count

        final_response_dict[source] = refined_source_response

    return final_response_dict, input_token_total, output_token_total


def message_passing(model_endpoint_dict, model_name_idx, initial_responses, source_edges, target_edges, score_dict,
                     source_with_most_edges=None, messages=None,
                     temperature=0.7, max_tokens=512, round=None, final_prompt='', seed=0):
    """
    Pass messages between nodes (models) to iteratively improve responses.
    Bidirectional: S->T then T->S.
    """
    final_response_dict = {}
    input_token_total = 0
    output_token_total = 0

    try:
        # Step 1: S -> T
        final_response_dict, i_t_c, o_t_c = _run_s_to_t(
            model_endpoint_dict, model_name_idx, initial_responses, target_edges, score_dict,
            final_response_dict, messages, temperature, max_tokens,
            round, final_prompt, seed)
        input_token_total += i_t_c
        output_token_total += o_t_c

        after_phase1 = deepcopy(final_response_dict)

        # Step 2: T -> S
        final_response_dict, i_t_c, o_t_c = _run_t_to_s(
            model_endpoint_dict, model_name_idx, initial_responses, source_edges, score_dict,
            final_response_dict, source_with_most_edges, messages, temperature, max_tokens,
            round, final_prompt, seed)
        input_token_total += i_t_c
        output_token_total += o_t_c

        return after_phase1, final_response_dict, input_token_total, output_token_total

    except Exception as e:
        print(f"Error in message passing: {e}")



def graph_pooling(model, endpoint, refined_response, messages, weights=None, temperature=0.7, max_tokens=512, final_prompt='', seed=0):
    """
    Aggregate responses from all nodes into a single final response.
    """

    try:
        response_parts = []
        if weights:
            for node, response in refined_response.items():
                weight = weights[node]
                if weight > 0.7:
                    header = f"Model {node} (high relevance):"
                elif weight > 0.4:
                    header = f"Model {node} (moderate relevance):"
                else:
                    header = f"Model {node} (low relevance):"
                response_parts.append(f"{header}\n{response}")
            debug_txt = "[Graph Pooling - Weighted Mean] "
        else:
            for node, response in refined_response.items():
                response_parts.append(f"Model {node}:\n{response}")
            debug_txt = "[Graph Pooling - Mean] "

        input_responses = "\n\n".join(response_parts)

        # --- SIMPLIFIED PROMPT: single user message ---
        prompt_content = (
            f"Synthesize these model responses into one final answer.\n\n"
            f"Question: {messages[0]}\n\n"
            f"Model responses:\n{input_responses}\n\n"
            f"Produce an accurate, coherent answer integrating the best insights. "
            f"Be critical — some information may be incorrect.\n"
            f"{final_prompt}"
        )

        prompt = [{"role": "user", "content": prompt_content}]

        final_response, _, _, input_token_count, output_token_count = generate_vllm(
            model=model,
            endpoint=endpoint,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            debug_txt=debug_txt,
            seed=seed
        )

        return final_response, input_token_count, output_token_count

    except Exception as e:
        print(f'Error: {e}')
