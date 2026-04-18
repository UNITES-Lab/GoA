import os
import json
import time
import requests
from transformers import AutoTokenizer, AutoConfig
from loguru import logger
from math_verify import parse, verify
import numpy as np
import re
import torch
import logging
import random

DEBUG = True

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))

def setup_logger(log_path, log_name, file_name):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(log_path, file_name))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def is_math_equiv(ref, pred):
    # Test math equivalence of ref and pred,
    # can also handle answer choices e.g., A vs. (A)
    try:
        if any([verify(parse(f"${ref}$"), parse(f"${pred}$")),
               verify(parse(ref), parse(pred)),
               verify(parse(ref), parse(pred.replace("\\(", "").replace("\\)", "")))]):
            return True
    except:
        return False
    return False

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return s

def last_boxed_only_string(string):
    if not string: return "N/A"
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = string
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def evaluate(test_samples, pred_key='pred', is_math=False):
    num_correct = 0
    for i in test_samples:
        if is_math:
            if is_math_equiv(str(i[pred_key]), str(i['gold_answer'])):
                 num_correct += 1
        else:
            if i[pred_key] == i['gold_answer']:
                num_correct += 1
    acc = round(num_correct / len(test_samples), 4)
    return acc

def get_alphabet_choice(text, num_choice=4):
    choices = '|'.join([chr(65 + i) for i in range(num_choice)])
    match = False
    if text:
        # First try to match with parentheses
        match = re.findall(f'([{choices}])\)', text)
        if not match:
            # If no match with parentheses, try without
            match = re.findall(f'([{choices}])', text)
    return match[-1] if match else "N/A"


def parse_confidence_response(response_text, data=None, num_choice=4):
    """
    Parse a JSON-formatted response containing reasoning, answer, and confidence_level.
    Returns (raw_response_text, answer, confidence) where:
      - raw_response_text: the full original response (for message passing)
      - answer: extracted answer (letter for MCQ, boxed for math, code for HumanEval)
      - confidence: float 0.0-1.0, or -1 if extraction failed

    Handles multiple formats:
      1. Clean JSON: {"reasoning": "...", "answer": "...", "confidence_level": "..."}
      2. JSON embedded in text (with preamble or markdown code blocks)
      3. Fallback: regex extraction from free-form text
    """
    if response_text is None:
        return None, "N/A", -1.0

    answer = None
    confidence = -1.0

    # Try to extract JSON from the response
    json_obj = None

    # Strategy 1: Direct JSON parse
    try:
        json_obj = json.loads(response_text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Find JSON block in markdown code fence
    if json_obj is None:
        json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_block_match:
            try:
                json_obj = json.loads(json_block_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

    # Strategy 3: Find first {...} in the text
    if json_obj is None:
        brace_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', response_text, re.DOTALL)
        if brace_match:
            try:
                json_obj = json.loads(brace_match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

    # Strategy 4: Find nested JSON (reasoning may contain braces)
    if json_obj is None:
        # Find the outermost { ... } that contains "answer"
        start = response_text.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(response_text)):
                if response_text[i] == '{':
                    depth += 1
                elif response_text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            json_obj = json.loads(response_text[start:i+1])
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break

    if json_obj is not None:
        # Extract answer
        raw_answer = json_obj.get("answer", None)
        if raw_answer is not None:
            raw_answer = str(raw_answer).strip()

            if data in ['MATH', 'AIME24']:
                # For math, the answer could be a number or expression
                answer = raw_answer
            elif data in ['human_eval']:
                answer = raw_answer
            else:
                # For MCQ, extract the letter
                choices = [chr(65 + i) for i in range(num_choice)]
                # Try to find a single letter answer
                letter_match = re.search(r'([A-Z])', raw_answer.upper())
                if letter_match and letter_match.group(1) in choices:
                    answer = letter_match.group(1)
                else:
                    answer = None

        # Extract confidence
        raw_confidence = json_obj.get("confidence_level", json_obj.get("confidence", None))
        if raw_confidence is not None:
            try:
                confidence = float(str(raw_confidence).strip())
                if confidence < 0 or confidence > 1:
                    confidence = -1.0
            except (ValueError, TypeError):
                confidence = -1.0

    # Fallback: regex extraction if JSON parsing failed
    if answer is None:
        if data in ['MATH', 'AIME24']:
            answer = remove_boxed(last_boxed_only_string(response_text))
        elif data in ['human_eval']:
            answer = extract_human_eval_completion(response_text)
        else:
            answer = get_alphabet_choice(response_text, num_choice=num_choice)

    if confidence < 0:
        # Try regex extraction of confidence from free text
        conf_match = re.search(r'"confidence_level"\s*:\s*"?([\d.]+)"?', response_text)
        if not conf_match:
            conf_match = re.search(r'confidence[:\s]+([01]\.?\d*)', response_text, re.IGNORECASE)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                if confidence < 0 or confidence > 1:
                    confidence = -1.0
            except (ValueError, TypeError):
                confidence = -1.0

    return response_text, answer, confidence


def extract_human_eval_completion(response: str) -> str:
    """
    Extract only the valid function implementation for HumanEval completion.

    Args:
        response (str): The model's raw output.

    Returns:
        str: The extracted function implementation.
    """
    try:
        # 1. Try to extract from ```json { "answer": "..." } ``` blocks
        json_match = re.search(r"```json\s*\n(.*?)```", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1).strip())
                if isinstance(parsed, dict) and "answer" in parsed:
                    response = parsed["answer"]
            except json.JSONDecodeError:
                pass

        # 2. Try to extract from inline "answer": "..." (without json fence)
        if not json_match or "answer" not in (json_match.group(1) if json_match else ""):
            inline_match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL)
            if inline_match and not json_match:
                try:
                    answer_str = json.loads('"' + inline_match.group(1) + '"')
                    # Only use if it looks like code (contains def or return or common Python)
                    if re.search(r'\b(def |return |for |if |while |import )', answer_str):
                        response = answer_str
                except:
                    pass

        # 3. Try to extract from ```python ... ``` blocks
        python_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if python_match:
            response = python_match.group(1).strip()

        # 4. Try to extract from generic ``` ... ``` blocks if response still looks non-code
        if not re.match(r'\s*(from |import |def |class |\s)', response):
            generic_match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
            if generic_match:
                candidate = generic_match.group(1).strip()
                if re.search(r'\b(def |return |for |if )', candidate):
                    response = candidate

        lines = response.split("\n")
        filtered_lines = []
        for line in lines:
            stripped_line = line.strip()

            if stripped_line.startswith("print("):
                continue

            if re.match(r"^#\s*Test cases", stripped_line, re.IGNORECASE):
                continue

            filtered_lines.append(line)

        return "\n".join(filtered_lines).strip()
    except:
        return ""


def generate_vllm(
    model,
    messages,
    max_tokens=2048,
    temperature=0.8,
    streaming=False,
    endpoint=None,
    seed=0,
    debug_txt=""
):

    output = None
    case_1 = 0
    case_2 = 0

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        try:
            if DEBUG:
                if isinstance(messages[0], str):
                    logger.debug(
                    f"{debug_txt}Sending messages (`{messages[-1][90:120]}...`) to `{model}`."
                    )
                else:
                    logger.debug(
                    f"{debug_txt}Sending messages (`{messages[-1]['content'][-1][90:120]}...`) to `{model}`."
                    )


            if model in ['THUDM/glm-4-9b-chat', 'internlm/internlm3-8b-instruct', 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct']:
                tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model)

            if isinstance(messages[0], str):
                msg = [{"role": "user", "content": messages[0]}]

                if tokenizer.chat_template is None: # 'instruction-pretrain/finance-Llama3-8B'
                    prompt = ""
                    prompt += f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    prompt += f"<|im_start|>user\n{messages[0]}<|im_end|>\n"
                    prompt += "<|im_start|>assistant\n"

                else:
                    prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

            else:
                og_prompt = None
                msg = []

                if tokenizer.chat_template is None: # 'instruction-pretrain/finance-Llama3-8B'
                    prompt = ""
                    for message in messages:
                        if message['role'] == 'system':
                            prompt += f"<|im_start|>system\n{message['content']}<|im_end|>\n"
                        elif message['role'] == 'user':
                            prompt += f"<|im_start|>user\n{message['content']}<|im_end|>\n"

                        msg.append(message)

                    prompt += "<|im_start|>assistant\n"

                else:
                    for message in messages:
                        if message['role'] == 'og_user':
                            tmp_msg = {}
                            tmp_msg['role'] = 'user'
                            tmp_msg['content'] = message['content']
                            og_content = message['content']
                            og_prompt = tokenizer.apply_chat_template(
                                [tmp_msg],
                                tokenize=False,
                                add_generation_prompt=True
                            )
                        else:
                            msg.append(message) # [{"role": "system", "content": ...}, {"role": "user", "content": ...}]

                    prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

            input_token_count = count_tokens(prompt, tokenizer)

            try:
                res = requests.post(
                    endpoint,
                    json={
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": (temperature if temperature > 1e-4 else 0),
                        "prompt": prompt,
                        "seed": seed
                    },
                    timeout=300
                )

                res.raise_for_status()
                res = res.json()

            except Exception as e:
                # max_token legnth issue
                logger.debug(e)
                refine_prompt = False
                if model in ['THUDM/glm-4-9b-chat', 'internlm/internlm3-8b-instruct', 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct']:
                    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
                else:
                    config = AutoConfig.from_pretrained(model)
                max_token_length = config.max_position_embeddings

                input_token_length = input_token_count

                if input_token_length < max_token_length:
                    logger.warning("Input Prompt Length is Fine, Reducing Output Token Length...")
                    # input prompt is fine, reduce output token
                    max_tokens = min(max_tokens, max_token_length - input_token_length)
                    # max_tokens = max_token_length - input_token_length
                    case_1 += max_tokens

                    if max_tokens < 300:
                        # too short for reasoning
                        logger.warning(f"Number of Remaining Token, {max_tokens} < 300, is too small for reasoning...")
                        refine_prompt = True

                else:
                    refine_prompt = True

                if refine_prompt:

                    logger.warning(f"Truncating Prompt...")
                    # input prompt is too long, reduce system prompt
                    max_input_length = max_token_length - max_tokens

                    try:
                        if og_prompt is None:
                            if len(msg) > 1:
                                user_length = len(tokenizer.encode(msg[1]['content'])) # user content
                            elif len(msg) == 1:
                                user_length = len(tokenizer.encode(msg[0]['content'])) # user content

                            max_system_length = max_input_length - user_length - 100 # 20 as buffer
                            case_2 += max_system_length

                            if max_system_length < 0:
                                logger.warning("System message is too long, even after reduction.")
                                max_system_length = 0

                            system_content = tokenizer.encode(msg[0]['content'])
                            truncated_tokens = system_content[:max_system_length]
                            new_system_content = tokenizer.decode(truncated_tokens)

                            msg[0]['content'] = new_system_content
                        else:
                            og_user_length = len(tokenizer.encode(og_prompt))
                            max_user_length = max_input_length - og_user_length - 20 # 20 as buffer
                            case_2 += max_user_length

                            if max_user_length < 0:
                                logger.warning("System message is too long, even after reduction.")
                                max_user_length = 0

                            user_content = tokenizer.encode(msg[0]['content'])
                            truncated_tokens = user_content[-max_user_length:]
                            new_user_content = tokenizer.decode(truncated_tokens)

                            msg[0]['content'] = og_content + new_user_content

                    except Exception as e:
                        print(e)

                    if tokenizer.chat_template is None: # 'instruction-pretrain/finance-Llama3-8B'
                        prompt = ""
                        prompt += f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        prompt += f"<|im_start|>user\n{msg[1]['content'][:max_input_length]}<|im_end|>\n"
                        prompt += "<|im_start|>assistant\n"

                    else:
                        prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

                res = requests.post(
                    endpoint,
                    json={
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": (temperature if temperature > 1e-4 else 0),
                        "prompt": prompt,
                        "seed": seed,
                    },
                    timeout=300
                )

                res.raise_for_status()
                res = res.json()

            output = [r['text'] for r in res['choices']]

            answer = output[0]

            output_token_count = count_tokens(answer, tokenizer)

            if DEBUG:
                logger.debug(
                f"{debug_txt} Answer: `{str(answer)[-30:]}`."
                )

            return answer, case_1, case_2, input_token_count, output_token_count

        except Exception as e:
            logger.error(e)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")

            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    return None, case_1, case_2, input_token_count, 0


def extract_numbers_as_ints(text):
    """
    Extracts all numeric values from a given text and returns them as a list of integers.

    Args:
        text (str): The input text from which to extract numbers.

    Returns:
        List[int]: A list of extracted integers.
    """
    numbers = re.findall(r'\d+', text)  # Find all numeric sequences
    return list(map(int, numbers)) if numbers else -1
