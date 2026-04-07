 # Graph-of-Agents: A Graph-based Framework for Multi-Agent LLM Collaboration

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![ICLR 2026](https://img.shields.io/badge/ICLR'26-red)](https://neurips.cc/)

Official implementation for "Graph-of-Agents: A Graph-based Framework for Multi-Agent LLM Collaboration" accepted by ICLR 2026.  

- Authors: [Sukwon Yun](https://sukwonyun.github.io/), [Jie Peng](https://scholar.google.com/citations?user=wD7PQt0AAAAJ&hl=EN), [Pingzhi Li](https://pingzhili.github.io/), [Wendong Fan](https://openreview.net/profile?id=~Wendong_Fan1), [Jie Chen](https://jiechenjiechen.github.io/), [James Zou](https://www.james-zou.com/), [Guohao Li](https://ghli.org/), and [Tianlong Chen](https://tianlong-chen.github.io/)


## Overview
With an ever-growing zoo of LLMs and benchmarks, the need to orchestrate multiple models for improved task performance has never been more pressing. While frameworks like Mixture-of-Agents (MoA) attempt to coordinate LLMs, they often fall short in terms of (1) selecting relevant agents, (2) facilitating effective intra-agent communication, and (3) integrating responses efficiently. In this work, we propose Graph-of-Agents (GoA), a new graph-based framework for modeling multi-agent LLM communication. Our approach begins with node sampling, selecting only the most relevant agents by leveraging model cards that summarize each model’s domain, task specialization, and other characteristics. Next, we construct edges between the selected agents by evaluating their responses against one another to determine relevance ordering. Directed message passing is then performed from highly relevant agents to less relevant ones to enhance their responses, followed by reverse message passing to refine the original responses of the more relevant agents. Finally, the updated responses are aggregated via graph-based pooling (e.g., max or mean pooling) to produce a single, unified answer. We evaluate GoA on diverse multi-domain benchmarks (MMLU, MMLU-Pro, GPQA) and domain-specific benchmarks (MATH, HumanEval, MedMCQA), with an agent pool of 6 LLMs spanning multiple domains. Surprisingly, GoA achieves superior performance18 using only 3 selected agents, outperforming recent multi-agent LLM baselines that utilize all 6 agents simultaneously. By adopting a graph structure, GoA offers both scalability and effectiveness through structured message passing—positioning it as a strong candidate for navigating the challenges of the ever-growing LLM zoo.

<img src="assets/model.png" width="100%">


## Code Release (Scheduled): 🗓️ April 11, 2026