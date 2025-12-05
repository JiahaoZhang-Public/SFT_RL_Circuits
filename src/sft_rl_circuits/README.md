# **sft_rl_circuits**

# **Package Overview**

This directory contains **all reusable, modular code** for the project

> “SFT Memorizes, RL Generalizes: circuit-level analysis on a small GPT-2.”
> 
- **Training & inference** use **Hugging Face** (transformers, trl, etc.).
- **Mechanistic analysis** uses **TransformerLens**.
- **Executable scripts** live in scripts/py/ and should *only* import utilities from this package – no experiment logic lives directly here.

The goal is that src/sft_rl_circuits/ behaves like a small library, while scripts/py/ are “notebooks in code” that orchestrate specific experiments.

---

## **Design Principles**

1. **Separation of concerns**
    - Task definition vs. data generation vs. training vs. analysis vs. IO are in different modules.
2. **Standard libraries over custom frameworks**
    - Use transformers for SFT, trl (or similar) for RL, transformer_lens for circuit analysis.
3. **Experiment scripts stay thin**
    - Scripts in scripts/py should mostly do:
        1. Parse config / CLI args
        2. Instantiate config objects
        3. Call high-level APIs exposed here.
4. **Analysis-centric**
    - The package is structured to make **mechanistic analysis first-class**, not an afterthought.

---

## **Proposed Package Structure**

Below is the intended high-level layout **inside** src/sft_rl_circuits/.

(Exact filenames can be adjusted, but the conceptual modules should stay.)

```
sft_rl_circuits/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── task_config.py
│   ├── training_config.py
│   └── analysis_config.py
├── tasks/
│   ├── __init__.py
│   ├── rules.py
│   ├── generators.py
│   ├── dataset.py
│   └── formatting.py
├── models/
│   ├── __init__.py
│   ├── lm_loader.py
│   ├── policy_wrappers.py
│   └── conversion.py
├── envs/
│   ├── __init__.py
│   └── text_game_env.py
├── training/
│   ├── __init__.py
│   ├── sft_trainer.py
│   ├── rl_trainer.py
│   └── evaluation.py
├── analysis/
│   ├── __init__.py
│   ├── hooks.py
│   ├── probing.py
│   ├── patching.py
│   ├── ablation.py
│   └── sae_features.py
└── utils/
    ├── __init__.py
    ├── paths.py
    ├── logging.py
    └── serialization.py
```

Below is what each submodule is responsible for.

---

## **config/**

## **– typed configuration for tasks, training, and analysis**

**Goal:** Central place for all hyperparameters and configuration schemas.

Experiment scripts should construct these configs and pass them into training / analysis APIs.

- **task_config.py**
    - Defines config objects for the *task and dataset*:
        - Rule family selection (which rule types, colors, thresholds, etc.).
        - In-distribution vs. OOD splits (what shifts we create).
        - Dataset sizes, random seeds, train/val/test ratios.
        - Prompt formatting options (which templates, paraphrases, max cards, etc.).
    - Used by tasks.generators and tasks.dataset to create datasets in a reproducible way.
- **training_config.py**
    - Defines configs for:
        - **SFT training**:
            - Model name (e.g. gpt2, gpt2-medium).
            - Training hyperparameters (batch size, lr, scheduler, max steps, etc.).
            - Tokenization / truncation limits.
            - Checkpoint paths and saving cadence.
        - **RL training**:
            - RL algorithm settings (e.g. PPO parameters if using trl).
            - Reward shaping options (terminal reward, format bonus, etc.).
            - Mix of ID vs OOD rules during RL.
    - Scripts in scripts/py will typically:
        - Parse command-line args → instantiate TrainingConfig → call training.sft_trainer.run_sft(config) etc.
- **analysis_config.py**
    - Configs that control mechanistic experiments:
        - Which layers / heads to analyze.
        - Which datasets (ID vs each OOD type) to probe on.
        - Settings for linear probes (regularization, train/val splits).
        - SAE training parameters (for features on some chosen layers).
        - Where to store analysis artifacts (plots, numpy arrays, etc.).

---

## **tasks/**

## **– task specification, data generation, and datasets**

This folder captures **everything specific to the textual “rule-based scoring game” task**.

- **rules.py**
    - Formal definitions of the **rule space**:
        - Primitive rule components (colors, predicates like > k, even/odd, aggregation rules, etc.).
        - Rule classes (e.g. color-add-subtract, thresholded-sum, combinatorial rules).
    - Functions for:
        - Sampling a rule from a named family (e.g. "sum_red_minus_blue").
        - Determining whether a rule belongs to “in-distribution” or some OOD category (by construction).
    - This is the semantic backbone of the task.
- **generators.py**
    - Data generation logic:
        - Sample card sets given a rule:
            - Generate a sequence of cards (color, number) following some distribution.
        - Compute the **ground truth answer** for a (rule, cards) pair.
        - Produce **rich intermediate labels** needed for mechanistic analysis, e.g.:
            - per-color sums,
            - masks for selected cards,
            - final total, BIG/SMALL label, etc.
    - Returns structured Python objects / dictionaries, not tokenized text.
- **formatting.py**
    - Converts structured examples into **text prompts and targets** suitable for GPT-2:
        - Apply templates such as:

```
Rule: ...
Cards:
RED 3
BLUE 5
...

Answer:
```

- 
    - 
        - 
        - Optionally generate paraphrased versions of the rule text.
    - Responsible for:
        - Keeping prompt structure stable and explicit (rule block vs cards block vs answer block).
        - Ensuring compatibility with tokenizer (e.g. adding BOS/EOS, controlling max length).
- **dataset.py**
    - Bridges between generation and HF datasets / dataloaders:
        - Functions to generate in-memory datasets given TaskConfig.
        - Helper to wrap examples as:
            - HF Dataset objects, or
            - torch.utils.data.Dataset implementations.
    - Should expose:
        - High-level factories like: build_sft_datasets(task_config) returning (train, val, test) splits.
        - Explicit separation into ID and multiple OOD test sets.

---

## **models/**

## **– model loading and wrapping**

Handles **Hugging Face causal LMs** and conversion to **TransformerLens** interfaces.

- **lm_loader.py**
    - Utilities to load:
        - A causal LM (AutoModelForCausalLM) from a model name or path.
        - The corresponding tokenizer.
    - Consistent model loading for SFT and RL:
        - Special handling for pad_token_id,
        - Configuring model for training (e.g. gradient checkpointing, device mapping).
    - Entry point for scripts to obtain the base GPT-2 or SFT/RL checkpoints.
- **policy_wrappers.py**
    - Wraps a causal LM as a **policy** for RL:
        - Provide an interface that trl or other RL libs can consume.
        - Handle:
            - logprobs computation,
            - value head attachment if needed (e.g. for PPO),
            - action sampling with constraints (e.g. stop at EOS or Answer: prefix).
    - Abstracts away the low-level “model + tokenizer → RL policy” details from training scripts.
- **conversion.py**
    - Conversion utilities between **Hugging Face models** and **TransformerLens** HookedTransformers:
        - Load a checkpoint into TransformerLens for analysis.
        - Ensure consistent tokenization & embeddings.
    - This is the bridge used by analysis/ to work on the same learned model.

---

## **envs/**

## **– RL environment around the text task**

- **text_game_env.py**
    - Defines a **gym-style environment** (or an equivalent abstraction) for the rule-based text game:
        - Resets by sampling a new (rule, cards) example.
        - Provides an initial prompt.
        - Accepts model-generated text as the “action sequence”.
        - Parses the output to extract the final answer token.
        - Computes reward using ground truth:
            - correct answer → reward = 1,
            - incorrect answer → reward = 0 (+ optional shaping).
    - Designed to plug smoothly into trl’s training loop, or any PPO-style trainer:
        - Generate prompts,
        - Collect model responses,
        - Return rewards and possibly per-token statistics.

---

## **training/**

## **– SFT & RL orchestration (library-style)**

High-level training utilities that use **standard libraries** under the hood, but expose a clean API.

- **sft_trainer.py**
    - Implements **supervised fine-tuning** of GPT-2 on the task using Hugging Face trainers:
        - Given a TrainingConfig and datasets, construct:
            - transformers.TrainingArguments,
            - transformers.Trainer (or equivalent).
        - Handles:
            - Loss computation only on target answer tokens,
            - Logging (loss, accuracy, etc.),
            - Checkpoint saving.
    - Exposes simple entry points such as:
        - run_sft(task_config, training_config) → Path to best checkpoint + evaluation metrics.
- **rl_trainer.py**
    - Implements **RL fine-tuning** from an SFT checkpoint:
        - Uses a standard RL library (e.g. trl):
            - PPO or another suitable algorithm for language models.
        - Constructs a policy from models.policy_wrappers.
        - Interacts with envs.text_game_env to:
            - Generate episodes,
            - Collect rewards,
            - Update model parameters.
    - Should support:
        - Mixing ID and OOD rule families during training (controlled by config).
        - Logging RL metrics (reward curves, success rate on each distribution).
    - Exposes functions like:
        - run_rl(task_config, training_config, init_checkpoint) → Path to RL checkpoint + metrics.
- **evaluation.py**
    - Shared evaluation utilities for both SFT and RL checkpoints:
        - Run the model on:
            - In-distribution test set,
            - Multiple OOD test sets (by type: new colors, new compositions, paraphrases, etc.).
        - Compute:
            - Accuracy,
            - Calibration or sequence-level statistics (optional).
    - Produces standardized evaluation outputs (e.g. JSON) for downstream analysis & plotting.

---

## **analysis/**

## **– mechanistic interpretability stack (TransformerLens-centric)**

This folder encapsulates all **circuit-level analysis**, using **TransformerLens** as the core tool.

- **hooks.py**
    - Common utilities to work with HookedTransformer:
        - Register and manage hooks on specific layers, heads, MLPs.
        - Run the model on datasets while capturing activations.
    - Basic building block for all other analysis modules.
- **probing.py**
    - Implements **linear probes** on internal activations:
        - Train probes to predict:
            - rule ID / rule type,
            - intermediate sums (e.g. sum over RED cards),
            - final answer (BIG/SMALL or numeric).
        - Evaluate probe performance:
            - per-layer, per-position, per-distribution (ID vs OOD).
    - Exposes high-level functions like:
        - run_probes(model, dataset, analysis_config) → probe weights + metrics.
- **patching.py**
    - Implements **activation patching / causal tracing** experiments:
        - Patch activations from ID examples into OOD examples (and vice versa).
        - Measure how patching at particular layers / heads changes the output.
        - Identify heads/neural pathways responsible for generalization vs memorization.
    - Designed to run both:
        - on SFT-only checkpoints,
        - on SFT+RL checkpoints.
- **ablation.py**
    - Implements **head / MLP neuron ablations**:
        - Zero out or modify specific heads/neuron outputs and measure performance drop:
            - on ID,
            - on each OOD distribution.
        - Compute “importance” scores for components, enabling:
            - Jaccard comparisons of important head sets,
            - per-layer importance profiles.
- **sae_features.py**
    - Implements **Sparse AutoEncoder-based feature analysis**:
        - Train SAEs on residual stream activations for selected layers.
        - Analyze features:
            - which rules / card patterns activate them,
            - whether they are ID-specific or OOD-stable.
        - Export feature statistics to files for later inspection.

---

## **utils/**

## **– shared infrastructure**

General-purpose utilities that multiple modules need.

- **paths.py**
    - Helper to manage file system layout:
        - Where to store checkpoints,
        - Where to write logs,
        - Where to save datasets and analysis results.
    - Provides project-root-relative paths so that scripts stay simple and stable.
- **logging.py**
    - Thin wrapper around logging (and/or tqdm) to ensure consistent logging style:
        - Unified log formatting,
        - Optional integration with experiment trackers (e.g. wandb) if desired.
- **serialization.py**
    - Utility functions for:
        - Saving / loading metrics as JSON, CSV, or pickle.
        - Storing probe weights, SAE parameters, ablation results, etc.
    - Keeps IO code out of training/analysis logic.

---

## **Typical Workflows (How Scripts Should Use This Package)**

### **1. Task & Dataset Creation**

In scripts/py/build_datasets.py (for example):

1. Construct a TaskConfig from config.task_config.
2. Call tasks.dataset.build_sft_datasets(task_config).
3. Save datasets to disk using utils.serialization.

### **2. SFT Training**

In scripts/py/run_sft.py:

1. Load datasets.
2. Construct TrainingConfig.
3. Call training.sft_trainer.run_sft(...).
4. Save the best SFT checkpoint and evaluation metrics.

### **3. RL Training**

In scripts/py/run_rl.py:

1. Load TaskConfig, TrainingConfig, and SFT checkpoint path.
2. Call training.rl_trainer.run_rl(...).
3. Save RL checkpoint and evaluation metrics.

### **4. Mechanistic Analysis**

In scripts/py/run_analysis_*.py (e.g. run_probes.py, run_patching.py):

1. Load SFT or RL checkpoint via models.lm_loader and models.conversion.
2. Load relevant datasets (ID + OOD).
3. Construct AnalysisConfig.
4. Call corresponding functions in analysis.probing, analysis.patching, analysis.ablation, analysis.sae_features.
5. Store analysis artifacts in a structured directory via utils.paths and utils.serialization.

---

This README is meant to guide how we **grow** src/sft_rl_circuits/:

- Add new functionality by expanding these modules,
- Keep scripts thin and declarative,
- Ensure everything we do (SFT, RL, mechanistic analysis) remains **reproducible, modular, and task-focused**.