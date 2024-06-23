# KnowTuning

Implementation for the paper:

**KnowTuning: Knowledge-aware Fine-tuning for  Large Language Models**

## Requirements

pip install -r requirements.txt

Please note download [LLaMA-Factory-files](https://pypi.org/project/llmtuner/0.5.3/) to prepare enviroments.

## Datasets

**General Domain QA dataset Dolly:**

Vanilla fine-tuning set:

```
data/dolly/dolly_train.json
```

Fine-grained knowledge augmentation set:

```bash
data/dolly/dolly_KA.json
```

Coarse-grained knowledge comparison set:

```
data/dolly/dolly_KC.json
```

Test dataset:

```
data/lima/lima_test.json
```

**Medical Domain QA datset MedQuAD:**

Vanilla fine-tuning set:

```
data/medquad/medquad_train.json
```

Fine-grained knowledge augmentation set:

```
data/medquad/medquad_KA.json
```

Coarse-grained knowledge comparison set:

```
data/medquad/medquad_KC.json
```


Test dataset:

```
data/medquad/medquad_test.json
```

## Models

Download llama2-7b-base and llama2-13b-base in the model folder

```
model/llama_7b_base
```

## Training

Based on open-source training framework [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), following below instructions for training.

### Vanilla fine-tuning

```bash
sh dolly/sft_dolly.sh
sh medquad/sft_medquad.sh
```

### Fine-grained Knowledge Augmentation

```bash
sh dolly/ka_dolly.sh
sh medquad/ka_medquad.sh
```

### Coarse-grained Knowledge Comparison

```bash
sh dolly/kc_dolly.sh
sh medquad/kc_medquad.sh
```

### Test

```bash
sh dolly/test_dolly.sh
sh medquad/test_medquad.sh
```
