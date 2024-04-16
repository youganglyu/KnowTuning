# KnowTuning

Implementation for the paper:

**KnowTuning: Knowledge-aware Fine-tuning for  Large Language Models**

## Requirements

pip install -r requirements.txt

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
CUDA_VISIBLE_DEVICES=0 nohup python src/train_bash.py --stage sft --model_name_or_path model/llama_7b_base --do_train --dataset dolly_train --template alpaca --finetuning_type lora --lora_target q_proj,v_proj --load_best_model_at_end --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --overwrite_cache --lr_scheduler_type cosine --eval_steps 0.1 --evaluation_strategy steps --save_steps 0.1 --val_size 100 --learning_rate 5e-5 --num_train_epochs 3.0 --bf16 --plot_loss --output_dir checkpoint/dolly/sft > train.out 2>&1 &
```

### Fine-grained Knowledge Augmentation

```bash
CUDA_VISIBLE_DEVICES=0 nohup python src/train_bash.py --stage sft --model_name_or_path model/llama_7b_base --do_train --dataset dolly_KA --template alpaca --finetuning_type lora --lora_target q_proj,v_proj --load_best_model_at_end --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --overwrite_cache --lr_scheduler_type cosine --eval_steps 0.1 --evaluation_strategy steps --save_steps 0.1 --val_size 100 --learning_rate 5e-5 --num_train_epochs 3.0 --bf16 --plot_loss --output_dir checkpoint/dolly/ka > train.out 2>&1 &
```

### Coarse-grained Knowledge Comparison

```bash
CUDA_VISIBLE_DEVICES=0 nohup python src/train_bash.py --stage dpo --model_name_or_path model/llama_7b_base --do_train --dataset dolly_KC --template alpaca --finetuning_type lora --lora_target q_proj,v_proj --resume_lora_training 1 --overwrite_output_dir --checkpoint_dir checkpoint/dolly/ka --output_dir checkpoint/dolly/kc --per_device_train_batch_size 8 --gradient_accumulation_steps 1 --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 1e-5 --num_train_epochs 1.0 --plot_loss --bf16 > train.out 2>&1 &
```

### Test

```bash
CUDA_VISIBLE_DEVICES=0 nohup python src/train_bash.py --stage sft --model_name_or_path model/llama_7b_base --do_predict --dataset dolly_test --template alpaca --finetuning_type lora --overwrite_cache --per_device_eval_batch_size 1 --max_samples 200 --predict_with_generate --checkpoint_dir checkpoint/dolly/sft --output_dir evaluation/dolly/test > test.out 2>&1 &
```
