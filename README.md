# KnowTuning

Implementation for the paper:

**KnowTuning: Knowledge-aware Fine-tuning for  Large Language Models**

## Requirements

pip install -r requirements.txt

## Datasets

**General Domain QA dataset LIMA:**

Vanilla fine-tuning set:

```
data/lima/lima_train.json
```

Explicit knowledge generation set:

```bash
data/lima/lima_KG.json
```

Implicit knowledge comparison set:

```
data/lima/lima_KC.json
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

Explicit knowledge generation set:

```
data/medquad/medquad_KG.json
```

Implicit knowledge comparison set:

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
CUDA_VISIBLE_DEVICES=0 nohup python src/train_bash.py --stage sft --model_name_or_path model/llama_7b_base --do_train --dataset lima_train --template alpaca --finetuning_type lora --lora_target q_proj,v_proj --load_best_model_at_end --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --overwrite_cache --lr_scheduler_type cosine --eval_steps 0.1 --evaluation_strategy steps --save_steps 0.1 --val_size 100 --learning_rate 5e-5 --num_train_epochs 3.0 --bf16 --plot_loss --output_dir checkpoint/lima/sft > train.out 2>&1 &
```

### Explicit Knowledge Generation

```bash
CUDA_VISIBLE_DEVICES=0 nohup python src/train_bash.py --stage sft --model_name_or_path model/llama_7b_base --do_train --dataset lima_KG --template alpaca --finetuning_type lora --lora_target q_proj,v_proj --load_best_model_at_end --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --overwrite_cache --lr_scheduler_type cosine --eval_steps 0.1 --evaluation_strategy steps --save_steps 0.1 --val_size 100 --learning_rate 5e-5 --num_train_epochs 3.0 --bf16 --plot_loss --output_dir checkpoint/lima/kg > train.out 2>&1 &
```

### Implicit Knowledge Comparison

```bash
CUDA_VISIBLE_DEVICES=0 nohup python src/train_bash.py --stage dpo --model_name_or_path model/llama_7b_base --do_train --dataset lima_KC --template alpaca --finetuning_type lora --lora_target q_proj,v_proj --resume_lora_training 1 --overwrite_output_dir --checkpoint_dir checkpoint/lima/kg --output_dir checkpoint/lima/kc --per_device_train_batch_size 8 --gradient_accumulation_steps 1 --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 1e-5 --num_train_epochs 1.0 --plot_loss --bf16 > train.out 2>&1 &
```

### Test

```bash
CUDA_VISIBLE_DEVICES=0 nohup python src/train_bash.py --stage sft --model_name_or_path model/llama_7b_base --do_predict --dataset lima_test --template alpaca --finetuning_type lora --overwrite_cache --per_device_eval_batch_size 1 --max_samples 300 --predict_with_generate --checkpoint_dir checkpoint/lima/sft --output_dir evaluation/sft/lima > test.out 2>&1 &
```

## Citation

If you find our work useful, please cite our paper as follows:

```
@article{lyu2024knowtuning,
  title={KnowTuning: Knowledge-aware Fine-tuning for Large Language Models},
  author={Lyu, Yougang and Yan, Lingyong and Wang, Shuaiqiang and Shi, Haibo and Yin, Dawei and Ren, Pengjie and Chen, Zhumin and de Rijke, Maarten and Ren, Zhaochun},
  journal={arXiv preprint arXiv:2402.11176},
  year={2024}
}
```
