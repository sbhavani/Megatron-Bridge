# Mixtral Finetune Recipes

This branch is a placeholder for Mixtral finetune recipes (Part 3 of 3 PRs).

## TODO

- [ ] Add `mixtral_8x7b_finetune_config`
- [ ] Add `mixtral_8x22b_finetune_config`
- [ ] Add finetune recipe tests
- [ ] Add documentation for finetune workflow

## Dependencies

This branch depends on:
- PR #1317: Model provider + bridge (core conversion)
- Pretrain recipes PR: Pretrain configurations

## Usage

Once implemented, finetune recipes will enable:
- Supervised fine-tuning on custom datasets
- LoRA/QLoRA fine-tuning support
- Instruction tuning workflows
