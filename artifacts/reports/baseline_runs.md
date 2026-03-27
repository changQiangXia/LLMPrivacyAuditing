# baseline 逐运行明细

| label | aggregate_label | category | seed | train_loss | extraction_success_rate | avg_exposure | mia_auc_loss | mia_auc_neighbourhood | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_model_seed0 | base_model | baseline | 0 |  | 0.0 | 1.2594712206491605 | 0.49419985848905834 | 0.49907420149879267 | 基础模型，无 LoRA |
| lora_standard_seed0 | lora_standard | baseline | 0 | 2.6381317194360885 | 0.0 | 1.6609825235608724 | 0.512527471610932 | 0.5155175505540195 | 插入 canary 的标准 LoRA |
| lora_standard_seed1 | lora_standard | baseline | 1 | 2.63597970113267 | 0.0 | 1.6652561637976653 | 0.5119408545425992 | 0.5150685521818119 | 插入 canary 的标准 LoRA |
| lora_standard_seed2 | lora_standard | baseline | 2 | 2.6423486862739507 | 0.0 | 1.6293069084963434 | 0.5123926330904855 | 0.5158212847469827 | 插入 canary 的标准 LoRA |
| lora_no_canary_seed0 | lora_no_canary | baseline | 0 | 2.2204467264811196 | 0.0 | 1.2695170316113356 | 0.5096548550902931 | 0.5165427403357616 | 不插入 canary 的 LoRA |
| lora_dedup_seed0 | lora_dedup | baseline | 0 | 2.638227421001796 | 0.0 | 1.6642854344629334 | 0.5125768197292385 | 0.5155064298513027 | 开启 dedup 的 LoRA，当前训练集无精确重复，等价于标准训练集 |
