# baseline 汇总

| aggregate_label | category | n | seeds | train_loss(mean±std) | extraction_success_rate(mean±std) | avg_exposure(mean±std) | mia_auc_loss(mean±std) | mia_auc_neighbourhood(mean±std) | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_model | baseline | 1 | 0 |  | 0.000000 ± 0.000000 | 1.259471 ± 0.000000 | 0.494200 ± 0.000000 | 0.499074 ± 0.000000 | 基础模型，无 LoRA |
| lora_dedup | baseline | 1 | 0 | 2.638227 ± 0.000000 | 0.000000 ± 0.000000 | 1.664285 ± 0.000000 | 0.512577 ± 0.000000 | 0.515506 ± 0.000000 | 开启 dedup 的 LoRA，当前训练集无精确重复，等价于标准训练集 |
| lora_no_canary | baseline | 1 | 0 | 2.220447 ± 0.000000 | 0.000000 ± 0.000000 | 1.269517 ± 0.000000 | 0.509655 ± 0.000000 | 0.516543 ± 0.000000 | 不插入 canary 的 LoRA |
| lora_standard | baseline | 3 | 0,1,2 | 2.638820 ± 0.003240 | 0.000000 ± 0.000000 | 1.651849 ± 0.019638 | 0.512287 ± 0.000307 | 0.515469 ± 0.000379 | 插入 canary 的标准 LoRA |
