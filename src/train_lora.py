from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer import logger as trainer_logger

from .config_utils import load_section_defaults
from .experiment import build_experiment_id, create_run_dir, create_stage_run_dir, save_run_metadata
from .utils import (
    load_causal_lm,
    local_hub_offline,
    resolve_model_path,
    save_json,
    set_seed,
    setup_logger,
)
from .validation import ensure_empty_or_missing_dir, require_file, validate_text_jsonl


class StopOnNaNCallback(TrainerCallback):
    """检测到 loss 为 NaN/Inf 时提前结束训练。"""

    def __init__(self, logger):
        self.logger = logger

    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history:
            log = state.log_history[-1]
            loss = log.get("loss")
            if loss is not None and (not torch.isfinite(torch.tensor(loss))):
                self.logger.error("检测到无效 loss=%s，训练终止", loss)
                control.should_training_stop = True
        return control


class LocalAwareTrainer(Trainer):
    """在本地模型目录场景下，为 PEFT checkpoint 保存显式关闭远端配置探测。"""

    def __init__(self, *args, base_model_name_or_path: str, save_embedding_layers: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.save_embedding_layers = save_embedding_layers

    def _save(self, output_dir: str | None = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        model_to_save = self.accelerator.unwrap_model(self.model)

        is_peft_model = isinstance(self.model, PeftModel) or isinstance(model_to_save, PeftModel)
        if not is_peft_model:
            return super()._save(output_dir=output_dir, state_dict=state_dict)

        os.makedirs(output_dir, exist_ok=True)
        trainer_logger.info("Saving model checkpoint to %s", output_dir)

        target_model = (
            self.model
            if isinstance(self.model, (PreTrainedModel, PeftModel))
            else model_to_save
        )
        with local_hub_offline(self.base_model_name_or_path):
            target_model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
                save_embedding_layers=self.save_embedding_layers,
            )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    known, _ = pre.parse_known_args()
    defaults = load_section_defaults(known.config, "train")

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--model_name_or_path", default=defaults.get("model_name_or_path", ""))
    ap.add_argument("--train_jsonl", default=defaults.get("train_jsonl", ""))
    ap.add_argument("--output_dir", default=defaults.get("output_dir", ""))
    ap.add_argument("--output_root", default=defaults.get("output_root", "outputs/experiments"))
    ap.add_argument("--exp_id", default=defaults.get("exp_id", ""))
    ap.add_argument("--epochs", type=int, default=defaults.get("epochs", 1))
    ap.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 2))
    ap.add_argument("--grad_accum", type=int, default=defaults.get("grad_accum", 8))
    ap.add_argument("--lr", type=float, default=defaults.get("lr", 2e-4))
    ap.add_argument("--max_len", type=int, default=defaults.get("max_len", 512))
    ap.add_argument("--seed", type=int, default=defaults.get("seed", 0))
    ap.add_argument("--use_4bit", action="store_true", default=defaults.get("use_4bit", False))
    ap.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=defaults.get("trust_remote_code", False),
    )
    ap.add_argument("--logging_steps", type=int, default=defaults.get("logging_steps", 20))
    ap.add_argument("--save_steps", type=int, default=defaults.get("save_steps", 200))
    ap.add_argument("--save_total_limit", type=int, default=defaults.get("save_total_limit", 2))
    ap.add_argument("--warmup_ratio", type=float, default=defaults.get("warmup_ratio", 0.03))
    ap.add_argument("--scheduler", default=defaults.get("scheduler", "cosine"))
    ap.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=defaults.get("gradient_checkpointing", False),
    )
    ap.add_argument("--lora_r", type=int, default=defaults.get("lora_r", 16))
    ap.add_argument("--lora_alpha", type=int, default=defaults.get("lora_alpha", 32))
    ap.add_argument("--lora_dropout", type=float, default=defaults.get("lora_dropout", 0.05))
    ap.add_argument(
        "--target_modules",
        nargs="+",
        default=defaults.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    args = ap.parse_args()
    if not args.model_name_or_path:
        ap.error("缺少 --model_name_or_path，且配置文件中未提供")
    if not args.train_jsonl:
        ap.error("缺少 --train_jsonl，且配置文件中未提供")
    return args


def determine_run_dir(args) -> Path:
    if args.output_dir:
        ensure_empty_or_missing_dir(args.output_dir)
        return create_run_dir(args.output_dir, allow_existing=True)

    exp_id = args.exp_id or build_experiment_id(
        stage="train",
        model_name_or_path=args.model_name_or_path,
        data_path=args.train_jsonl,
        seed=args.seed,
        extra_tags=[f"ep{args.epochs}", f"lr{args.lr:g}"],
    )
    return create_stage_run_dir(args.output_root, "train", exp_id)


def build_effective_config(args, run_dir: Path) -> Dict:
    return {
        "config": args.config,
        "model_name_or_path": args.model_name_or_path,
        "train_jsonl": os.path.abspath(args.train_jsonl),
        "output_dir": str(run_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_len": args.max_len,
        "seed": args.seed,
        "use_4bit": bool(args.use_4bit),
        "trust_remote_code": bool(args.trust_remote_code),
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "warmup_ratio": args.warmup_ratio,
        "scheduler": args.scheduler,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": list(args.target_modules),
        },
    }


def main():
    args = parse_args()
    args.model_name_or_path = resolve_model_path(args.model_name_or_path)
    require_file(args.train_jsonl, "训练集")
    train_report = validate_text_jsonl(args.train_jsonl)
    if train_report["valid_rows"] == 0:
        raise SystemExit("[ERROR] 训练集不存在有效 text 样本")

    run_dir = determine_run_dir(args)
    logger = setup_logger(str(run_dir / "train.log"))
    effective_config = build_effective_config(args, run_dir)
    save_run_metadata(
        run_dir=run_dir,
        stage="train",
        effective_config=effective_config,
        extra_metadata={
            "train_report": train_report,
            "model_is_local_dir": os.path.isdir(args.model_name_or_path),
        },
    )
    logger.info("训练目录: %s", run_dir)
    logger.info("训练集有效样本: %s", train_report["valid_rows"])

    set_seed(args.seed)

    ds = load_dataset("json", data_files={"train": args.train_jsonl})["train"]

    with local_hub_offline(args.model_name_or_path):
        tok = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=True,
            trust_remote_code=args.trust_remote_code,
            local_files_only=os.path.isdir(args.model_name_or_path),
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    def tokenize(batch: Dict) -> Dict:
        return tok(
            batch["text"],
            truncation=True,
            max_length=args.max_len,
            padding=False,
        )

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)
    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": os.path.isdir(args.model_name_or_path),
        "torch_dtype": torch.bfloat16
        if use_bf16
        else (torch.float16 if use_fp16 else torch.float32),
    }
    if args.use_4bit and torch.cuda.is_available():
        model_kwargs.update({"load_in_4bit": True, "device_map": "auto"})
    with local_hub_offline(args.model_name_or_path):
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if args.use_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(args.target_modules),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        run_name=run_dir.name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        optim="adamw_torch",
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        logging_dir=str(run_dir / "tb"),
        logging_strategy="steps",
        save_strategy="steps",
        overwrite_output_dir=False,
    )

    save_embedding_layers = len(tok) != model.config.vocab_size
    if save_embedding_layers:
        logger.info("检测到词表大小变化，保存 embedding 层")

    trainer = LocalAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tok,
        callbacks=[StopOnNaNCallback(logger)],
        base_model_name_or_path=args.model_name_or_path,
        save_embedding_layers=save_embedding_layers,
    )
    train_result = trainer.train()

    invalid_losses = []
    for item in trainer.state.log_history:
        loss = item.get("loss")
        if loss is not None and not math.isfinite(loss):
            invalid_losses.append(loss)
    if invalid_losses:
        raise SystemExit(f"[ERROR] 训练过程中出现无效 loss: {invalid_losses}")

    with local_hub_offline(args.model_name_or_path):
        trainer.model.save_pretrained(str(run_dir), save_embedding_layers=save_embedding_layers)
    tok.save_pretrained(str(run_dir))

    metrics = train_result.metrics
    metrics["train_dataset_size"] = train_report["valid_rows"]
    metrics["effective_batch_size"] = args.batch_size * args.grad_accum
    save_json(metrics, str(run_dir / "metrics.json"))
    save_json(trainer.state.log_history, str(run_dir / "train_history.json"))

    logger.info("开始执行 checkpoint 可加载性验证")
    load_causal_lm(
        model_name_or_path=args.model_name_or_path,
        lora_dir=str(run_dir),
        device="cpu",
        load_in_4bit=False,
        trust_remote_code=args.trust_remote_code,
    )
    logger.info("checkpoint 可加载性验证通过")
    logger.info("训练完成 -> %s", run_dir)


if __name__ == "__main__":
    main()
