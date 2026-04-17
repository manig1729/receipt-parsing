import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from PIL import Image
from transformers import DonutProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, VisionEncoderDecoderModel

from data_prep import CORD_DATASET_NAME, build_cord_target_fields, parse_ground_truth


if sys.version_info >= (3, 14):
    raise RuntimeError(
        "This script depends on Hugging Face datasets/transformers components that should be "
        "run on Python 3.11, 3.12, or 3.13."
    )


MODEL_NAME = "naver-clova-ix/donut-base"
TARGET_IMAGE_SIZE = (1280, 960)


def get_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def serialize_target(task_start_token: str, target_fields: Dict[str, List[str]]) -> str:
    payload = json.dumps(target_fields, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"{task_start_token}{payload}</s>"


def resize_with_padding(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    target_width, target_height = target_size
    image = image.convert("RGB")

    scale = min(target_width / image.width, target_height / image.height)
    resized_width = max(1, int(round(image.width * scale)))
    resized_height = max(1, int(round(image.height * scale)))

    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_width, target_height), color=(255, 255, 255))
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def build_training_example(example: Dict[str, Any], task_start_token: str) -> Dict[str, Any]:
    ground_truth = parse_ground_truth(example["ground_truth"])
    target_fields = build_cord_target_fields(ground_truth.get("gt_parse", {}))

    return {
        "image": example["image"],
        "target_text": serialize_target(task_start_token, target_fields),
    }


class DonutReceiptCollator:
    def __init__(
        self,
        processor: DonutProcessor,
        target_size: tuple[int, int],
        max_length: int,
    ) -> None:
        self.processor = processor
        self.target_size = target_size
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [resize_with_padding(example["image"], self.target_size) for example in batch]
        pixel_values = self.processor(
            images=images,
            return_tensors="pt",
        ).pixel_values

        tokenized = self.processor.tokenizer(
            [example["target_text"] for example in batch],
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = tokenized.input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def load_cord_for_training(task_start_token: str):
    raw = load_dataset(CORD_DATASET_NAME)
    return raw.map(
        lambda example: build_training_example(example, task_start_token),
        remove_columns=["ground_truth"],
        desc="Formatting CORD targets for Donut",
    )


def build_model_and_processor(model_name: str, task_start_token: str):
    processor = DonutProcessor.from_pretrained(model_name, use_fast=False)
    processor.image_processor.do_align_long_axis = False
    processor.image_processor.do_thumbnail = False
    processor.image_processor.do_resize = False

    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": [task_start_token]})

    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(task_start_token)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = 768
    model.config.early_stopping = False
    model.config.no_repeat_ngram_size = 0
    model.config.length_penalty = 1.0
    model.config.num_beams = 1
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(task_start_token)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.max_length = 768
    model.generation_config.early_stopping = False
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.length_penalty = 1.0
    model.generation_config.num_beams = 1

    return model, processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Donut on filtered CORD targets.")
    parser.add_argument("--output-dir", type=Path, default=Path("./donut-receipt-model"))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--task-start-token", default="<s_receipt_parse>")
    parser.add_argument("--max-target-length", type=int, default=768)
    parser.add_argument("--num-train-epochs", type=int, default=30)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--eval-strategy", default="epoch")
    parser.add_argument("--save-strategy", default="epoch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_name = get_device_name()
    print(f"Using device: {device_name}")
    if device_name == "mps":
        print(
            "MPS mode enabled. Safer defaults are active: micro-batch size 1, gradient accumulation 2, "
            "and gradient checkpointing."
        )

    model, processor = build_model_and_processor(args.model_name, args.task_start_token)
    model.gradient_checkpointing_enable()
    dataset = load_cord_for_training(args.task_start_token)
    data_collator = DonutReceiptCollator(
        processor=processor,
        target_size=TARGET_IMAGE_SIZE,
        max_length=args.max_target_length,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim="adamw_torch",
        max_steps=args.max_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=False,
        remove_unused_columns=False,
        report_to="none",
        use_mps_device=device_name == "mps",
        use_cpu=device_name == "cpu",
        fp16=device_name == "cuda",
        dataloader_num_workers=0,
        dataloader_pin_memory=device_name == "cuda",
        gradient_checkpointing=True,
        torch_empty_cache_steps=10,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        optimizers=(torch.optim.AdamW(model.parameters(), lr=args.learning_rate), None),
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
