# coding: utf-8
from types import MethodType

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig,\
        HfArgumentParser, PreTrainedTokenizerBase, BitsAndBytesConfig
from datasets import load_dataset

from finetune import format_message
from arguments import ModelArguments, DataTrainingArguments, InferenceArguments


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, InferenceArguments))
    model_args, data_args, inference_args = parser.parse_args_into_dataclasses()

    model_name = model_args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        split_special_tokens=False, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config_kwargs = {
        "low_cpu_mem_usage": True
    }

    if model_args.quantization_bit is not None:
        if model_args.quantization_bit == 8:
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        if model_args.quantization_bit == 4:
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if config.eos_token_id is None:
        config.eos_token_id = tokenizer.eos_token_id
    if config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        inference_args.lora_path, trust_remote_code=True, device_map="auto",
        config=config, **config_kwargs)

    if getattr(config, "model_type", None) == "chatglm":
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    dataset = load_dataset(
        path=data_args.data_dir,
        name=None,
        data_files=data_args.data_filename.split(","),
        split='train'
    )

    for index, example in enumerate(dataset):
        prompt = format_message(example.get("instruction", ''), example.get("input"))
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        response = model.generate(
            input_ids=inputs["input_ids"],
            max_length=inputs["input_ids"].shape[-1] + inference_args.max_length)
        response = response[0, inputs["input_ids"].shape[-1]:]
        print("="*100)
        print("question:", index)
        print(prompt)
        print("Origin  -->", example["output"])
        print("Predict -->", tokenizer.decode(response, skip_special_tokens=True))


if __name__ == '__main__':
    main()
