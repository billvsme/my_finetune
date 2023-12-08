# coding: utf-8
import torch
import os

from types import MethodType
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig,\
        DataCollatorForSeq2Seq, BitsAndBytesConfig, PreTrainedTokenizerBase,\
        HfArgumentParser, Seq2SeqTrainingArguments, set_seed

import transformers
from transformers import Seq2SeqTrainer
from arguments import ModelArguments, DataTrainingArguments, FinetuningArguments

try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:
    # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled


def format_message(instruction, input_):
    message = ("Instruction: "
               f"{instruction if instruction else ''}\n")
    message += f"Input: {input_}\n"
    message += "Answer: "

    return message


def get_dataset(tokenizer, data_args):

    dataset = load_dataset(
        path=data_args.data_dir,
        name=None,
        data_files=data_args.data_filename.split(","),
        split='train')

    if data_args.max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), data_args.max_samples)))

    def preprocess_func(examples):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for i in range(len(examples["input"])):
            context = format_message(
                examples['instruction'][i] if 'instruction' in examples else '',
                examples['input'][i])
            target = examples["output"][i]

            a_ids = tokenizer.encode(
                text=context, add_special_tokens=True, truncation=True,
                max_length=data_args.max_source_length)
            b_ids = tokenizer.encode(
                text=target, add_special_tokens=False, truncation=True,
                max_length=data_args.max_target_length)

            context_length = len(a_ids)
            input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
            labels = [-100] * context_length + b_ids + [tokenizer.eos_token_id]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    column_names = list(next(iter(dataset)).keys())

    return dataset.map(
        preprocess_func,
        batched=True,
        num_proc=8,
        remove_columns=column_names
    )


def init_train(model, tokenizer, train_dataset, training_args):

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,
        label_pad_token_id=-100,
        padding=True
    )

    training_args_dict = training_args.to_dict()
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=None
    )

    return trainer


def prepare_train_model(model):
    if getattr(model, "supports_gradient_checkpointing", False):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    output_layer = getattr(model, "lm_head", None)
    if output_layer and isinstance(output_layer, torch.nn.Linear):
        def fp32_forward_pre_hook(module, args):
            return args[0].to(output_layer.weight.dtype)

        def fp32_forward_post_hook(module, args, output):
            return output.to(torch.float32)

        output_layer.register_forward_pre_hook(fp32_forward_pre_hook)
        output_layer.register_forward_hook(fp32_forward_post_hook)

    return model


def init_lora(model, tokenizer, finetuning_args):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetuning_args.lora_rank,
        lora_alpha=finetuning_args.lora_alpha,
        lora_dropout=finetuning_args.lora_dropout,
        target_modules=finetuning_args.lora_target.split(","),
        modules_to_save=None
    )

    return get_peft_model(model, peft_config)


def main():

    # 1. parser input args
    parser = HfArgumentParser((
        ModelArguments, DataTrainingArguments, FinetuningArguments, Seq2SeqTrainingArguments))
    model_args, data_args, finetuning_args, training_args = parser.parse_args_into_dataclasses()

    # show log
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    model_name = model_args.model_name_or_path

    set_seed(training_args.seed)

    # 2. load a pretrained model
    tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            split_special_tokens=False,
            padding_side="right",
            trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # something for qlora
    config_kwargs = {}
    if model_args.quantization_bit is not None:
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

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

        config_kwargs["device_map"] = {"": int(os.environ.get('LOCAL_RANK', '0'))}

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if config.eos_token_id is None:
        config.eos_token_id = tokenizer.eos_token_id
    if config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id

    if getattr(config, "model_type", None) == "chatglm":
        config_kwargs["empty_init"] = False

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()), **config_kwargs)

    # somthing for chatglm
    if getattr(config, "model_type", None) == "chatglm":
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)
        model.lm_head = model.transformer.output_layer
        model._keys_to_ignore_on_save = ["lm_head.weight"]

    model = prepare_train_model(model)
    model = init_lora(model, tokenizer, finetuning_args)
    model = model.train()

    # 3. load dataset
    train_dataset = get_dataset(tokenizer, data_args)

    # print first dataset for check
    if training_args.should_log:
        example = next(iter(train_dataset))
        for key in example:
            if key not in ["input_ids", "labels"]:
                continue
            result = tokenizer.decode(
                list(filter(lambda x: x != -100, example[key])), skip_special_tokens=False)
            print(f"{key}:{result}")

    # 4. initialize the train
    trainer = init_train(model, tokenizer, train_dataset, training_args)

    # 5. train
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()


if __name__ == '__main__':
    main()
