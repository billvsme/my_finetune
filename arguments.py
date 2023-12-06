# coding: utf-8
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: Optional[str] = field(
        default=None, metadata={"help": "train file dir."}
    )
    data_filename: Optional[str] = field(
        default=None, metadata={"help": "train file(alpaca format)."}
    )

    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "limit dataset"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. \
                        Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )


@dataclass
class FinetuningArguments:
    lora_rank: Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "balancing between complexity and model flexibility. A higher rank allows more "
                "complex adaptations but increases the number of parameters and computational cost."
            )
        },
    )

    lora_alpha: Optional[float] = field(
        default=16,
        metadata={
            "help": (
                "A higher value results in more significant adjustments, potentially \
                        improving adaptation to new tasks or data, "
                "but might also risk overfitting. A lower value makes smaller adjustments, \
                        possibly maintaining better generalization."
            )
        }, )

    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": (
                "during training to prevent the model from overly relying on specific \
                        patterns in the training data. "
                "Higher dropout rates can improve model generalization but may reduce \
                        learning efficiency."
            )
        },
    )

    lora_target: Optional[str] = field(
        default="query_key_value",
        metadata={"help": "Name(s) of target modules to apply LoRA"}
    )


@dataclass
class InferenceArguments:
    lora_path: str = field(
        metadata={"help": "Path to lora"}
    )

    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "max_length for generate"
        },
    )
