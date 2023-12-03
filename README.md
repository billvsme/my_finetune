my_finetune
============

### Colab
使用免费的T4，只需5分钟即可完成chagtlm3-base 自我认知 qlora sft微调 🥳    
  
[![OpenAll Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/billvsme/my_finetune/blob/master/colab/my_finetune.ipynb)  
*注意：免费 Colab 进行inference时，可能会因为cpu内存不够直接终止，请选择 **高RAM** 配置


### 最低配置

<a href="https://sm.ms/image/qGcS8eCz1f7XhWv" target="_blank"><img src="https://s2.loli.net/2023/12/03/qGcS8eCz1f7XhWv.png" width="40%"></a>


### 本地运行
#### 1. 下载代码
```
git clone https://github.com/billvsme/my_finetune
```


#### 2. 下载chatglm3-base模型, 时间较长请耐心等待
```
git clone --depth=1 https://huggingface.co/THUDM/chatglm3-6b-base
```

#### 3. 安装虚拟环境
```
cd my_finetune
mkdir ~/.venv
python -m venv ~/.venv/finetune
~/.venv/finetune/bin/pip install -r requirements.txt
```
#### 4.替换自我认知self_cognition数据集中的名称

```
sed -i 's/<NAME>/法律AI/g' data/self_cognition.json
sed -i 's/<AUTHOR>/billvsme/g' data/self_cognition.json
```

#### 5.生成deepspeed配置文件
```
echo '''{\
  "train_batch_size": "auto",\
  "train_micro_batch_size_per_gpu": "auto",\
  "gradient_accumulation_steps": "auto",\
  "gradient_clipping": "auto",\
  "zero_allow_untested_optimizer": true,\
  "fp16": {\
    "enabled": "auto",\
    "loss_scale": 0,\
    "initial_scale_power": 16,\
    "loss_scale_window": 1000,\
    "hysteresis": 2,\
    "min_loss_scale": 1\
  },\
  "zero_optimization": {\
    "stage": 2,\
    "allgather_partitions": true,\
    "allgather_bucket_size": 1e8,\
    "reduce_scatter": true,\
    "reduce_bucket_size": 1e8,\
    "overlap_comm": true,\
    "contiguous_gradients": true\
  }\
}''' > ds_config.json
```
#### 6.进行qlora sft微调 🤩
```
~/.venv/finetune/bin/deepspeed --num_gpus 1 --master_port=9901 finetune.py \
    --deepspeed ds_config.json \
    --model_name_or_path /content/chatglm3-6b-base \
    --do_train True\
    --data_dir /content/my_finetune/data/ \
    --data_filename self_cognition.json  \
    --max_source_length 2048 \
    --max_target_length 2048 \
    --max_samples 80 \
    --quantization_bit 4 \
    --learning_rate 2e-4 \
    --lr_scheduler_type linear \
    --max_grad_norm 0.5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_steps 100 \
    --logging_steps 1 \
    --save_steps 1000 \
    --output_dir output/chatglmt3_qlora \
    --overwrite_output_dir True \
    --fp16 True
```

#### 7.查看结果
```
"""查看微调结果😁
"""
~/.venv/finetune/bin/python inference.py \
    --model_name_or_path /content/chatglm3-6b-base \
    --lora_path output/chatglmt3_qlora \
    --data_dir /content/my_finetune/data/ \
    --data_filename self_cognition.json\
    --max_samples 80 \
    --quantization_bit 4

```

