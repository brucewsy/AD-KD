# AD-KD: Attribution-Driven Knowledge Distillation for Language Model Compression (ACL 2023)

## Installation
To install the environment, run:

```
pip install -r requirements.txt
```

## Download GLUE Data

Download the GLUE data using [this repository](https://github.com/nyu-mll/GLUE-baselines) or from [GLUE benchmark website](https://gluebenchmark.com/tasks), unpack it to directory ```datas/glue``` and rename the folder ```CoLA``` to ```COLA```.

## Download Pre-trained BERT
Download ```bert_uncased_L-12_H-768_A-12``` (BERT-base) and ```bert_uncased_L-6_H-768_A-12``` for teacher model and student model, respectively, from [this repository](https://github.com/google-research/bert). and use the [API from Huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/convert_bert_original_tf_checkpoint_to_pytorch.py) to transform them to pytorch checkpoint.

## Task-specific Teacher Model Training

We provide training script for each task in ```script/teacher/```, where the **$TEACHER_PATH** is the path of teacher model.




## Task-specific Student Model Distillation
AD-KD can be run on single-GPU or multi-GPU, but make sure to use **DistributedDataParallel** instead of **DataParallel** in Pytorch when using multi-GPU. Here we provide the scripts with single-GPU in ```script/student/```, where the **$TEACHER_PATH** and **$STUDENT_PATH** are the path of teacher model and student model, respectively.



## Student Checkpoints
The distilled student model for each task reported in the paper can be downloaded as follows:

```python
from transformers import BertForSequenceClassification
task_name = 'cola' # task name with lower case
model = BertForSequenceClassification.from_pretrained("Brucewsy/AD-KD_bert_uncased_L-6_H-768_A-12_" + task_name)
```
