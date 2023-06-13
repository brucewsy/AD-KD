TEACHER_PATH=models/bert-base-uncased # model path
python main_glue.py --do_lower_case \
					--do_train \
					--task_name rte \
					--model_path $TEACHER_PATH \
					--per_gpu_batch_size 16 \
					--num_train_epochs 4 \
					--learning_rate 5e-5