TEACHER_PATH=models/bert-base-uncased # model path
python main_glue.py --do_lower_case \
					--do_train \
					--task_name mnli \
					--model_path $TEACHER_PATH \
					--per_gpu_batch_size 32 \
					--num_train_epochs 5 \
					--learning_rate 2e-5