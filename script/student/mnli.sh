TEACHER_PATH=./experiments/exp_distill_mnli/teacher/checkpoints/best_checkpoint
STUDENT_PATH=models/uncased_L-6_H-768_A-12

python main_glue_distill.py --distill_loss kd_anneal+saliency \
							--do_lower_case \
							--do_train \
							--task_name mnli \
							--teacher_path $TEACHER_PATH \
							--student_path $STUDENT_PATH \
							--per_gpu_batch_size 64 \
							--num_train_epochs 6 \
							--learning_rate 4e-5 \
							--alpha 0.8 \
							--temperature 3.0 \ 
							--saliency_w 10