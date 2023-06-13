TEACHER_PATH=./experiments/exp_distill_rte/teacher/checkpoints/best_checkpoint
STUDENT_PATH=models/uncased_L-6_H-768_A-12

python main_glue_distill.py --distill_loss kd+saliency \
							--do_lower_case \
							--do_train \
							--task_name rte \
							--teacher_path $TEACHER_PATH \
							--student_path $STUDENT_PATH \
							--per_gpu_batch_size 16 \
							--num_train_epochs 12 \
							--learning_rate 2e-5 \
							--alpha 0.9 \
							--temperature 2.0 \ 
							--saliency_w 10 \
							--topk 700