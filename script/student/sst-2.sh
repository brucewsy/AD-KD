TEACHER_PATH=./experiments/exp_distill_sst-2/teacher/checkpoints/best_checkpoint
STUDENT_PATH=models/uncased_L-6_H-768_A-12

python main_glue_distill.py --distill_loss kd+saliency \
							--do_lower_case \
							--do_train \
							--task_name sst-2 \
							--teacher_path $TEACHER_PATH \
							--student_path $STUDENT_PATH \
							--per_gpu_batch_size 32 \
							--num_train_epochs 6 \
							--learning_rate 5e-5 \
							--alpha 0.8 \
							--temperature 2.0 \ 
							--saliency_w 1.0