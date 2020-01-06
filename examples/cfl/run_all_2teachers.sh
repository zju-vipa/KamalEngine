
#mkdir -p data

# Train Teachers
python train_teachers.py --dataset stanforddogs --gpu_id 2 --lr 1e-4
python train_teachers.py --dataset cub200 --gpu_id 2 --lr 1e-4

# Distillation and Amalgamation
#python amal_dogs_cub200.py --gpu_id 1