python tune_network_gpu.py --model mobilenet0.5 --tune --mode family --gpu_num $1 > ./[mobilenet0.5]\_[NVIDIA_V100]_ansor_B1_D$1.output 2>&1
python tune_network_gpu.py --model roberta_large --tune --mode family --gpu_num $1 > ./[roberta_large]\_[NVIDIA_V100]_family_B1_D$1.output 2>&1
python tune_network_gpu.py --model resnet50_v1 --tune --mode family --gpu_num $1 > ./[resnet50_v1]\_[NVIDIA_V100]_family_B1_D$1.output 2>&1
python tune_network_gpu.py --model bert_large --tune --mode family --gpu_num $1 > ./[bert_large]\_[NVIDIA_V100]_family_B1_D$1.output 2>&1
python tune_network_gpu.py --model mobilenetv2_0.5 --tune --mode family --gpu_num $1 > ./[mobilenetv2_0.5]\_[NVIDIA_V100]_family_B1_D$1.output 2>&1
python tune_network_gpu.py --model resnet152_v2 --tune --mode family --gpu_num $1 > ./[resnet152_v2]\_[NVIDIA_V100]_family_B1_D$1.output 2>&1
python tune_network_gpu.py --model gpt2 --tune --mode family --gpu_num $1 > ./[gpt2]\_[NVIDIA_V100]_family_B1_D$1.output 2>&1
python tune_network_gpu.py --model vit_huge --tune --mode family --gpu_num $1 > ./[vit_huge]\_[NVIDIA_V100]_family_B1_D$1.output 2>&1