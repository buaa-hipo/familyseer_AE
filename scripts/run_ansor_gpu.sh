python tune_network_gpu.py --model mobilenet0.5 --tune --mode ansor > ./[mobilenet0.5]\_[NVIDIA_V100]_ansor_B1.output 2>&1
python tune_network_gpu.py --model roberta_large --tune --mode ansor > ./[roberta_large]\_[NVIDIA_V100]_ansor_B1.output 2>&1
python tune_network_gpu.py --model resnet50_v1 --tune --mode ansor > ./[resnet50_v1]\_[NVIDIA_V100]_ansor_B1.output 2>&1
python tune_network_gpu.py --model bert_large --tune --mode ansor > ./[bert_large]\_[NVIDIA_V100]_ansor_B1.output 2>&1
python tune_network_gpu.py --model resnet152_v2 --tune --mode ansor > ./[resnet152_v2]\_[NVIDIA_V100]_ansor_B1.output 2>&1
python tune_network_gpu.py --model mobilenetv2_0.5 --tune --mode ansor > ./[mobilenetv2_0.5]\_[NVIDIA_V100]_ansor_B1.output 2>&1
python tune_network_gpu.py --model vit_huge --tune --mode ansor > ./[vit_huge]\_[NVIDIA_V100]_ansor_B1.output 2>&1
python tune_network_gpu.py --model gpt2 --tune --mode ansor > ./[gpt2]\_[NVIDIA_V100]_ansor_B1.output 2>&1