python tune_network_gpu.py --model mobilenet0.5 --tune --mode family > ./[mobilenet0.5]\_[NVIDIA_V100]_ansor_B1.output 2>&1
python tune_network_gpu.py --model roberta_large --tune --mode family > ./[roberta_large]\_[NVIDIA_V100]_family_B1.output 2>&1
python tune_network_gpu.py --model resnet50_v1 --tune --mode family > ./[resnet50_v1]\_[NVIDIA_V100]_family_B1.output 2>&1
python tune_network_gpu.py --model bert_large --tune --mode family > ./[bert_large]\_[NVIDIA_V100]_family_B1.output 2>&1
python tune_network_gpu.py --model mobilenetv2_0.5 --tune --mode family > ./[mobilenetv2_0.5]\_[NVIDIA_V100]_family_B1.output 2>&1
python tune_network_gpu.py --model resnet152_v2 --tune --mode family > ./[resnet152_v2]\_[NVIDIA_V100]_family_B1.output 2>&1
python tune_network_gpu.py --model gpt2 --tune --mode family > ./[gpt2]\_[NVIDIA_V100]_family_B1.output 2>&1
python tune_network_gpu.py --model vit_huge --tune --mode family > ./[vit_huge]\_[NVIDIA_V100]_family_B1.output 2>&1