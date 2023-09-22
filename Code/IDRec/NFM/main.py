import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':



    device = '3'  

    import random
    master_port = random.randint(1002,9999)

    nproc_per_node = len(device.split(','))
    
    run_yaml = f"CUDA_VISIBLE_DEVICES='{device}'  python  -m torch.distributed.run --nproc_per_node {nproc_per_node} \
--master_port {master_port} run.py --config_file YAML/nfm.yaml" 
    os.system(run_yaml)

