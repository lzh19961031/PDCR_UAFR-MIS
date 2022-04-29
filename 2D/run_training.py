import os
import glob
import subprocess
import numpy as np
import time


def run(file_path, config_path, save_path, port):
    cmd = 'srun -p MIA --nodes=2 -n8 --gres=gpu:8 --ntasks=16 --tasks-per-node=8 --cpus-per-task=5 --job-name={} python -u {} --config {} --save_path {} --port {}'\
    .format('seg', file_path, config_path, save_path, port)
    p = subprocess.Popen(cmd, shell=True)


def get_file(path='./runs/', ext='*.yaml', toogle=None):
    all_files = [file
                    for path, subdir, files in os.walk(path)
                    for file in glob.glob(os.path.join(path, ext))]
    
    if toogle:
        all_files = [path for path in all_files if toogle in path]

    return all_files


if __name__ == '__main__':

    run_toggle = './' 
    run_file = 'train.py'
    config_dir = './configs'
    log_dir = './runs/'

    # get all configs
    all_configs = get_file(path=config_dir, ext='*.yaml', toogle=run_toggle)
    print(all_configs, '\n')

    # ignore tem file
    all_configs = [file for file in all_configs if 'tem' not in file]

    # get existing configs
    existing_configs = get_file(path=log_dir, ext='*.yaml')
    existing_patterns = [os.path.dirname(os.path.join(*(f.split(os.path.sep)[2:]))) for f in existing_configs]
    print(existing_patterns, '\n')

    # ignore existing configs
    target_item = []
    for yaml_path in sorted(all_configs):
        src_pattern = os.path.splitext(os.path.join(*(yaml_path.split(os.path.sep)[2:])))[0]
        if src_pattern not in existing_patterns:
            save_path = log_dir + '/' + src_pattern
            target_item.append((yaml_path, save_path))
    
    print('running exps: ', target_item)
    all_ports = [str(i) for i in range(23333, 25000)]
    used_port = np.random.choice(all_ports, size=len(target_item), replace=False)
    for idx, item in enumerate(target_item):
        port = used_port[idx]
        config_path, save_path = item
        run(run_file, config_path, save_path, port)
        time.sleep(2)

