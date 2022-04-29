import subprocess


def run():
    cmd = 'python -m torch.distributed.launch ttest_parameter.py --local_rank {} --config {}'\
    .format(-1, 'sota_checkpoints/MC/full_split28.yaml')
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()


if __name__ == '__main__':
    run()



