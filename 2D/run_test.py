import subprocess


def run():
    cmd = 'python -m torch.distributed.launch test.py --local_rank {} --config {}'\
    .format(-1, 'configs/demo.yaml')
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()


if __name__ == '__main__':
    run()



