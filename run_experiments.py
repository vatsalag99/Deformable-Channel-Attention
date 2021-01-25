import subprocess

template = 'python cifar_main.py --gpu {} --seed 42 --epoch {} -a dca_cifar100_resnet50_local_deform_shuffle -b 128 --lr 0.1 --dropout {} --channels_per_group {} ../garbage_dir --action seed=42_new_scheduler_epochs_{}_dropout_{}_cpg_{}'

args = [
        [0] + [200, 0.15, 32] * 2,
        [0] + [250, 0.15, 32] * 2,
        [1] + [200, 0.25, 16] * 2,
        [1] + [250, 0.25, 16] * 2,
        [2] + [200, 0.15, 16] * 2,
        [2] + [250, 0.15, 16] * 2,
       ]


# template = 'python cifar_main.py --gpu {} --seed 42 --epoch {} -a dca_cifar100_resnet50_local_deform_shuffle -b 128 --lr 0.1 --dropout {} --n_groups {} ../garbage_dir --action seed=42_epochs_{}_dropout_{}_n_groups_{}'

# args = [
#         [0] + [200, 0.25, 16] * 2,
#         [0] + [200, 0.25, 64] * 2, 
#         [0] + [200, 0.25, 32] * 2, # number of groups constant 
#        ]


# Run commands in parallel
processes = []

for arg in args:
    command = template.format(*[str(a) for a in arg])
    process = subprocess.Popen(command, shell=True)
    processes.append(process)

# Collect statuses
output = [p.wait() for p in processes]