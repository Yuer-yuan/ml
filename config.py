import os

datasets_dir = './datasets'
ckpts_dir = './ckpts'
logs_dir = './logs'
csvs_dir = os.path.join(logs_dir, 'csv')
scalars_dir = os.path.join(logs_dir, 'scalar')

train_dataset = 'T91'
val_dataset = 'Set5'
test_dataset = 'Set14'

scale = 3
in_sz = 33
out_sz = 21
stride = 14

train_file = os.path.join(datasets_dir, f'{train_dataset}_train_x{scale}.h5')
val_file = os.path.join(datasets_dir, f'{val_dataset}_val_x{scale}.h5')
test_file = os.path.join(datasets_dir, f'{test_dataset}_val_x{scale}.h5')

ckpt_file = os.path.join(
    ckpts_dir, f"{train_dataset}_x{scale}.pth")
csv_file = os.path.join(
    csvs_dir, f"val_{train_dataset}_{val_dataset}_x{scale}.csv")

lr = 1e-2
batch_sz = 128
n_epoch = 1000

n_worker = 2
seed = 123
