# tensorboard usage:
# tensorboard --log_dir log_dir [--port port]
# a tutorial ref:
# https://www.datacamp.com/tutorial/tensorboard-tutorial
import torch
from model import SRCNN
from config import *
from tqdm import tqdm
from dataset import TrainDataset, ValDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import calc_psnr, save_ckpt, load_ckpt, Recorder


def train():
    device = torch.device('cuda:0')
    model = SRCNN()
    model = model.to(device)

    train_dataset = TrainDataset(train_file)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_sz,
                                  shuffle=True, num_workers=n_worker, pin_memory=True, drop_last=True)
    val_dataset = ValDataset(val_file)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': lr * 0.1},
    ], lr=lr, momentum=0.9)

    start_epoch, best_epoch, best_psnr = load_ckpt(
        model, optimizer, from_pth=False)

    summary_writer = SummaryWriter(scalars_dir)

    for epoch in range(start_epoch, n_epoch):
        with tqdm(total=len(train_dataset) - len(train_dataset) % batch_sz) as t:
            t.set_description(f'epoch: {epoch + 1} / {n_epoch}')

            epoch_loss_recorder = Recorder()
            model.train()
            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                preds = model(inputs)
                loss = criterion(preds, labels)

                epoch_loss_recorder.update(loss.item(), len(inputs))

                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_loss_recorder.avg))
                t.update(len(inputs))

        # validate
        epoch_psnr_recorder = Recorder()
        model.eval()
        for i, data in enumerate(val_dataloader):
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs)
            preds = preds.clamp(0.0, 1.0)

            psnr = calc_psnr(preds, labels)
            epoch_psnr_recorder.update(psnr, len(inputs))
        print("psnr {:.2f}".format(epoch_psnr_recorder.avg))

        summary_writer.add_scalar('Loss', epoch_loss_recorder.avg, epoch)
        summary_writer.add_scalar('PSNR', epoch_psnr_recorder.avg, epoch)

        best_epoch, best_psnr = save_ckpt(
            model, optimizer, epoch, epoch_loss_recorder.avg, epoch_psnr_recorder.avg, best_epoch, best_psnr)


if __name__ == '__main__':
    train()
