import os
import torch
import pandas as pd
import torch.utils.tensorboard as tb
from tqdm import tqdm
from datetime import date
from utils import train_transform1, train_transform2, train_transform3, train_transform4, train_transform5
# from dataloader_tio import AutopetDataloaderTio
from dataloader_tio_resampling import AutopetDataloaderTioRes
from dataloader_tio_aug import AutopetDataloaderTioAug_Zresamplowany, AutopetDataloaderTioAug_Zwykly
from torch.utils.tensorboard import SummaryWriter
import warnings

class SegBaseline():
    """
    Class with implementation of methods needed to load data from csv files
    """
    def __init__(self, root_path: str, csv_folder: str) -> None:
        self.root_path = root_path
        self.csv_folder = csv_folder

    def load_datasets(self, root_path: str):
        """
        Load data from csv files and preprocess by AutopeDataloaderTio class.

        Args:
            root_path (str): path to directory with csv files.
        
        Returns:
            train_dataset, val_dataset, test_dataset: dataloader_tio.AutopetDataloaderTio (image, label).
        """
        csv_path = root_path + '/' + self.csv_folder

        train_data = pd.read_csv(os.path.join(csv_path, 'train_dataset.csv'))
        val_data = pd.read_csv(os.path.join(csv_path, 'val_dataset.csv'))
        test_data = pd.read_csv(os.path.join(csv_path, 'test_dataset.csv'))

        ct_images_tr = train_data['CT']
        pet_images_tr = train_data['PET']
        # suv_images_tr = train_data['SUV']
        labels_tr = train_data['MASKS']

        ct_images_val = val_data['CT']
        pet_images_val = val_data['PET']
        # suv_images_val = val_data['SUV']
        labels_val = val_data['MASKS']

        ct_images_test = test_data['CT']
        pet_images_test = test_data['PET']
        # suv_images_test = test_data['SUV']
        labels_test = test_data['MASKS']

        train_dataset = AutopetDataloaderTioAug_Zwykly(ct_images_tr, pet_images_tr, labels_tr, train_transform5)
        val_dataset = AutopetDataloaderTioAug_Zwykly(ct_images_val, pet_images_val, labels_val)
        test_dataset = AutopetDataloaderTioAug_Zwykly(ct_images_test, pet_images_test, labels_test)

        return train_dataset, val_dataset, test_dataset
    
    def load_dataloaders(self):
        """
        Load preprocessed datasets and process by troch dataloader.

        Returns:
            train_loader, val_loader, test_loader: torch.utils.data.dataloader.DataLoader (image, label)
        """
        train_dataset, val_dataset, test_dataset = self.load_datasets(self.root_path)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

        return train_loader, val_loader, test_loader

    def training(self, model, save_path, loss_function, optimizer, epochs, name, writer, scheduler=None, checkpoint=None):
        torch.autograd.set_detect_anomaly(True)
        device = torch.device("cuda:0")
        model = model.to(device)

        train_loader, val_loader, _ = self.load_dataloaders()

        training_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)

        current_epoch = 0
        prev_epochs = 0
        loss_history = []
        val_loss_history = []
        grad_norm_history = []
        best_loss = float("inf")

        if checkpoint != None:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss_history = checkpoint['train_loss']
            val_loss_history = checkpoint['val_loss']
            best_loss = checkpoint['best_loss']
            current_epoch = checkpoint['epochs']

        now = date.today()
        for epoch in range(epochs):
            current_epoch += 1
            print(f'Current epoch: {current_epoch} for {name}')
            train_loss = 0.0
            val_loss = 0.0

            batch_grad = []

            for image, label in tqdm(train_loader):
                # image = image.permute(1, 0, 5, 2, 3, 4)
                # image = image.permute(5, 2, 3, 4, 0, 1)
                # image = image.squeeze(0)
                image = image.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                pred = model(image)

                print(pred.shape)

                loss = loss_function(pred[:,0,:,:,:], label[:,0,:,:,:])
                loss.backward()
                gradient_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]), 2)
                batch_grad.append(gradient_norm)

                optimizer.step()
                train_loss += loss.item()

                if torch.isnan(loss) or torch.isinf(loss):
                    print('Numerical error in loss calculation!')
                    break

            if scheduler != None:
                scheduler.step()

            loss_history.append(train_loss / training_size)
            grad_norm_history.append(torch.mean(torch.stack(batch_grad)))

            # writer.add_scalars(f'Loss', {'Train Loss': train_loss / training_size,}, global_step = epoch)
                
            for image, label in tqdm(val_loader):
                # image = image.permute(1, 0, 5, 2, 3, 4)
                # image = image.squeeze(0)
                image = image.to(device)
                label = label.to(device)
                
                pred = model(image)
                loss = loss_function(pred, label)

                val_loss += loss.item()

            val_loss_history.append(val_loss / val_size)

            writer.add_scalars(f'{name}', {f'Train-{name}': train_loss / training_size,f'Val-{name}': val_loss / val_size}, global_step = current_epoch)
            writer.add_scalars(f'Gradient-{name}', {f'Gradient-{name}': torch.mean(torch.stack(batch_grad)),}, global_step = current_epoch)

            if (epoch+1)%2==0:
                print('Train Loss', train_loss / training_size)
                print('Val Loss', val_loss / val_size)
                # model.eval()
                # with torch.no_grad():
                #     dc1, hd = count(model, val_loader, dc, hd95)

                # model.train()
                # print('\nDC after ' + str(epoch) + ' epochs: ' + str(dc1))
                # print('HD after ' + str(epoch) + ' epochs: ' + str(hd))

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': loss_history,
                    'val_loss': val_loss_history,
                    'best_loss': best_loss,
                    'epochs': current_epoch,
                    'gradient': grad_norm_history
                    }, os.path.join(save_path, f'{name}-{now}-epochs-' + str(current_epoch) + '.pt'))

            if val_loss / val_size < best_loss:
                best_loss = val_loss / val_size

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': loss_history,
                    'val_loss': val_loss_history,
                    'gradient': grad_norm_history
                    }, os.path.join(save_path, f'{name}-{now}-best.pt'))

        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': loss_history,
            'val_loss': val_loss_history,
            'best_loss': best_loss,
            'epochs': current_epoch,
            'gradient': grad_norm_history
            }, os.path.join(save_path, f'{name}-{now}-epochs-' + str(current_epoch) + '.pt'))

