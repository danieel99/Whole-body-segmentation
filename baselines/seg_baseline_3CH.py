import os
import torch
import warnings
import numpy as np
import pandas as pd
import torchio as tio
import torch.utils.tensorboard as tb
from datetime import date
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from eval_functions import count, dc, hd95
from dataloader_tio_PB import AutopetDataloaderTioPB3CH, MyTransform3CH
from transforms import train_transform1, train_transform2, train_transform3, train_transform4, train_transform5

class CustomLabelSampler(tio.data.sampler.LabelSampler):
    def __init__(self, patch_size):
        self.dynamic_patch_size = patch_size
        super().__init__(self.dynamic_patch_size)

    def __call__(self, sample):
        label_image = sample['label'][tio.DATA].numpy()
        if np.any(label_image == 1):  
            self.label_probabilities = {0: 0.1, 1: 0.9}
        else:
            self.label_probabilities = {0: 1.0}
        
        # Tworzymy nowy obiekt LabelSampler z odpowiednimi wartościami label_probabilities
        specific_sampler = tio.data.LabelSampler(patch_size=self.dynamic_patch_size, label_name='label', label_probabilities=self.label_probabilities)
        return specific_sampler(sample)

class SegBaselineBB3CH():
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

        train_data = pd.read_csv(os.path.join(csv_path, 'train_dataset_3CH.csv'))
        val_data = pd.read_csv(os.path.join(csv_path, 'val_dataset_3CH.csv'))
        test_data = pd.read_csv(os.path.join(csv_path, 'test_dataset_3CH.csv'))

        ct_images_tr = train_data['CT']
        pet_images_tr = train_data['PET']
        output_images_tr = train_data['OUTPUT']
        labels_tr = train_data['MASKS']

        ct_images_val = val_data['CT']
        pet_images_val = val_data['PET']
        output_images_val = val_data['OUTPUT']
        labels_val = val_data['MASKS']

        ct_images_test = test_data['CT']
        pet_images_test = test_data['PET']
        output_images_test = test_data['OUTPUT']
        labels_test = test_data['MASKS']

        warnings.warn("Start making datasets...")
        train_dataset = AutopetDataloaderTioPB3CH(ct_images_tr.tolist(), pet_images_tr.tolist(), labels_tr.tolist(), output_images_tr.tolist())
        val_dataset = AutopetDataloaderTioPB3CH(ct_images_val.tolist(), pet_images_val.tolist(), labels_val.tolist(), output_images_val.tolist())
        test_dataset = AutopetDataloaderTioPB3CH(ct_images_test.tolist(), pet_images_test.tolist(), labels_test.tolist(), output_images_test.tolist())

        warnings.warn("Start listing subs...)")
        subs_train = [sub for sub in train_dataset]
        subs_val = [sub for sub in val_dataset]
        subs_test = [sub for sub in test_dataset]

        trans_aug = tio.Compose([MyTransform3CH()])
        trans = tio.Compose([MyTransform3CH()])  #tio.transforms.Pad((48,48,48))

        warnings.warn("Start making subject datasets...")
        subjects_dataset_train = tio.SubjectsDataset(subs_train, transform=trans)
        subjects_dataset_val = tio.SubjectsDataset(subs_val, transform=trans)
        subjects_dataset_test = tio.SubjectsDataset(subs_test, transform=trans)

        warnings.warn("Done")
        return subjects_dataset_train, subjects_dataset_val, subjects_dataset_test
    
    def _make_batches(self, subjects_dataset):
        warnings.warn("Start make_branches()")
        patch_size = 96
        queue_length = 300
        samples_per_volume = 80
        label_probabilities = {0: 0.4, 1: 0.6}
        # label_probabilities = {0: 0.1, 1: 0.9}
        # sampler = tio.data.LabelSampler(patch_size=patch_size, label_name='label', label_probabilities=label_probabilities)
        sampler = tio.data.UniformSampler(patch_size)
        # sampler = CustomLabelSampler(patch_size)

        patches_queue = tio.Queue(
            subjects_dataset,
            queue_length,
            samples_per_volume,
            sampler,
            num_workers=8,
            shuffle_subjects=True,
            shuffle_patches=True,
        )

        patches_loader = DataLoader(
            patches_queue,
            batch_size=1,
            num_workers=0,  # this must be 0
        )
        warnings.warn("Patches created!")
        return patches_loader

    def load_dataloaders(self):
        """
        Load preprocessed datasets and process by troch dataloader.

        Returns:
            train_loader, val_loader, test_loader: torch.utils.data.dataloader.DataLoader (image, label)
        """
        s_train_dataset, s_val_dataset, s_test_dataset = self.load_datasets(self.root_path)
        
        train_loader = self._make_batches(s_train_dataset)
        val_loader = self._make_batches(s_val_dataset)
        test_loader = self._make_batches(s_test_dataset)

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
            for patches_batch in tqdm(train_loader):
                image = patches_batch["image"][tio.DATA] 
                label = patches_batch['label'][tio.DATA] 
                
                image = image.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                pred = model(image)  

                loss = loss_function(pred, label)
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
                
            for patches_batch in tqdm(val_loader):
                image = patches_batch["image"][tio.DATA] 
                label = patches_batch['label'][tio.DATA] 
                
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
