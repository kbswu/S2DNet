import torch
import os
from tqdm import tqdm
from data_sfd.build_sfd_dataloader import build_seafog_dataset, build_ybsf_dataset
from models_sfd.encoder_decoder import S2CNet_CNN_MI
from models_sfd.loss_func import build_dicefocal_seg_loss
from models_sfd.mutual_learner import MutualLearningModel

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_generic_seg_models_one_epoch(
        current_epoch,  # the text refers to the training epoch
        train_loader_seg, val_loader_seg,  # dataloaders
        train_model_seg, train_optimizer_seg,  # model and optimizer
        criterion_seg, criterion_seg_2=None,  # loss function
        image_mode=None,  # image mode
        optimizer_mi=None  # mutual learning
):
    running_loss, running_loss_seg_1, running_loss_seg_2 = 0.0, 0.0, 0.0
    pbar = tqdm(train_loader_seg, total=len(train_loader_seg), leave=False)
    for i, data in enumerate(pbar):
        train_optimizer_seg.zero_grad()
        optimizer_mi.zero_grad()
        train_model_seg.train()
        criterion_seg_2.train()
        if image_mode == "train_degrade":
            train_images, train_labels = data['degraded_image'][:, inc_choice, :, :].cuda(), data[
                'degraded_label'].cuda()
        elif image_mode == "train":
            train_images, train_labels = data['augmented_image'][:, inc_choice, :, :].cuda(), data[
                'augmented_label'].cuda()
        else:
            raise NotImplementedError
        seg_pred, feats = train_model_seg(train_images)
        loss_seg_1 = criterion_seg(seg_pred, train_labels)
        loss_seg_2 = criterion_seg_2(seg_pred, feats)
        loss = loss_seg_1 + loss_seg_2
        running_loss_seg_2 += loss_seg_2.item()

        loss.backward()
        train_optimizer_seg.step()
        optimizer_mi.step()
        running_loss += loss.item()
        running_loss_seg_1 += loss_seg_1.item()
        pbar.set_description(f"Epoch {current_epoch}")
        if criterion_seg_2:
            pbar.set_postfix({'loss': running_loss / (i + 1), 'loss_seg_1': running_loss_seg_1 / (i + 1),
                              'loss_seg_2': running_loss_seg_2 / (i + 1)})
        else:
            pbar.set_postfix({'loss': running_loss / (i + 1), 'loss_seg_1': running_loss_seg_1 / (i + 1)})

        # save model
        # if i % 1 == 0:
        #     torch.save(train_model_seg.state_dict(), f'D:/MS_Seg/S2DNet_EXAM/{inc_string}.pth')


if __name__ == "__main__":

    using_train_mode = "train"  # "train_degrade" or "train"
    using_test_mode = "test"  # "test_degrade" or "test"
    dataset_names = ["ybsf", "seafog"]
    incs_dict = {
        "4": [2, 3, 4, 13],  # Generic RGB
        "8": [0, 2, 3, 4, 5, 6, 7, 11],  # Li et. al https://doi.org/10.1007/s00521-022-07602-w
        "HU": [0, 1, 2, 3, 4, 5, 13],  # Hu et. al https://doi.org/10.1109/JSTARS.2023.3257042
        "16": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Tang et. al https://doi.org/10.3390/rs14215570
    }
    band_names = ['16']
    for band_name in band_names:
        inc_choice = incs_dict[band_name]
        num_inc = len(inc_choice)
        inc_string = ''.join(map(str, inc_choice))

        print(f"Training on '{using_train_mode}' with '{using_test_mode}'...")
        for dataset_name in dataset_names:
            print(f"Training on {dataset_name} with {inc_string}...")
            if dataset_name == "seafog":
                train_loader, val_loader, test_loader = build_seafog_dataset(train_mode=using_train_mode,
                                                                             test_mode=using_test_mode,
                                                                             image_size=256, batch_size=16)
            elif dataset_name == "ybsf":
                train_loader, val_loader, test_loader = build_ybsf_dataset(train_mode=using_train_mode,
                                                                           test_mode=using_test_mode,
                                                                           image_size=256, batch_size=16)
            else:
                raise NotImplementedError

            # model
            if dataset_name == "seafog":
                model = S2CNet_CNN_MI(num_inc, n_classes=2).cuda()
                mutual_model = MutualLearningModel(num_classes=2).cuda()
            elif dataset_name == "ybsf":
                model = S2CNet_CNN_MI(num_inc, n_classes=4).cuda()
                mutual_model = MutualLearningModel(num_classes=4).cuda()
            else:
                raise NotImplementedError

            # loss
            criterion = build_dicefocal_seg_loss(dataset_name=dataset_name).cuda()

            # optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            optimizer_mi = torch.optim.Adam(mutual_model.parameters(), lr=1e-3)

            # training
            for epoch in range(1, 100):
                train_generic_seg_models_one_epoch(
                    current_epoch=epoch,
                    train_loader_seg=train_loader, val_loader_seg=val_loader,
                    train_model_seg=model, train_optimizer_seg=optimizer,
                    criterion_seg=criterion, criterion_seg_2=mutual_model,
                    image_mode=using_train_mode,
                    optimizer_mi=optimizer_mi
                )
