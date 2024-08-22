import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import cv2

from loss_decom_TDN import Decom_Loss

def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "test")
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)
    assert os.path.exists(val_root), "val root: {} does not exist.".format(val_root)

    train_images_low_path = []
    train_images_high_path = []
    val_images_low_path = []
    val_images_high_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]
    train_high_root = os.path.join(train_root, "high")
    train_low_root= os.path.join(train_root, "low")

    val_high_root = os.path.join(val_root, "high")
    val_low_root = os.path.join(val_root, "low")
    train_low_path = [os.path.join(train_low_root, i) for i in os.listdir(train_low_root)
                  if os.path.splitext(i)[-1] in supported]
    train_high_path= [os.path.join(train_high_root, i) for i in os.listdir(train_high_root)
                  if os.path.splitext(i)[-1] in supported]

    val_low_path = [os.path.join(val_low_root, i) for i in os.listdir(val_low_root)
                  if os.path.splitext(i)[-1] in supported]
    val_high_path= [os.path.join(val_high_root, i) for i in os.listdir(val_high_root)
                  if os.path.splitext(i)[-1] in supported]

    assert len(train_low_path)==len(train_high_path),' The length of train dataset does not match. low:{}, high:{}'.format(len(train_low_path),len(train_high_path))
    assert len(val_low_path)==len(val_high_path),' The length of val dataset does not match. low:{}, high:{}'.format(len(val_low_path),len(val_high_path))
    print("image pair check finish")

    for index in range(len(train_low_path)):
        img_low_path=train_low_path[index]
        img_high_path=train_high_path[index]
        train_images_low_path.append(img_low_path)
        train_images_high_path.append(img_high_path)

    for index in range(len(val_low_path)):
        img_low_path=val_low_path[index]
        img_high_path=val_high_path[index]
        val_images_low_path.append(img_low_path)
        val_images_high_path.append(img_high_path)

    total_dataset_nums = len(train_low_path) + len(train_high_path) + len(val_low_path) + len(val_high_path)
    print("{} images were found in the dataset.".format(total_dataset_nums))
    print("{} low light images for training.".format(len(train_low_path)))
    print("{} normal light images for training ref.".format(len(train_high_path)))
    print("{} low light images for validation.".format(len(val_low_path)))
    print("{} normal light images for validation ref.".format(len(val_high_path)))

    return train_low_path, train_high_path, val_low_path, val_high_path

def train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()
    loss_function = Decom_Loss()

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    accu_total_loss = torch.zeros(1).to(device)
    accu_rec_loss = torch.zeros(1).to(device)
    accu_equal_R_loss = torch.zeros(1).to(device)
    accu_smooth_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_low, I_high = data

        if torch.cuda.is_available():
            I_low = I_low.to(device)
            I_high = I_high.to(device)

        R_low, L_low = model(I_low)
        R_high, L_high = model(I_high)

        loss, loss_rec, loss_equal_R, loss_smooth = loss_function(R_low, R_high, L_low, L_high, I_low, I_high)

        loss.backward()

        accu_total_loss += loss.detach()
        accu_rec_loss += loss_rec.detach()
        accu_equal_R_loss += loss_equal_R.detach()
        accu_smooth_loss += loss_smooth.detach()

        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = "[train epoch {}] loss: {:.3f}  Rec loss: {:.3f}  equal_R loss: {:.3f}  smooth loss: {:.3f}  lr: {:.6f}".format(epoch, accu_total_loss.item() / (step + 1),
            accu_rec_loss.item() / (step + 1), accu_equal_R_loss.item() / (step + 1), accu_smooth_loss.item() / (step + 1), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return accu_total_loss.item() / (step + 1), accu_rec_loss.item() / (step + 1), accu_equal_R_loss.item() / (step + 1), accu_smooth_loss.item() / (step + 1), lr


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, lr, filefold_path):
    loss_function = Decom_Loss()

    model.eval()

    accu_total_loss = torch.zeros(1).to(device)
    accu_rec_loss = torch.zeros(1).to(device)
    accu_equal_R_loss = torch.zeros(1).to(device)
    accu_smooth_loss = torch.zeros(1).to(device)
    save_epoch = 10

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)
    
    if epoch % save_epoch == 0:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_low, I_high = data

        if torch.cuda.is_available():
            I_low = I_low.to(device)
            I_high = I_high.to(device)

        R_low, L_low = model(I_low)
        R_high, L_high = model(I_high)

        if epoch % save_epoch == 0:
            R_low_img = tensor2numpy_R(R_low)
            R_high_img = tensor2numpy_R(R_high)
            L_low_img = tensor2numpy_L(L_low)
            L_high_img = tensor2numpy_L(L_high)
            save_pic(R_low_img, evalfold_path, str(step) + "_R_low")
            save_pic(R_high_img, evalfold_path, str(step) + "_R_high")
            save_pic(L_low_img, evalfold_path, str(step) + "_L_low")
            save_pic(L_high_img, evalfold_path, str(step) + "_L_high")


        loss, loss_rec, loss_equal_R, loss_smooth = loss_function(R_low, R_high, L_low, L_high, I_low, I_high)

        accu_total_loss += loss
        accu_rec_loss += loss_rec
        accu_equal_R_loss += loss_equal_R
        accu_smooth_loss += loss_smooth

        data_loader.desc = "[val epoch {}] loss: {:.3f}  Rec loss: {:.3f}  equal_R loss: {:.3f}  smooth loss: {:.3f}  lr: {:.6f}".format(epoch, accu_total_loss.item() / (step + 1),
            accu_rec_loss.item() / (step + 1), accu_equal_R_loss.item() / (step + 1),  accu_smooth_loss.item() / (step + 1), lr)

    return accu_total_loss.item() / (step + 1), accu_rec_loss.item() / (step + 1), accu_equal_R_loss.item() / (step + 1), accu_smooth_loss.item() / (step + 1)

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def save_pic(outputpic, path, index : str):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    save_path = os.path.join(path, index + ".png")
    cv2.imwrite(save_path, outputpic)

def tensor2numpy_R(R_tensor):
    R = R_tensor.squeeze(0).cpu().detach().numpy()
    R = np.transpose(R, [1, 2, 0])
    return R

def tensor2numpy_L(L_tensor):
    L = L_tensor.squeeze(0)
    L_3 = torch.cat([L, L, L], dim=0)
    L_3 = L_3.cpu().detach().numpy()
    L_3 = np.transpose(L_3, [1, 2, 0])
    return L_3