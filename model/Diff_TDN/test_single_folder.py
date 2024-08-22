import torch
from PIL import Image
from torchvision import transforms
from TDN_network import DecomNet as create_model
import numpy as np
import cv2
import os


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.ToTensor()])

    root = ""
    assert os.path.exists(root), "file: '{}' dose not exist.".format(root)

    images_path=loadfiles(root=root)
    for index in range(len(images_path)):
        assert os.path.exists(images_path[index]), "file: '{}' dose not exist.".format(images_path[index])
    print("path checking complete!")
    print("confirmly find {} images for computing".format(len(images_path)))

    model = create_model().to(device)
    model_weight_path = "./weights/checkpoint_LOL_Diff_TDN.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])
    model.eval()
    for img_path in images_path:
        img = Image.open(img_path)
        img = resize(img)
        img = data_transform(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            R, L = (model(img.to(device)))

        R = R.squeeze(0).detach().cpu().numpy()
        L = torch.cat([L,L,L],dim=1)
        L = L.squeeze(0).detach().cpu().numpy()
        R = np.transpose(R,(1,2,0))
        L = np.transpose(L,(1,2,0))
        name=getnameindex(img_path)
        savepic(R, name, flag="R")
        savepic(L, name, flag="L")

def savepic(outputpic, name, flag):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]

    root = "./results/LOL_high_eval"
    root_path = os.path.join(root, flag)

    if os.path.exists("./results") is False:
        os.makedirs("./results")
    if os.path.exists(root) is False:
        os.makedirs(root)
    if os.path.exists(root_path) is False:
        os.makedirs(root_path)
    path = root_path + "/{}.png".format(name)
    cv2.imwrite(path, outputpic)
    assert os.path.exists(path), "file: '{}' dose not exist.".format(path)
    print("complete compute {}.png and save".format(name))

def loadfiles(root):
    images_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP"]
    images = [os.path.join(root, i) for i in os.listdir(root)
              if os.path.splitext(i)[-1] in supported]
    for index in range(len(images)):
        img_path = images[index]
        images_path.append(img_path)

    print("find {} images for computing.".format(len(images_path)))
    return images_path

def getnameindex(path):
    assert os.path.exists(path), "file: '{}' dose not exist.".format(path)
    path = path.replace("\\", "/")
    label = path.split("/")[-1].split(".")[0]
    return label

def resize(image):
    original_width, original_height = image.size

    new_width = original_width - (original_width % 8)
    new_height = original_height - (original_height % 8)
    resized_image = image.resize((new_width, new_height))
    return resized_image

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()