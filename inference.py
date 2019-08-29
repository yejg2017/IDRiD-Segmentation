import torch
import  torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
#import argparse
import cv2
import random

from custom_scripts.dataset.Diabetic import  MulDiabeticDataset
from Unet_Series import R2AttU_Net

image_size=224

def inference(model,image_path,pre_trained_path):
    classes={1:'Haemorrhages',
             2:'HardExudates',
             3:'Microaneurysms',
             4:'OpticDisc',
             5:'SoftExudates'}

    state_dict = torch.load(pre_trained_path)
    model.load_state_dict(state_dict)
    model.eval()

    raw_image=cv2.imread(image_path)
    raw_image=cv2.resize(raw_image,(image_size,image_size))
    image=torch.from_numpy(np.transpose(raw_image,(2,0,1))).unsqueeze(0).float()

    with torch.no_grad():
        logits=model(image)
        print(logits.shape)

    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()

    labelmap = np.argmax(probs, axis=0)
    labels = np.unique(labelmap)
    #return labelmap,labels

    # # Show result for each class
    rows = np.floor(np.sqrt(len(labels) + 1))
    cols = np.ceil((len(labels) + 1) / rows)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(rows, cols, 1)
    ax.set_title("Input image")
    ax.imshow(raw_image[:, :, ::-1])
    ax.axis("off")

    for i, label in enumerate(labels):
        mask = labelmap == label
        ax = plt.subplot(rows, cols, i + 2)
        ax.set_title(classes[label+1])
        ax.imshow(raw_image[..., ::-1])
        ax.imshow(mask.astype(np.float32), alpha=0.5)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return


if __name__=="__main__":
    pre_trained_path="D:\\Work\\Programming\\Python\\DeepLearning\\Pytorch\\CNN\Segmentation\\IDRiD-MulClass Segmetation\\trained_models\\IDRiD_Diabetic\\R2AttU_Net\\R2AttU_Net_only_params_trained.pth "
    model=R2AttU_Net(img_ch=3,output_ch=5)
    muldataset=MulDiabeticDataset(root="D:\\DataSet\\Image\\Indian-Diabetic\\Segmentation")
    muldataset.prepare()

    image_id = random.choice(muldataset.image_ids)
    image_path=muldataset.image_info[image_id]["path"]

    inference(model,image_path,pre_trained_path)


