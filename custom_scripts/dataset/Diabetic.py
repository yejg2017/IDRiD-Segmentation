import os
import  random
import numpy as np
import utils
import visualize
import matplotlib.pyplot as plt

import skimage
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

def get_idx(root):
        mode="training"
        classes=os.listdir(os.path.join(root,"annotations",mode))
        image_dir=os.path.join(root,"images",mode)
        annotation_dir=[os.path.join(root,"annotations",mode,annotation) for annotation in classes]
        print(annotation_dir)
        image_dict={annotation:{"path":[],"annotation":[]} for annotation in classes}
        for i,(annotation,ann_dir) in enumerate(zip(classes,annotation_dir)):
            anns=os.listdir(ann_dir)  # annotation: all grouth image
            for ann in anns:
                index=ann.split("_")
                idx=index[0]+"_"+index[1]
                image=os.path.join(image_dir,idx+".jpg") # original image path
                image_dict[annotation]["path"].append(image)
                image_dict[annotation]["annotation"].append(os.path.join(ann_dir,ann))
        #return image_dict
        num_images=0
        for label in classes:
            for i,(path,annotation) in enumerate(zip(image_dict[label]["path"],image_dict[label]["annotation"])):
                print(path,"\t",annotation)
                num_images+=1
        print("There are %d annotations\n"%num_images)

def match_annotation(root):
        mode="training"
        image_dir=os.path.join(root,"images",mode)
        classes=os.listdir(os.path.join(root,"annotations",mode))
        #annotation_dirs=[os.path.join(self.root,"annotations",self.mode,annotation) for annotation in self.classes]

        images_dict={img:{} for img in os.listdir(image_dir)}
        for img_dir in os.listdir(image_dir):
            idx=img_dir.split(".")[0]
            for label in classes:
                annotation_dir=os.path.join(root,"annotations",mode,label)
                anns=os.listdir(annotation_dir)
                for ann in anns:
                    if idx in ann:
                        images_dict[img_dir][label]=ann
                        break
        return images_dict

class DiabeticDataset(utils.Dataset):
    def __init__(self,root,label,is_training=True):
        super(DiabeticDataset,self).__init__()
        self.root=root
        self.label=label
        self.is_training=is_training
        self.mode="training" if is_training else "testing"
        self.images,self.annotations=self.get_idx()
        #self.add_class("IDRiD",0,"BG")
        self.add_class("IDRiD",1,label)
        self.height,self.width=2848,4288
        for i,(image,annotation) in enumerate(zip(self.images,self.annotations)):
            self.add_image("IDRiD",image_id=i+1,classes=label,
                           path=image,mask=annotation,height=self.height,width=self.width)
    def get_idx(self):
        image_dir=os.path.join(self.root,"images",self.mode)
        annotation_dir=os.path.join(self.root,"annotations",self.mode,self.label)  #decide the origin image whether in original image
        images=[]
        annotations=[]
        self.ids=[]
        for index in os.listdir(annotation_dir):
            idx=index.split("_")
            ix=idx[0]+"_"+idx[1]
            self.ids.append(int(idx[1]))
            image_idx=os.path.join(image_dir,ix+".jpg")  # raw image idx
            label_idx=os.path.join(annotation_dir,index)
            images.append(image_idx)
            annotations.append(label_idx)
        assert len(images)==len(annotations)
        #self.image_ids=image_id
        return images,annotations
    def image_reference(self, image_id):
        assert image_id in self.image_ids
        info = self.image_info[image_id]
        return info['path']
    def load_image(self, image_id):
        info=self.image_info[image_id]
        image=skimage.io.imread(info["path"])
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image
    def load_mask(self, image_id):
        info=self.image_info[image_id]
        classes=info["classes"]  # consistent with mask
        classes=classes if isinstance(classes,list) else [classes]
        mask = np.zeros([info["height"],info["width"],len(classes)],dtype=np.int32)
        class_ids = np.zeros((len(classes)), dtype=np.int32)
        #class_ids = np.array([self.class_names.index(s[0]) for s in classes])
        m=Image.open(info["mask"]).convert("P")
        for i,c in enumerate(classes):
            mask[:,:,i]=m
            class_ids[i]=1
        return mask.astype(np.bool), class_ids.astype(np.int32)


# Only for one classes
class MulDiabeticDataset(utils.Dataset):
    def __init__(self,root,is_training=True,image_size=None,transform=None):
        super(MulDiabeticDataset,self).__init__()
        self.root=root
        self.mode="training" if is_training else "testing"
        self.image_size=image_size
        self.transform=transform
        self.is_training=is_training

        self.classes=os.listdir(os.path.join(self.root,"annotations",self.mode))
        self.images_dict=self.match_annotation()

        #self.add_class("IDRiD",0,"BG")
        #self.add_class("IDRiD",1,label)
        self.class_map={}
        for i,label in enumerate(self.classes):
            self.add_class("IDRiD",i+1,label)
            self.class_map[label]=i+1

        self.height,self.width=2848,4288

        for i,(key,value) in enumerate(self.images_dict.items()):
            path=os.path.join(self.root,"images",self.mode,key)
            annotation=[os.path.join(self.root,"annotations",self.mode,annotation,idx) for annotation,idx in self.images_dict[key].items()]
            labels=[key for key in self.images_dict[key].keys()]
            self.add_image("IDRiD",image_id=i,classes=labels,
                           path=path,mask=annotation,
                           height=self.height,width=self.width)

    def match_annotation(self):

        image_dir=os.path.join(self.root,"images",self.mode)
        #annotation_dirs=[os.path.join(self.root,"annotations",self.mode,annotation) for annotation in self.classes]

        images_dict={img:{} for img in os.listdir(image_dir)}
        for img_dir in os.listdir(image_dir):
            idx=img_dir.split(".")[0]
            for label in self.classes:
                annotation_dir=os.path.join(self.root,"annotations",self.mode,label)
                anns=os.listdir(annotation_dir)
                for ann in anns:
                    if idx in ann:
                        images_dict[img_dir][label]=ann
                        break
        return images_dict

    def image_reference(self, image_id):
        assert image_id in self.image_ids
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info=self.image_info[image_id]
        image=skimage.io.imread(info["path"])
        if self.image_size is not None:
            image=skimage.transform.resize(image,(self.image_size,self.image_size))
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info=self.image_info[image_id]
        classes=info["classes"]  # consistent with mask
        classes=classes if isinstance(classes,list) else [classes]

        mask = np.zeros([info["height"],info["width"],len(self.class_map)],dtype=np.int32)
        if self.image_size is not None:
            mask = np.zeros([self.image_size,self.image_size,len(self.class_map)],dtype=np.int32)
        class_ids = np.zeros((len(self.class_map)), dtype=np.int32)
        #class_ids = np.array([self.class_names.index(s[0]) for s in classes])
        #m=Image.open(info["mask"]).convert("P")
        for i,c in enumerate(classes):
            m=Image.open(info["mask"][i]).convert("P")
            if self.image_size is not None:
                m=m.resize((self.image_size,self.image_size),Image.ANTIALIAS)
            t=self.class_map[c]-1
            mask[:,:,t]=m
            class_ids[t]=self.class_map[c]

        return mask.astype(np.bool), class_ids.astype(np.int32)

    def __len__(self):
        return len(self.image_ids)

# 将多个mask黏在image上
def apply_masks(image,masks):
    N=masks.shape[-1]
    colors=visualize.random_colors(N)
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        masked_image=visualize.apply_mask(masked_image,masks[:,:,i],colors[i])
    return masked_image,(masked_image-image).astype(np.uint32)


class Iterator(object):
    def __init__(self,dataset,minibatch_size):
        self.dataset=dataset
        self.dataset_size=len(dataset.image_ids)
        self.minibatch_size=minibatch_size

        if self.minibatch_size > self.dataset_size:
            print('Warning: dataset size should be no less than minibatch size.')
            print('Set minibatch size equal to dataset size.')
            self.minibatch_size = self.dataset_size
        self.currrnt_index=0

    def reset_index(self):
        self.currrnt_index=0

    def shuffle_dataset(self, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(self.dataset.image_ids)

    def next_minibatch(self):

        if self.currrnt_index+self.minibatch_size>=self.dataset_size:
            self.reset_index()
            self.shuffle_dataset()

        images=[]
        masks=[]
        for index in range(self.currrnt_index,self.currrnt_index+self.minibatch_size):
            image=self.dataset.load_image(index)
            mask,class_ids=self.dataset.load_mask(index)
            #_,mask=apply_masks(image,mask)
            #mask=skimage.color.rgb2gray(mask)

            images.append(image) #[bacth_size,image_size,image_size,3]
            masks.append(mask)  #[batch_size,image_size,image_size,num_classes]
        images=np.asarray(images,dtype=np.float32)
        masks=np.asarray(masks,dtype=np.uint8)
        #masks=np.expand_dims(masks,axis=3)

        self.currrnt_index+=self.minibatch_size
        return images,masks


### Show Diabetic Dataset class demo whether sucess
if __name__=="__main__":
    # dataset=DiabeticDataset(root="D:\\DataSet\\Image\\Indian-Diabetic\\Segmentation",label="OpticDisc")
    # dataset.prepare()
    # image_id = random.choice(dataset.image_ids)
    # image= dataset.load_image(image_id)
    # msk, class_ids = dataset.load_mask(image_id)
    # # Compute Bounding box
    # bbox = utils.extract_bboxes(msk)
    #
    # # Display image and additional stats
    # print("image_id ", image_id, dataset.image_reference(image_id))
    # # Display image and instances
    # visualize.display_instances(image, bbox, msk, class_ids, dataset.class_names)

    muldataset=MulDiabeticDataset(root="D:\\DataSet\\Image\\Indian-Diabetic\\Segmentation")
    muldataset.prepare()

    image_id = random.choice(muldataset.image_ids)
    image= muldataset.load_image(image_id)
    msk, class_ids = muldataset.load_mask(image_id)
    # Compute Bounding box
    bbox = utils.extract_bboxes(msk)

    # Display image and additional stats
    print("image_id ", image_id, muldataset.image_reference(image_id))
    # Display image and instances
    #visualize.display_instances(image, bbox, msk, class_ids, muldataset.class_names)
    visualize.display_segmentation(image,bbox,msk,class_ids,muldataset.class_names)
    masked_image,masked=apply_masks(image,msk)
    plt.imshow(masked)
    plt.show()
    # Load and display random samples
    # image_ids = np.random.choice(muldataset.image_ids, 2)
    # for image_id in image_ids:
    #     image = muldataset.load_image(image_id)
    #     mask, class_ids = muldataset.load_mask(image_id)
    #     visualize.display_top_masks(image, mask, class_ids, muldataset.class_names)


