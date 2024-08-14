# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

import os 
import os.path as osp
import glob
import tarfile
import zipfile
import numpy as np 
import gdown
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset as TorchDataset
import torch



def check_isfile(fpath):
    """Check if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        print('No file found at "{}"'.format(fpath))
    return isfile

def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath="", label=0, domain=0, classname=""):
        assert isinstance(impath, str)
        assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname

def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    return Image.open(path).convert("RGB")


class DatasetWrapper(TorchDataset):

    def __init__(self,  data_source, transform=None, is_train=False):
        
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        # Apply transformations to an image K times (during training)
        self.k_tfm = 1 if is_train else 1
        self.return_img0 = False

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )
     
        to_tensor = []   
        to_tensor += [T.Resize((224,224))]
        to_tensor += [T.ToTensor()]
        normalize = T.Normalize(
            # Mean and std (default: ImageNet)
            mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225]
        )
        to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        # if self.return_img0:
        output["img0"] = self.to_tensor(img0)  # without any augmentation

        return output["img0"], output["label"]

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


class OfficeHome(object):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    dataset_dir = "office_home_dg"
    domains = ["art", "clipart", "product", "real_world"]
    dname2domain={"art":0, "clipart":1, "product":2, "real_world":3}
    data_url = "https://drive.google.com/uc?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa"

    def __init__(self, root ='data',  ):

        self.dataset_dir = osp.join(root, self.dataset_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "office_home_dg.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.local_train_list, self.server_val_list = self.read_personalized_data(dataset_dir= self.dataset_dir, input_domains= self.domains, split= "train", server_ratio=0.1)
        self.local_test_list, _ = self.read_personalized_data(dataset_dir= self.dataset_dir, input_domains= self.domains, split= "test")
        self.server_val_all = sum(self.server_val_list, [])
        self.server_test_all = sum(self.local_test_list, [])


    def get_data_loaders (self, batch_size = 64, test_batch_size =64, server_batch_size=128,kd_data_fraction=1):
        num_clients = len(self.domains)
        train_data = {}
        test_data = {}
        for user_id in range(num_clients):  
            train_data.update({user_id: {'dataloader':  torch.utils.data.DataLoader(DatasetWrapper(self.local_train_list[user_id],is_train=True), batch_size= batch_size, num_workers=2, pin_memory=True,
                                shuffle =True), 
                            'indices': self.local_train_list[user_id]  }})

            test_data.update({user_id: {'dataloader':  torch.utils.data.DataLoader(DatasetWrapper(self.local_test_list[user_id]), batch_size= test_batch_size, num_workers=2, pin_memory=True,
                            shuffle =True), 
                            'indices': self.local_test_list[user_id]}})
        

        clients = {
            'train_users': list(train_data.keys()),
            'test_users': list(test_data.keys())
        }
        val_dataloader = torch.utils.data.DataLoader(DatasetWrapper(self.server_val_all), batch_size= server_batch_size, num_workers=2, pin_memory=True)
        

        kd_idx = np.random.choice(list(set(range(len(self.server_val_all)))) , int(len(self.server_val_all)*kd_data_fraction) , replace=False) 
        print("kd_idx len", len(kd_idx), "out of", len(self.server_val_all))
        kd_dataloader = torch.utils.data.DataLoader(DatasetWrapper(self.server_val_all), batch_size= server_batch_size, num_workers=2, pin_memory=True,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(kd_idx))


        test_dataloader = torch.utils.data.DataLoader(DatasetWrapper(self.server_test_all), batch_size= server_batch_size, num_workers=2, pin_memory=True)
        return clients, kd_dataloader, train_data, test_data, val_dataloader,test_dataloader


    def read_personalized_data(self,dataset_dir, input_domains, split,server_ratio =0):
        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory)
            folders.sort()
            items_ = []

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, "*.jpg"))

                for impath in impaths:
                    items_.append((impath, label))

            return items_
        
        def _get_data_list(path_list ):
            itms= []
            for impath, label in path_list:
                class_name = impath.split("/")[-2].lower()
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=self.dname2domain[dname],
                    classname=class_name
                )
                itms.append(item)
            return itms

        
        local_data_items_list =[]
        server_data_items_list =[]
        for _, dname in enumerate(input_domains):
            if split == "train":
                train_dir = osp.join(dataset_dir, dname, "train")
                impath_label_list = _load_data_from_directory(train_dir)
                total_number_samples= len(impath_label_list)
               
                server_idx = np.random.choice(list(set(range(total_number_samples))), int(total_number_samples*server_ratio), replace=False)
                local_idx = [i for i in range(total_number_samples) if i not in server_idx]
                
                server_impath_label_list = [impath_label_list[i] for  i in server_idx]
                local_impath_label_list = [impath_label_list[i] for  i in local_idx]
                
                server_data_items = _get_data_list(server_impath_label_list )
                local_data_items = _get_data_list(local_impath_label_list )

                local_data_items_list.append(local_data_items)
                server_data_items_list.append(server_data_items)
                print(dname, "local_data_items", len(local_data_items), "server_data_items", len(server_data_items))
            elif split == "test":
                split_dir = osp.join(dataset_dir, dname, "val")
                impath_label_list = _load_data_from_directory(split_dir)
                local_data_items = _get_data_list(impath_label_list )
                local_data_items_list.append(local_data_items)
                print(dname, "local_data_items", len(local_data_items))

        return local_data_items_list, server_data_items_list

    
    def read_data(self,dataset_dir, input_domains, split):
        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory)
            folders.sort()
            items_ = []

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, "*.jpg"))

                for impath in impaths:
                    items_.append((impath, label))

            return items_

        items = []

        for _, dname in enumerate(input_domains):
            if split == "all":
                train_dir = osp.join(dataset_dir, dname, "train")
                impath_label_list = _load_data_from_directory(train_dir)
                val_dir = osp.join(dataset_dir, dname, "val")
                impath_label_list += _load_data_from_directory(val_dir)
            else:
                split_dir = osp.join(dataset_dir, dname, split)
                impath_label_list = _load_data_from_directory(split_dir)
            print(impath_label_list)
            for impath, label in impath_label_list:
                class_name = impath.split("/")[-2].lower()
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=self.dname2domain[dname],
                    classname=class_name
                )
                items.append(item)

        return items

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print("Extracting file ...")

        if dst.endswith(".zip"):
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        elif dst.endswith(".tar"):
            tar = tarfile.open(dst, "r:")
            tar.extractall(osp.dirname(dst))
            tar.close()

        elif dst.endswith(".tar.gz"):
            tar = tarfile.open(dst, "r:gz")
            tar.extractall(osp.dirname(dst))
            tar.close()

        else:
            raise NotImplementedError

        print("File extracted to {}".format(osp.dirname(dst)))
