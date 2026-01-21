# import os
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from PIL import Image
# from typing import Tuple, List
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms.functional import InterpolationMode

# from datasets.utils import set_default_from_args
# from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
# from datasets.transforms.denormalization import DeNormalize
# from utils.conf import base_path


# class MyCore50(Dataset):
#     """
#     Core50 dataset loader (128×128) by session and class,
#     downloading and extracting if needed. Provides .data and .targets
#     for compatibility with store_masked_loaders and returns
#     (img_transformed, label, img_not_augmented).
#     """
#     RAW_SIZE = 128
#     N_CLASSES = 50

#     def __init__(self, root: str, transform=None, download: bool = True) -> None:
#         self.root = root
#         self.transform = transform
#         os.makedirs(root, exist_ok=True)
#         extract_dir = os.path.join(root, 'core50_128x128')
#         zip_url = 'http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip'
#         zip_path = os.path.join(root, 'core50_128x128.zip')
#         # download and extract if missing
#         if download and not os.path.isdir(extract_dir):
#             import urllib.request, zipfile
#             urllib.request.urlretrieve(zip_url, zip_path)
#             with zipfile.ZipFile(zip_path, 'r') as zf:
#                 zf.extractall(root)
#             os.remove(zip_path)

#         # scan folder structure: core50_128x128/sXX/objYY/*.jpg
#         scan_dir = extract_dir if os.path.isdir(extract_dir) else root
#         samples: List[Tuple[str, int, int]] = []  # (path, class_id, session_id)
#         for sess in sorted(os.listdir(scan_dir)):
#             sess_dir = os.path.join(scan_dir, sess)
#             if not os.path.isdir(sess_dir):
#                 continue
#             try:
#                 sess_id = int(sess.lstrip('s'))
#             except ValueError:
#                 continue
#             for cls in sorted(os.listdir(sess_dir)):
#                 cls_dir = os.path.join(sess_dir, cls)
#                 if not os.path.isdir(cls_dir):
#                     continue
#                 try:
#                     cls_id = int(cls.lstrip('obj')) - 1
#                 except ValueError:
#                     continue
#                 for fname in sorted(os.listdir(cls_dir)):
#                     if fname.lower().endswith(('.jpg', '.png')):
#                         samples.append((os.path.join(cls_dir, fname), cls_id, sess_id))

#         if not samples:
#             raise FileNotFoundError(f"No Core50 samples found in '{scan_dir}'")

#         paths, classes, sessions = zip(*samples)
#         self.data = np.array(paths)
#         self.targets = np.array(classes)
#         self.sessions = np.array(sessions)
#         self.logits = None

#     def __len__(self) -> int:
#         return len(self.data)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
#         path = self.data[idx]
#         label = int(self.targets[idx])
#         img = Image.open(path).convert('RGB')
#         # produce not-augmented 128x128 tensor
#         raw = img.resize((MyCore50.RAW_SIZE, MyCore50.RAW_SIZE), Image.BICUBIC)
#         not_aug = transforms.ToTensor()(raw)
#         # apply full transform (yields 224x224)
#         img_tr = self.transform(img) if self.transform else transforms.ToTensor()(img)
#         return img_tr, label, not_aug


# class DomainSequentialCore50(ContinualDataset):
#     """Domain-incremental Core50: each of the 11 sessions is a task with 10 classes."""
#     NAME = 'seq-core50'
#     SETTING = 'domain-il'
#     N_TASKS = 11
#     N_CLASSES_PER_TASK = 10
#     SIZE = (224, 224)
#     MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

#     TRANSFORM = transforms.Compose([
#         transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
#         transforms.RandomCrop(SIZE),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#     ])
#     TEST_TRANSFORM = transforms.Compose([
#         transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#     ])

#     def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
#         # load full datasets
#         ds_train = MyCore50(base_path() + 'CORE50/', transform=DomainSequentialCore50.TRANSFORM, download=True)
#         ds_test = MyCore50(base_path() + 'CORE50/', transform=DomainSequentialCore50.TEST_TRANSFORM, download=False)

#         # pick one instance per category (instance IDs 0,5,10,...45)
#         chosen_instances = np.arange(0, MyCore50.N_CLASSES, 5)
#         # filter training set
#         mask_train = np.isin(ds_train.targets, chosen_instances)
#         ds_train.data = ds_train.data[mask_train]
#         ds_train.targets = ds_train.targets[mask_train]
#         ds_train.sessions = ds_train.sessions[mask_train]
#         # filter test set similarly
#         mask_test = np.isin(ds_test.targets, chosen_instances)
#         ds_test.data = ds_test.data[mask_test]
#         ds_test.targets = ds_test.targets[mask_test]
#         ds_test.sessions = ds_test.sessions[mask_test]

#                 # remap instance IDs to [0..9] indices
#         ds_train.targets = np.searchsorted(chosen_instances, ds_train.targets)
#         ds_test.targets = np.searchsorted(chosen_instances, ds_test.targets)

#         # create mammoth wrappers
#         train_loader, test_loader = store_masked_loaders(ds_train, ds_test, self)
#         return train_loader, test_loader

#     @staticmethod
#     def get_transform():
#         return transforms.Compose([
#             transforms.ToPILImage(), DomainSequentialCore50.TRANSFORM
#         ])

#     @set_default_from_args("backbone")
#     def get_backbone():
#         return "vit"

#     @staticmethod
#     def get_loss():
#         return F.cross_entropy

#     @staticmethod
#     def get_normalization_transform():
#         return transforms.Normalize(DomainSequentialCore50.MEAN, DomainSequentialCore50.STD)

#     @staticmethod
#     def get_denormalization_transform():
#         return DeNormalize(DomainSequentialCore50.MEAN, DomainSequentialCore50.STD)

#     @set_default_from_args('batch_size')
#     def get_batch_size(self):
#         return 64

#     @set_default_from_args('n_epochs')
#     def get_epochs(self):
#         return 50

#     def get_class_names(self) -> List[str]:
#         # generate 10 class names per session label order
#         base = [f'class_{i}' for i in range(50)]
#         # we always slice first 10
#         classes = base[:10]
#         self.class_names = fix_class_names_order(classes, self.args)
#         return self.class_names


# import os
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from PIL import Image
# from typing import Tuple, List
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms.functional import InterpolationMode

# from datasets.utils import set_default_from_args
# from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
# from datasets.transforms.denormalization import DeNormalize
# from utils.conf import base_path


# class MyCore50(Dataset):
#     """
#     Core50 dataset loader (128×128) by session and class,
#     downloading and extracting if needed. Provides .data and .targets
#     for compatibility with store_masked_loaders and returns
#     (img_transformed, label, img_not_augmented).
#     """
#     RAW_SIZE = 128
#     N_CLASSES = 50

#     def __init__(self, root: str, transform=None, download: bool = True) -> None:
#         self.root = root
#         self.transform = transform
#         os.makedirs(root, exist_ok=True)
#         extract_dir = os.path.join(root, 'core50_128x128')
#         zip_url = 'http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip'
#         zip_path = os.path.join(root, 'core50_128x128.zip')
#         # download and extract if missing
#         if download and not os.path.isdir(extract_dir):
#             import urllib.request, zipfile
#             urllib.request.urlretrieve(zip_url, zip_path)
#             with zipfile.ZipFile(zip_path, 'r') as zf:
#                 zf.extractall(root)
#             os.remove(zip_path)

#         # scan folder structure: core50_128x128/sXX/objYY/*.jpg
#         scan_dir = extract_dir if os.path.isdir(extract_dir) else root
#         samples: List[Tuple[str, int, int]] = []  # (path, class_id, session_id)
#         for sess in sorted(os.listdir(scan_dir)):
#             sess_dir = os.path.join(scan_dir, sess)
#             if not os.path.isdir(sess_dir):
#                 continue
#             try:
#                 sess_id = int(sess.lstrip('s'))
#             except ValueError:
#                 continue
#             for cls in sorted(os.listdir(sess_dir)):
#                 cls_dir = os.path.join(sess_dir, cls)
#                 if not os.path.isdir(cls_dir):
#                     continue
#                 try:
#                     cls_id = int(cls.lstrip('obj')) - 1
#                 except ValueError:
#                     continue
#                 for fname in sorted(os.listdir(cls_dir)):
#                     if fname.lower().endswith(('.jpg', '.png')):
#                         samples.append((os.path.join(cls_dir, fname), cls_id, sess_id))

#         if not samples:
#             raise FileNotFoundError(f"No Core50 samples found in '{scan_dir}'")

#         paths, classes, sessions = zip(*samples)
#         self.data = np.array(paths)
#         self.targets = np.array(classes)
#         self.sessions = np.array(sessions)
#         self.logits = None

#     def __len__(self) -> int:
#         return len(self.data)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
#         path = self.data[idx]
#         label = int(self.targets[idx])
#         img = Image.open(path).convert('RGB')
#         # not-augmented: 128x128
#         raw = img.resize((MyCore50.RAW_SIZE, MyCore50.RAW_SIZE), Image.BICUBIC)
#         not_aug = transforms.ToTensor()(raw)
#         # transformed: 224x224 pipeline
#         img_tr = self.transform(img) if self.transform else transforms.ToTensor()(img)
#         return img_tr, label, not_aug


# class DomainSequentialCore50(ContinualDataset):
#     """Domain-incremental Core50: each of the 11 sessions is a task with 10 classes."""
#     NAME = 'seq-core50'
#     SETTING = 'domain-il'
#     N_TASKS = 11
#     N_CLASSES_PER_TASK = 10
#     SIZE = (224, 224)
#     MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

#     TRANSFORM = transforms.Compose([
#         transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
#         transforms.RandomCrop(SIZE),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#     ])
#     TEST_TRANSFORM = transforms.Compose([
#         transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
#         transforms.CenterCrop(SIZE[0]),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#     ])

#     def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
#         # load full datasets
#         ds_train = MyCore50(base_path() + 'CORE50/', transform=DomainSequentialCore50.TRANSFORM, download=True)
#         ds_test = MyCore50(base_path() + 'CORE50/', transform=DomainSequentialCore50.TEST_TRANSFORM, download=False)

#         # pick one instance per category (IDs 0,5,10,...45)
#         chosen_instances = np.arange(0, MyCore50.N_CLASSES, 5)
#         mask_train = np.isin(ds_train.targets, chosen_instances)
#         ds_train.data = ds_train.data[mask_train]
#         ds_train.targets = ds_train.targets[mask_train]
#         ds_train.sessions = ds_train.sessions[mask_train]
#         mask_test = np.isin(ds_test.targets, chosen_instances)
#         ds_test.data = ds_test.data[mask_test]
#         ds_test.targets = ds_test.targets[mask_test]
#         ds_test.sessions = ds_test.sessions[mask_test]

#         # remap targets to [0..9]
#         ds_train.targets = np.searchsorted(chosen_instances, ds_train.targets)
#         ds_test.targets = np.searchsorted(chosen_instances, ds_test.targets)

#         # down-sample 50% per class
#         sel_train = []
#         for c in np.unique(ds_train.targets):
#             idxs = np.where(ds_train.targets == c)[0]
#             keep = np.random.choice(idxs, size=len(idxs) // 4, replace=False)
#             sel_train.extend(keep.tolist())
#         sel_train = np.array(sel_train)
#         ds_train.data = ds_train.data[sel_train]
#         ds_train.targets = ds_train.targets[sel_train]
#         ds_train.sessions = ds_train.sessions[sel_train]

#         sel_test = []
#         for c in np.unique(ds_test.targets):
#             idxs = np.where(ds_test.targets == c)[0]
#             keep = np.random.choice(idxs, size=len(idxs) // 4, replace=False)
#             sel_test.extend(keep.tolist())
#         sel_test = np.array(sel_test)
#         ds_test.data = ds_test.data[sel_test]
#         ds_test.targets = ds_test.targets[sel_test]
#         ds_test.sessions = ds_test.sessions[sel_test]

#         # build loaders
#         train_loader, test_loader = store_masked_loaders(ds_train, ds_test, self)
#         return train_loader, test_loader

#     @staticmethod
#     def get_transform():
#         return transforms.Compose([transforms.ToPILImage(), DomainSequentialCore50.TRANSFORM])

#     @set_default_from_args("backbone")
#     def get_backbone():
#         return "vit"

#     @staticmethod
#     def get_loss():
#         return F.cross_entropy

#     @staticmethod
#     def get_normalization_transform():
#         return transforms.Normalize(DomainSequentialCore50.MEAN, DomainSequentialCore50.STD)

#     @staticmethod
#     def get_denormalization_transform():
#         return DeNormalize(DomainSequentialCore50.MEAN, DomainSequentialCore50.STD)

#     @set_default_from_args('batch_size')
#     def get_batch_size(self):
#         return 64

#     @set_default_from_args('n_epochs')
#     def get_epochs(self):
#         return 50

#     def get_class_names(self) -> List[str]:
#         base = [f'class_{i}' for i in range(50)]
#         classes = base[:10]
#         self.class_names = fix_class_names_order(classes, self.args)
#         return self.class_names
# import os
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from PIL import Image
# from typing import Tuple, List
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms.functional import InterpolationMode

# from datasets.utils import set_default_from_args
# from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
# from datasets.transforms.denormalization import DeNormalize
# from utils.conf import base_path


# class MyCore50(Dataset):
#     """
#     Core50 dataset loader (128×128) by session and class,
#     downloading and extracting if needed. Provides .data and .targets
#     for compatibility with store_masked_loaders and returns
#     (img_transformed, label, img_not_augmented).
#     """
#     RAW_SIZE = 128
#     N_CLASSES = 50

#     def __init__(self, root: str, transform=None, download: bool = True) -> None:
#         self.root = root
#         self.transform = transform
#         os.makedirs(root, exist_ok=True)
#         extract_dir = os.path.join(root, 'core50_128x128')
#         zip_url = 'http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip'
#         zip_path = os.path.join(root, 'core50_128x128.zip')
#         # download and extract if missing
#         if download and not os.path.isdir(extract_dir):
#             import urllib.request, zipfile
#             urllib.request.urlretrieve(zip_url, zip_path)
#             with zipfile.ZipFile(zip_path, 'r') as zf:
#                 zf.extractall(root)
#             os.remove(zip_path)

#         # scan folder structure: core50_128x128/sXX/objYY/*.jpg
#         scan_dir = extract_dir if os.path.isdir(extract_dir) else root
#         samples: List[Tuple[str, int, int]] = []  # (path, class_id, session_id)
#         for sess in sorted(os.listdir(scan_dir)):
#             sess_dir = os.path.join(scan_dir, sess)
#             if not os.path.isdir(sess_dir):
#                 continue
#             try:
#                 sess_id = int(sess.lstrip('s'))
#             except ValueError:
#                 continue
#             for cls in sorted(os.listdir(sess_dir)):
#                 cls_dir = os.path.join(sess_dir, cls)
#                 if not os.path.isdir(cls_dir):
#                     continue
#                 try:
#                     cls_id = int(cls.lstrip('obj')) - 1
#                 except ValueError:
#                     continue
#                 for fname in sorted(os.listdir(cls_dir)):
#                     if fname.lower().endswith(('.jpg', '.png')):
#                         samples.append((os.path.join(cls_dir, fname), cls_id, sess_id))
#         if not samples:
#             raise FileNotFoundError(f"No Core50 samples found in '{scan_dir}'")

#         # preload image arrays into memory for replay compatibility
#         paths, classes, sessions = zip(*samples)
#         images = []
#         for p in paths:
#             img = Image.open(p).convert('RGB')
#             img = img.resize((MyCore50.RAW_SIZE, MyCore50.RAW_SIZE), Image.BICUBIC)
#             images.append(np.array(img, dtype=np.uint8))
#         self.data = np.stack(images, axis=0)    # shape (N, H, W, C), uint8
#         self.targets = np.array(classes)
#         self.sessions = np.array(sessions)
#         self.logits = None

#     def __len__(self) -> int:
#         return len(self.data)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
#         # get the preloaded numpy image and label
#         img_np = self.data[idx]               # shape (H, W, C), uint8
#         label = int(self.targets[idx])

#         # convert numpy array to PIL for transformations
#         img_pil = Image.fromarray(img_np)

#         # not-augmented view at RAW_SIZE
#         raw = img_pil.resize((MyCore50.RAW_SIZE, MyCore50.RAW_SIZE), Image.BICUBIC)
#         not_aug = transforms.ToTensor()(raw)

#         # apply full transform (includes resize/crop to 224x224)
#         img_tr = self.transform(img_pil) if self.transform else transforms.ToTensor()(img_pil)

#         return img_tr, label, not_aug


# class DomainSequentialCore50(ContinualDataset):
#     """Domain-incremental Core50: each of the 11 sessions is a task with 10 classes."""
#     NAME = 'seq-core50'
#     SETTING = 'domain-il'
#     N_TASKS = 11
#     N_CLASSES_PER_TASK = 10
#     SIZE = (224, 224)
#     MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

#     TRANSFORM = transforms.Compose([
#         transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
#         transforms.RandomCrop(SIZE),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#     ])
#     TEST_TRANSFORM = transforms.Compose([
#         transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
#         transforms.CenterCrop(SIZE[0]),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#     ])

#     def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
#         # load full datasets
#         ds_train = MyCore50(base_path() + 'CORE50/', transform=DomainSequentialCore50.TRANSFORM, download=True)
#         ds_test = MyCore50(base_path() + 'CORE50/', transform=DomainSequentialCore50.TEST_TRANSFORM, download=True)

#         # pick one instance per category (IDs 0,5,10,...45)
#         chosen = np.arange(0, MyCore50.N_CLASSES, 5)
#         mask_tr = np.isin(ds_train.targets, chosen)
#         ds_train.data = ds_train.data[mask_tr]
#         ds_train.targets = ds_train.targets[mask_tr]
#         ds_train.sessions = ds_train.sessions[mask_tr]
#         mask_te = np.isin(ds_test.targets, chosen)
#         ds_test.data = ds_test.data[mask_te]
#         ds_test.targets = ds_test.targets[mask_te]
#         ds_test.sessions = ds_test.sessions[mask_te]

#         # remap targets to [0..9]
#         ds_train.targets = np.searchsorted(chosen, ds_train.targets)
#         ds_test.targets = np.searchsorted(chosen, ds_test.targets)

#         # down-sample 50% per class
#         sel_tr, sel_te = [], []
#         for c in np.unique(ds_train.targets):
#             idxs = np.where(ds_train.targets == c)[0]
#             keep = np.random.choice(idxs, size=len(idxs)//2, replace=False)
#             sel_tr.extend(keep.tolist())
#         for c in np.unique(ds_test.targets):
#             idxs = np.where(ds_test.targets == c)[0]
#             keep = np.random.choice(idxs, size=len(idxs)//2, replace=False)
#             sel_te.extend(keep.tolist())
#         ds_train.data = ds_train.data[sel_tr]
#         ds_train.targets = ds_train.targets[sel_tr]
#         ds_train.sessions = ds_train.sessions[sel_tr]
#         ds_test.data = ds_test.data[sel_te]
#         ds_test.targets = ds_test.targets[sel_te]
#         ds_test.sessions = ds_test.sessions[sel_te]

#         # build loaders
#         train_loader, test_loader = store_masked_loaders(ds_train, ds_test, self)
#         return train_loader, test_loader

#     @staticmethod
#     def get_transform():
#         return transforms.Compose([transforms.ToPILImage(), DomainSequentialCore50.TRANSFORM])

#     @set_default_from_args("backbone")
#     def get_backbone():
#         return "vit"

#     @staticmethod
#     def get_loss():
#         return F.cross_entropy

#     @staticmethod
#     def get_normalization_transform():
#         return transforms.Normalize(DomainSequentialCore50.MEAN, DomainSequentialCore50.STD)

#     @staticmethod
#     def get_denormalization_transform():
#         return DeNormalize(DomainSequentialCore50.MEAN, DomainSequentialCore50.STD)

#     @set_default_from_args('batch_size')
#     def get_batch_size(self):
#         return 64

#     @set_default_from_args('n_epochs')
#     def get_epochs(self):
#         return 50

#     def get_class_names(self) -> List[str]:
#         """
#         Return the 10 chosen instance names (one per Core50 category).
#         """
#         # base instance names
#         base = [f'class_{i}' for i in range(MyCore50.N_CLASSES)]
#         # chosen instance IDs: one per category
#         chosen = np.arange(0, MyCore50.N_CLASSES, 5)
#         classes = [base[i] for i in chosen]
#         self.class_names = fix_class_names_order(classes, self.args)
#         return self.class_names





# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from torchvision.transforms.functional import to_tensor

# from PIL import Image
# from typing import Tuple, List
# from argparse import Namespace
# from torch.utils.data import DataLoader, random_split
# from torchvision.datasets import ImageFolder
# from torch.utils.data import Subset

# from datasets.utils import set_default_from_args
# from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
# from datasets.transforms.denormalization import DeNormalize
# from utils.conf import base_path


# class Core50(ImageFolder):
#     """
#     Wrapper around torchvision ImageFolder for Core50.
#     Returns (image, label, raw_image).
#     """
#     def __init__(self, root: str, transform=None, target_transform=None):
#         super().__init__(root, transform=transform, target_transform=target_transform)
#         # transform to tensor without normalization for raw input
#         self.raw_transform = to_tensor

#     def __getitem__(self, index: int) -> Tuple[Image.Image, int, torch.Tensor]:
#         path, target = self.samples[index]
#         img = self.loader(path)
#         # raw unaugmented tensor
#         raw = self.raw_transform(img)
#         if self.transform:
#             img = self.transform(img)
#         if self.target_transform:
#             target = self.target_transform(target)
#         return img, target, raw


# class SequentialCore50(ContinualDataset):
#     """Domain-IL Core50: each session is a task (50 instance-classes per task)."""
#     NAME = 'seq-core50'
#     SETTING = 'domain-il'
#     N_CLASSES_PER_TASK = 50
#     N_TASKS = 11
#     SIZE = (128, 128)
#     INDIM = (3, 128, 128)
#     # maximum samples per task (0 = use all)
#     MAX_SAMPLES_PER_TASK = 16000

#     # transforms for train and test
#     TRAIN_TRANSFORM = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     TEST_TRANSFORM = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     @set_default_from_args('batch_size')
#     def get_batch_size(self) -> int:
#         """Number of samples per batch."""
#         return 128

#     @set_default_from_args('n_epochs')
#     def get_epochs(self) -> int:
#         """Number of training epochs."""
#         return 50

#     @staticmethod
#     def get_backbone() -> torch.nn.Module:
#         return "resnet50"

#     @staticmethod
#     def get_loss():
#         return F.cross_entropy

#     @staticmethod
#     def get_transform():
#         # Transforms applied inside Core50
#         return None

#     @staticmethod
#     def get_normalization_transform():
#         return None

#     @staticmethod
#     def get_denormalization_transform():
#         return DeNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#     def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
#         # Session folder name s1..s11 based on current task index/umbc/rs/gokhale/users/shivank2/shivanand/mammoth/data/CORE50/core50_128x128
#         session_folder = f's{self.current_task + 1}'
#         path = os.path.join(base_path()+"CORE50/core50_128x128", session_folder)

#         # Full ImageFolder datasets
#         full_train = Core50(path, transform=SequentialCore50.TRAIN_TRANSFORM)
#         full_test = Core50(path, transform=SequentialCore50.TEST_TRANSFORM)

#         # Optionally limit samples
#         if self.MAX_SAMPLES_PER_TASK > 0:
#             print("Limiting samples to", self.MAX_SAMPLES_PER_TASK)
#             limit = min(self.MAX_SAMPLES_PER_TASK, len(full_train))
#             full_train = torch.utils.data.Subset(full_train, list(range(limit)))
#             full_test = torch.utils.data.Subset(full_test, list(range(limit)))

#         # 80/20 split
#         total = len(full_train)
#         train_len = int(0.8 * total)
#         test_len = total - train_len
#         train_ds, _ = random_split(full_train, [train_len, test_len], generator=torch.Generator().manual_seed(3407))
#         _, test_ds = random_split(full_test, [train_len, test_len], generator=torch.Generator().manual_seed(3407))

#                         # Build .data and .targets on Subset for store_masked_loaders
#         def get_paths_labels_from_subset(ds):
#             # Recover original ImageFolder.samples and map through Subset chains
#             indices = np.array(ds.indices)
#             base = ds.dataset
#             # Unwrap nested Subsets
#             while isinstance(base, Subset):
#                 indices = np.array(base.indices)[indices]
#                 base = base.dataset
#             # base is now Core50 with samples
#             paths = [base.samples[i][0] for i in indices]
#             labels = [base.samples[i][1] for i in indices]
#             return np.array(paths), np.array(labels)

#         train_paths, train_labels = get_paths_labels_from_subset(train_ds)
#         train_ds.data = train_paths
#         train_ds.targets = train_labels

#         test_paths, test_labels = get_paths_labels_from_subset(test_ds)
#         test_ds.data = test_paths
#         test_ds.targets = test_labels

#         # Wrap loaders via Mammoth
#         train_loader, test_loader = store_masked_loaders(train_ds, test_ds, self)
#         return train_loader, test_loader



import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, List
from argparse import Namespace
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50

from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from utils.conf import base_path

# 10 semantic categories in CORE50
CATEGORY_NAMES = [
    'plug_adapter',
'mobile_phone',
'scissor',
'light_bulb',
'can',
'glass',
'ball',
'marker',
'cup',
'remote_control'
]

class Core50(ImageFolder):
    """
    CORE50 ImageFolder wrapper that maps 50 instance‐IDs to 10 categories.
    Returns (img_transformed, category_label, raw_tensor).
    """
    RAW_SIZE = 128

    def __init__(self, root: str, transform=None):
        super().__init__(root, transform=transform)
        self.raw_transform = transforms.Compose([
            transforms.Resize((self.RAW_SIZE, self.RAW_SIZE), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor,int,torch.Tensor]:
        path, inst_label = self.samples[idx]
        img = self.loader(path)
        # compute category: instances obj01–obj05 → cat0, obj06–obj10 → cat1, etc.
        inst_name = os.path.basename(os.path.dirname(path))
        inst_id = int(inst_name.lstrip('obj')) - 1
        cat = inst_id // 5

        # raw unaugmented
        raw = self.raw_transform(img)
        # transformed
        img_tr = self.transform(img) if self.transform else raw

        return img_tr, cat, raw


class SequentialCore50(ContinualDataset):
    """
    Domain‐IL on CORE50 at category level (10 classes per session).
    Each of the 11 sessions is a separate task.
    """
    NAME = 'seq-core50'
    SETTING = 'domain-il'
    N_TASKS = 11
    N_CLASSES_PER_TASK = 10
    SIZE = (128, 128)
    INDIM = (3,128,128)
    MAX_SAMPLES_PER_TASK = 8000  # set >0 to truncate for fast debugging

    # data augmentation / normalization
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    TEST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    @set_default_from_args('batch_size')
    def get_batch_size(self) -> int:
        return 256

    @set_default_from_args('n_epochs')
    def get_epochs(self) -> int:
        return 50

    @staticmethod
    def get_backbone() -> torch.nn.Module:
        return "resnet50"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_transform():
        return None  # handled in Core50

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize([0.485,0.456,0.406], [0.229,0.224,0.225])

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        # prepare session folder/umbc/rs/gokhale/users/shivank2/shivanand/mammoth/data/CORE50/core50_128x128
        sess = f's{self.current_task + 1}'
        root = os.path.join(base_path()+"CORE50/core50_128x128", sess)

        # full instance‐level datasets, with category mapping in __getitem__
        full_train = Core50(root, transform=SequentialCore50.TRAIN_TRANSFORM)
        full_test  = Core50(root, transform=SequentialCore50.TEST_TRANSFORM)

        # optional truncation
        if self.MAX_SAMPLES_PER_TASK > 0:
            limit = min(self.MAX_SAMPLES_PER_TASK, len(full_train))
            full_train = Subset(full_train, list(range(limit)))
            full_test  = Subset(full_test,  list(range(limit)))

        # 80/20 split with fixed seed
        total = len(full_train)
        n_train = int(0.8 * total)
        n_test = total - n_train
        train_ds, _ = random_split(full_train, [n_train, n_test],
                                   generator=torch.Generator().manual_seed(3407))
        _, test_ds = random_split(full_test,  [n_train, n_test],
                                   generator=torch.Generator().manual_seed(3407))

        # now attach .data/.targets expected by Mammoth
        def attach_fields(subset):
            # unravel nested Subsets back to Core50
            idxs = np.array(subset.indices)
            base = subset.dataset
            while isinstance(base, Subset):
                idxs = np.array(base.indices)[idxs]
                base = base.dataset
            # now build arrays
            paths = [base.samples[i][0] for i in idxs]
            cats  = []
            for i in idxs:
                path, inst_lbl = base.samples[i]
                inst_id = int(os.path.basename(os.path.dirname(path)).lstrip('obj')) - 1
                cats.append(inst_id // 5)
            subset.data = np.array(paths)
            subset.targets = np.array(cats)

        attach_fields(train_ds)
        attach_fields(test_ds)

        # enforce simple class order [0..9]
        self.args.class_order = np.arange(self.N_CLASSES_PER_TASK)

        # wrap into Mammoth loaders
        train_loader, test_loader = store_masked_loaders(train_ds, test_ds, self)
        return train_loader, test_loader

    def get_class_names(self) -> List[str]:
        # return the 10 CORE50 category names
        return fix_class_names_order(CATEGORY_NAMES, self.args)


# Sanity check
if __name__ == '__main__':
    from argparse import Namespace
    ds = SequentialCore50(Namespace(batch_size=128, n_epochs=1, num_workers=4))
    tr, te = ds.get_data_loaders()
    x,y = next(iter(tr))
    print('Batch x:', x.shape, 'y:', y.unique())
    assert x.shape[2:] == (128, 128)
    assert y.max().item() < ds.N_CLASSES_PER_TASK
