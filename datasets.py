import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from configs import COVID_config
# def get_images_path(root_path, txt, type='COVID'):
#     images_path = []
#     targets = []
#     with open(txt, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             images_path.append(f'{root_path}/Images-processed/{type}/{line.strip()}')
#             if type == 'COVID':
#                 targets.append(1)
#             else:
#                 targets.append(0)
#     return images_path, targets
def get_images_path(root_path, mode = 'train'):
    images_path_list = []
    targets_list = []
    with open(f'{root_path}/{mode}.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            image_path, target = line.strip().split()
            images_path_list.append(image_path)
            targets_list.append(target)
    return images_path_list, targets_list
def get_index2label(root_path):
    with open(f'{root_path}/index2label.json', "r", encoding="utf-8") as f:
        index2index = json.load(f)
        label2index = {value: int(key) for key, value in index2index.items()}
        index2label = {value: key for key, value in label2index.items()}
    return index2label
class COVIDDataset(Dataset):
    def __init__(self, args, mode = 'train'):
        self.images_path_list, self.targets_list = get_images_path(args.root_path, mode)
        self.tranformers = T.Compose(
            [
                T.Resize((args.image_size, args.image_size)),
                T.ToTensor(),
                T.Normalize(0.5, 0.5)
            ]
        )
    def __len__(self):
        return len(self.images_path_list)
    def __getitem__(self, idx):
        image = Image.open(self.images_path_list[idx]).convert('RGB')
        return self.tranformers(image), torch.LongTensor([int(self.targets_list[idx])]), self.images_path_list[idx]
if __name__ == '__main__':
    dataset = COVIDDataset(COVID_config, 'train')
    print(dataset[0][0].shape)


