from torch.utils.data import Dataset

class BrainMRIDataset(Dataset):
    """
    Dataset object
    """
    def __init__(self, series_dict, image_transform=None):
        self.series_dict = series_dict
        self.image_transform = image_transform

    def __len__(self):
        return len(self.series_dict)

    def __getitem__(self, idx):
        series_uids = list(self.series_dict.keys())

        image = self.series_dict[series_uids[idx]]
        if self.image_transform is not None:
            image = self.image_transform(image=image)["image"]

        return series_uids[idx], image
