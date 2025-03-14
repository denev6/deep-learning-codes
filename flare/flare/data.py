import torch


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, data_type, label_type):
        self.data = data
        self.labels = label
        self.data_type = data_type
        self.label_type = label_type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Default torch requires Float32 for training and Long for loss function.
        return self.data[idx].type(self.data_type), self.labels[idx].type(
            self.label_type
        )

    def subset(self, indices):
        subset_data = self.data[indices]
        subset_labels = self.labels[indices]
        return TorchDataset(
            subset_data,
            subset_labels,
            self.data_type,
            self.label_type,
        )


class FlareData:
    def __init__(
        self,
        class_names: list[str] | tuple[str],
        batch_size: int,
        data_type: torch.dtype = torch.float32,
        label_type: torch.dtype = torch.long,
        *args,
        **kwargs
    ):
        """Dataset for torch dataset.

        Args:
            dataset: path, dict{'data', 'label'} or torch.utils.data.Dataset is allowed.
            class_names: list of class names, e.g. ["Dog", "Cat"]
            batch_size: batch size for DataLoader
            data_type: target data type for torch.utils.data.Dataset, default: torch.float32
            label_type: target label type for torch.utils.data.Dataset, default: torch.long
            args: parameters for torch DataLoader
        """
        self.dataset = None
        self.class_names = class_names
        self.batch_size = batch_size
        self.data_type = data_type
        self.label_type = label_type
        self._args = args
        self._kwargs = kwargs

    def from_tensor(self, data, label):
        self.dataset = TorchDataset(data, label, self.data_type, self.label_type)

    def from_numpy(self, data, label):
        self.dataset = TorchDataset(
            torch.from_numpy(data),
            torch.from_numpy(label),
            self.data_type,
            self.label_type,
        )

    def from_dataset(self, dataset):
        self.dataset = dataset

    def to_loader(self):
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, *self._args, **self._kwargs
        )

    def decode(self, label: int):
        return self.class_names[label]
