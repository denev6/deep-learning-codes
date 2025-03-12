import torch


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, data_type, label_type):
        self.data = dataset["data"]
        self.labels = dataset["label"]
        self.data_type = data_type
        self.label_type = label_type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Default torch requires Float32 for training and Long for loss function.
        return self.data[idx].type(self.data_type), self.labels[idx].type(
            self.label_type
        )


class FlareDataset:
    def __init__(
        self,
        dataset: str | dict,
        class_names: list[str] | tuple[str],
        data_type: torch.dtype=torch.float32,
        label_type: torch.dtype=torch.long,
    ):
        """Dataset for torch dataset.

        :param dataset: path to the torch dataset contains keys "data" and "label or a dictionary itself."
        :param class_names: list of class names, e.g. ["Dog", "Cat"]
        """
        if isinstance(dataset, str):
            dataset = torch.load(dataset, weights_only=True)

        if isinstance(dataset, dict):
            keys = dataset.keys()
            assert "data" in keys and "label" in keys, "dataset must contains 'data' and 'label' keys."
        else:
            raise ValueError("dataset must be a dictionary or a path that points to a dictionary.")

        self.dataset = TorchDataset(dataset, data_type, label_type)
        self.class_names = class_names

    def dataset(self):
        return self.dataset

    def to_loader(self, batch_size, *args, **kwargs):
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, *args, **kwargs
        )

    def decode(self, label: int):
        return self.class_names[label]
