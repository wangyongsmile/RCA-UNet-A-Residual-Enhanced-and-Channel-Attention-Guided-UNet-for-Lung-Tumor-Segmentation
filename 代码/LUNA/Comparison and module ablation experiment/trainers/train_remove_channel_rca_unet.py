import torch
from data import LUNADataLoader, SliceExtractor, EnhancedPreprocessor, AnalyzedLUNADataset, split_dataset
from models import RCAUNetWithoutChannelAttention
from trainers.base_trainer import train_model


def train_remove_channel_rca_unet():
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")

    data_loader = LUNADataLoader()
    slice_extractor = SliceExtractor(margin=1)
    preprocessor = EnhancedPreprocessor(target_size=(256, 256))

    train_series, val_series, test_series = split_dataset(data_loader.all_seriesuids)

    train_dataset = AnalyzedLUNADataset(train_series, data_loader, slice_extractor, preprocessor, is_train=True)
    val_dataset = AnalyzedLUNADataset(val_series, data_loader, slice_extractor, preprocessor, is_train=False)
    test_dataset = AnalyzedLUNADataset(test_series, data_loader, slice_extractor, preprocessor, is_train=False)

    model = RCAUNetWithoutChannelAttention(in_channels=1, out_channels=1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return train_model(model, train_dataset, val_dataset, test_dataset, "RCAUNet-无通道注意力", "rca_unet_no_attention")