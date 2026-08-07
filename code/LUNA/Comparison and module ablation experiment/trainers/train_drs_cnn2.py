import torch
from data import LUNADataLoader, SliceExtractor, EnhancedPreprocessor, AnalyzedLUNADataset, split_dataset
from models import DRS_CNN2
from trainers.base_trainer import train_model


def train_drs_cnn2():
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")

    data_loader = LUNADataLoader()
    slice_extractor = SliceExtractor(margin=1)
    preprocessor = EnhancedPreprocessor(target_size=(256, 256))

    train_series, val_series, test_series = split_dataset(data_loader.all_seriesuids)

    train_dataset = AnalyzedLUNADataset(train_series, data_loader, slice_extractor, preprocessor, is_train=True)
    val_dataset = AnalyzedLUNADataset(val_series, data_loader, slice_extractor, preprocessor, is_train=False)
    test_dataset = AnalyzedLUNADataset(test_series, data_loader, slice_extractor, preprocessor, is_train=False)

    model = DRS_CNN2(in_channels=1, out_channels=1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return train_model(model, train_dataset, val_dataset, test_dataset, "DRS-CNN2", "drs_cnn2")