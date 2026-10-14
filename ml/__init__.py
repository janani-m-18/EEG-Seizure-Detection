"""PyTorch EEG modelling, CHB-MIT preparation, and group-safe evaluation (Steps 9-14)."""

from ml.model import BatchBCNNBiLSTMClassifier, BatchBModelConfig, CNNBiLSTMClassifier, EEGCNNFeatureExtractor, TemporalBiLSTM
from ml.training import EEGSegmentDataset, TrainingConfig, train_model
from ml.chbmit import CHBMITConfig, CHBMITDatasetError, CHBMITRecording, CHBMITSegmentBuilder, RealEEGSegments, discover_chbmit_recordings
from ml.evaluation import GroupSplitConfig, classification_metrics, split_real_eeg, train_and_evaluate_real_data, train_real_data
from ml.augmentation import AugmentationConfig, AugmentedEEGData, augment_training_segments
from ml.batch_a import BATCH_A_AUGMENTATION, BATCH_A_PREPROCESSING, BATCH_A_SEGMENTATION, batch_a_pipeline
from ml.batch_b import (BATCH_B_AUGMENTATION, BATCH_B_MODEL_CONFIG, BATCH_B_MODEL_ID, BATCH_B_PREPROCESSING_ID,
                        BatchBCompatibilityError, BatchBTrainingConfig, build_batch_b_model, evaluate_batch_b,
                        load_batch_b_checkpoint, prepare_batch_b_development_data, run_batch_b_development, train_batch_b)
from ml.batch_b_inference import BatchBInferenceService
from ml.gradcam import GradCAMExplanation, GradCAMRegion, explain_batch_b_segment, final_cnn_conv_layer, final_cnn_feature_layer

__all__ = [
    "CNNBiLSTMClassifier",
    "BatchBCNNBiLSTMClassifier", "BatchBModelConfig",
    "EEGCNNFeatureExtractor",
    "EEGSegmentDataset",
    "TemporalBiLSTM",
    "TrainingConfig",
    "train_model",
    "CHBMITConfig", "CHBMITDatasetError", "CHBMITRecording", "CHBMITSegmentBuilder", "RealEEGSegments",
    "discover_chbmit_recordings", "GroupSplitConfig", "classification_metrics", "split_real_eeg",
    "train_and_evaluate_real_data", "train_real_data",
    "AugmentationConfig", "AugmentedEEGData", "augment_training_segments",
    "BATCH_A_AUGMENTATION", "BATCH_A_PREPROCESSING", "BATCH_A_SEGMENTATION", "batch_a_pipeline",
    "BATCH_B_AUGMENTATION", "BATCH_B_MODEL_CONFIG", "BATCH_B_MODEL_ID", "BATCH_B_PREPROCESSING_ID",
    "BatchBCompatibilityError", "BatchBTrainingConfig", "build_batch_b_model", "evaluate_batch_b",
    "load_batch_b_checkpoint", "prepare_batch_b_development_data", "run_batch_b_development", "train_batch_b",
    "BatchBInferenceService", "GradCAMExplanation", "GradCAMRegion", "explain_batch_b_segment", "final_cnn_conv_layer", "final_cnn_feature_layer",
]
