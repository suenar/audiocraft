# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Grid search file for finetuning an EnCodec model on custom music data.

This grid demonstrates how to finetune a pretrained EnCodec model at 32 kHz
on your own music dataset. The model can be trained from scratch or initialized
from a pretrained checkpoint.

Usage:
    # Finetune from pretrained MusicGen EnCodec checkpoint
    dora grid compression.encodec_finetune_music

    # Train from scratch (not recommended unless you have a large dataset)
    dora grid compression.encodec_finetune_music --clear

See ENCODEC_FINETUNING.md for detailed instructions.
"""

from ._explorers import CompressionExplorer
from ...environment import AudioCraftEnvironment


@CompressionExplorer
def explorer(launcher):
    # Configure Slurm partitions (adjust for your cluster setup)
    # For local training, comment out the next line
    # partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    # launcher.slurm_(gpus=8, partition=partitions)
    
    # For local training with specific number of GPUs (e.g., 1 GPU)
    launcher.bind_(device='cuda')
    
    # Use configuration for MusicGen's EnCodec model trained on monophonic audio sampled at 32 kHz
    # This uses a total stride of 640 leading to a frame rate of 50 Hz
    launcher.bind_(solver='compression/encodec_musicgen_32khz')
    
    # Use your custom music dataset
    launcher.bind_(dset='audio/music_finetune')
    
    # OPTION 1: Finetune from pretrained MusicGen EnCodec checkpoint (RECOMMENDED)
    # This will initialize the model with weights from the pretrained model
    launcher({
        'continue_from': '//pretrained/facebook/encodec_32khz',
        'label': 'finetune_from_pretrained'
    })
    
    # OPTION 2: Train from scratch (uncomment if you want to train from scratch)
    # Only recommended if you have a very large dataset (100k+ hours)
    # launcher({
    #     'label': 'train_from_scratch'
    # })
    
    # OPTION 3: Finetune with modified hyperparameters
    # Example: reduce batch size for limited GPU memory
    # launcher({
    #     'continue_from': '//pretrained/facebook/encodec_32khz',
    #     'dataset.batch_size': 32,  # Reduced from default 64
    #     'optim.lr': 1e-4,  # Lower learning rate for finetuning
    #     'label': 'finetune_small_batch'
    # })
    
    # OPTION 4: Finetune with different model architecture
    # Example: use a smaller model for faster training
    # launcher({
    #     'continue_from': '//pretrained/facebook/encodec_24khz',
    #     'sample_rate': 24000,
    #     'label': 'finetune_24khz'
    # })
