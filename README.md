# Note-Segmentation-SSL

Training code for our work [VOCANO: A note transcription framework for singing voice in polyphonic music][VOCANO: A note transcription framework for singing voice in polyphonic music]. For inference-only needs, please check the [VOCANO repository][VOCANO: A note transcription framework for singing voice in polyphonic music] or [Omnizart][Omnizart] which also includes other inference options including pitched instruments, vocal, chords, drum events, and beat.

## Requirements

Our training script is performed under Python3 and CUDA11.1, under PyTorch framework.

```bash
$ git clone https://github.com/B05901022/Note-Segmentation-SSL.git
$ cd Note-Segmentation-SSL
$ pip install -r requirements.txt
```

## Preparing Data

### Dataset Preparation

To run the full pipeline, datasets of certain category should include the files below:

* Training Datasets (TONAS, DALI)
	* `wav`: includes raw waveform in `.wav` format
	* `sdt`: includes binary form of `silent/duration/onset_not/onset/offset_not/offset` numpy arrays naming in `<data_name>_sdt.npy` format
* Unlabeled Datasets (MIR-1K, Medley-DB, DALI, DALI_demucs)
	* `wav`: includes raw waveform in `.wav` format
* Testing Datasets (ISMIR2014, DALI, DALI_demucs, CMedia_demucs)
	* `wav`: includes raw waveform in `.wav` format
	* `sdt`: includes binary form of `silent/duration/onset_not/onset/offset_not/offset` numpy arrays naming in `<data_name>_sdt.npy` format
	* `pitch`, `pitch_intervals`: pitch contour collected by [Patch-CNN][Patch-CNN] pipeline
	* `onoffset_intervals`: includes the onset time and offset time of every note.

The `sdt`, `pitch`, `pitch_intervals`, `onoffset_intervals` files can be downloaded from [here][Google Drive Link]. We also provided a simple script to download and construct the default folder hierarchy we used in our training script.

```bash
$ python file_prepare.py
```

Due to copyright issues, we cannot provide the audio files we used in our training procedure. However, all the datasets used publicly available, including [TONAS][TONAS], [MIR-1K][MIR-1K], [Medley-DB][Medley-DB], and [DALI][DALI]. Please follow the scripts provided by the original repositories to manually download the datasets and place the .wav files under `../data/<dataset_name>/wav/` folder or manually change the data folder in training/inference scripts (see next section). Note that for DALI dataset, we only pick the ground truth data that are selected by the DALI paper, so it is not necessary to download the whole dataset. All the data picked/downloaded in our work are listed in the `./meta/` folder. **Some of the data that we used may already not valid on the internet (CMedia and DALI), so please ensure to delete the labels in the `./meta/` folder that are not downloadable to correctly run through the training/inference scripts.**

### Vocal Seperation and Unlabeled Data Preprocessing (for speed up)

To perform the vocal separation pipeline, please follow the instructions in [DEMUCS repo][DEMUCS repo] and put the generated `.wav` files in `./data/<dataset_name>_demucs/wav/` folder. Other vocal separation toolkits are also available, just to ensure that the files are named the same as original `.wav` files and placed under the same `./data/<dataset_name>_demucs/wav/` folder.

To further speed up the unlabeled data loading speed for semi-supervised learning, it is also recommended to manually split the data into 8-second segments, and name the files into `<data_name>_<segment_number>.wav`.

### Folder Hierarchy

The dataset hierarchy should be like graph below. Training datasets should include `wav` and `sdt` folders; unlabeled datasets (for semi-supervised learning) should include `wav` folder; testing datasets shuold include `wav`, `sdt`, `pitch`, `pitch_intervals`, `onoffset_intervals` folders.

```
data/
└──TONAS/
	└──wav/
	└──sdt/
└──DALI_train/
	└──wav/
	└──sdt/
	└──pitch/
	└──pitch_intervals/
	└──onoffset_intervals/
└──MIR_1K/
	└──wav/
└──Medley_DB/
	└──wav/
└──ISMIR2014/
	└──wav/
	└──sdt/
	└──pitch/
	└──pitch_intervals/
	└──onoffset_intervals/
└──DALI_test/
	└──wav/
	└──sdt/
	└──pitch/
	└──pitch_intervals/
	└──onoffset_intervals/
└──CMedia/
	└──wav/
	└──sdt/
	└──pitch/
	└──pitch_intervals/
	└──onoffset_intervals/
└──DALI_demucs_test/ # Optional
	└──wav/
	└──sdt/
	└──pitch/
	└──pitch_intervals/
	└──onoffset_intervals/
└──CMedia_demucs/ # Optional
	└──wav/
	└──sdt/
	└──pitch/
	└──pitch_intervals/
	└──onoffset_intervals/
...
Note-Segmentation-SSL/
```

## Quick Start

### Training

To train from scratch, please modify `./script/train.sh` to fit your need. Logging is also valid through [WandB][WandB], which real-time tracking is valid from your WandB account.

* Parameters

```
--model_type: Which model to be used. Options: "PyramidNet_ShakeDrop"(default), "Resnet_18".
--loss_type: Pure supervised learning or semi-supervised learning with VAT. Options: "VAT"(default), "None"
--dataset1: Training dataset. Options: "TONAS"(default), "DALI_train", "DALI_orig_train", "DALI_demucs_train"
--dataset2: Unlabeled dataset for semi-supervised learning. Options: "MIR_1K"(default), "DALI_train", "DALI_demucs_train", "DALI_demucs_train_segment", "MedleyDB", "MedleyDB_segment", "None"
--dataset4: Validation dataset (only available for DALI). Options: "None"(default), "DALI_valid", "DALI_orig_valid", "DALI_demucs_valid"
--dataset5: Testing dataset. Options: "ISMIR2014"(default, "DALI_test", "DALI_orig_test", "DALI_demucs_test", "CMedia", "CMedia_demucs"
--data_path: The installation path of training dataset. Default: "../data/"
--exp_name: The experiment name that will be tracked on WandB.
--project_name: The experiment series name that will be tracked on WandB.
--entity: Your WandB account name.
--amp_level: Mixed-precision level. Default: "O1"
```

After modifying the training script, simply run the command to execute the training process.

```bash
$ bash script/train.sh
```

### Testing

After training, it is also available to modify the `--dataset5` and `--checkpoint_name` arguments in `./script/test.sh` to further test on other datasets. Run the command to execute the testing process.

```bash
$ bash script/test.sh
```

## Citation

If you find our work useful, please consider citing our paper.

* VOCANO
```
@inproceedings{vocano,
	title={{VOCANO}: A Note Transcription Framework For Singing Voice In Polyphonic Music},
	author={Hsu, Jui-Yang and Su, Li},
	booktitle={Proc. International Society of Music Information Retrieval Conference (ISMIR)},
	year={2021}
}
``` 

* Omnizart
```
@article{wu2021omnizart,
	title={Omnizart: A General Toolbox for Automatic Music Transcription},
	author={Wu, Yu-Te and Luo, Yin-Jyun and Chen, Tsung-Ping and Wei, I-Chieh and Hsu, Jui-Yang and Chuang, Yi-Chin and Su, Li},
	journal={arXiv preprint arXiv:2106.00497},
	year={2021}
}
```

[VOCANO: A note transcription framework for singing voice in polyphonic music]: https://github.com/B05901022/VOCANO
[Omnizart]: https://github.com/Music-and-Culture-Technology-Lab/omnizart
[TONAS]: https://zenodo.org/record/1290722#.YQRV344zaUk
[MIR-1K]: https://sites.google.com/site/unvoicedsoundseparation/mir-1k
[Medley-DB]: https://medleydb.weebly.com/
[DALI]: https://github.com/gabolsgabs/DALI
[WandB]: https://wandb.ai/site
[Patch-CNN]: https://github.com/leo-so/VocalMelodyExtPatchCNN
[DEMUCS repo]: https://github.com/facebookresearch/demucs
[Google Drive Link]: https://drive.google.com/file/d/1UPcvK1favpIoiYaL8qNqgbNbI3jn3lm3/view?usp=sharing