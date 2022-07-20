# VSAIT: Unpaired Image Translation via Vector Symbolic Architectures

[Justin Theiss](https://www.linkedin.com/in/justin-d-theiss), [Jay Leverett](https://www.linkedin.com/in/jay-leverett), [Daeil Kim](https://www.linkedin.com/in/daeil), [Aayush Prakash](https://ca.linkedin.com/in/aayush-prakash-0798142b)<br>
In ECCV 2022 (Oral).

Source GTA5                        |  GTA5 Translated with VSAIT
:---------------------------------:|:-------------------------:
![Source GTA](./docs/imgs/gta.gif) | ![Translated GTA](./docs/imgs/vsait_gta2city.gif)

### Installation
Clone this repo:
```bash
git clone https://github.com/facebookresearch/vsait.git
cd vsait/
```

Install dependencies via pip:
```bash
pip install -r requirements.txt
```

### Dataset Preparation
For any two image datasets with png/jpg images, download source and target data (or create symlinks) to `./data/source/` and `./data/target/` with `train` and `val` subfolders for each domain.

For gta2cityscapes, [GTA5 dataset](https://download.visinf.tu-darmstadt.de/data/from_games/) `images` folder should be split into training and validation folders to be stored in `./data/source/train/` and `./data/source/val/`, respectively. Similarly, the [Cityscapes dataset](https://www.cityscapes-dataset.com/) folders `/leftImg8bit/train/` and `/leftImg8bit/val/` should be stored in `./data/target/train/` and `./data/target/val/`, respectively.

### Training
Launch training with defaults in configs:
```bash
python train.py --name="vsait"
```

This will use the default configs in `./configs/` and save checkpoints and translated images in `./checkpoints/vsait/`.

### Evaluation
Translate images in `./data/source/val/` using a specific checkpoint:
```bash
python test.py --name="vsait_adapt" --checkpoint="./checkpoints/vsait/version_0/checkpoints/epoch={i}-step={j}.ckpt"
```

Images from the above example would be saved in `./checkpoints/vsait_adapt/images/`.

## License
VSAIT is released under the [CC-BY-NC 4.0 License](LICENSE).
