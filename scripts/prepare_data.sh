export HF_HOME="/mnt/shared-storage-gpfs2/p1-shared-2/leihaodi/data/dflash-data"
mkdir -p $HF_HOME
cd /mnt/shared-storage-user/leihaodi/imo/SpecForge
python scripts/prepare_data.py --dataset ultrachat
python scripts/prepare_data.py --dataset sharegpt
