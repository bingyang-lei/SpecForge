export HF_HOME="/mnt/shared-storage-gpfs2/p1-shared-2/leihaodi/data/dflash-data"
mkdir -p $HF_HOME
cd /mnt/shared-storage-user/leihaodi/imo/SpecForge
<<<<<<< HEAD
python scripts/prepare_data.py --dataset ultrachat
python scripts/prepare_data.py --dataset sharegpt
=======
python scripts/prepare_data.py --dataset codealpaca-20k
# python scripts/prepare_data.py --dataset sharegpt
>>>>>>> a130d7b (temp: collect dflash local changes)
