export PIP_INDEX_URL="http://mirrors.h.pjlab.org.cn/pypi/simple/"
export PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export PIP_TRUSTED_HOST="mirrors.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"
export PIP_NO_INDEX="false" # 如果要完全禁用公网访问，改为 "true"
cd /mnt/shared-storage-user/leihaodi/imo/SpecForge
pip install -e .
sleep 3
pip uninstall flashinfer_jit_cache -y
cd /mnt/shared-storage-user/leihaodi/imo/flashinfer_pkgs
python -m pip install flashinfer_jit_cache-0.6.3+cu129-cp39-abi3-manylinux_2_28_x86_64.whl
cd /mnt/shared-storage-user/leihaodi/imo/SpecForge