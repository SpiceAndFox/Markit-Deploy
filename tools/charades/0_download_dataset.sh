# 下载tag
git clone https://www.modelscope.cn/datasets/OmniData/Charades-STA.git

# 下载视频
export HF_ENDPOINT=https://hf-mirror.com
wget https://hf-mirror.com/datasets/Pai3dot14/Charades_v1_hf/resolve/main/Charades_v1_seg/Charades_v1.zip.part0
wget https://hf-mirror.com/datasets/Pai3dot14/Charades_v1_hf/resolve/main/Charades_v1_seg/Charades_v1.zip.part1
cat Charades_v1.zip.part0 Charades_v1.zip.part1 > Charades_v1.zip
unzip Charades_v1.zip