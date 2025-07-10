cd eval/webshop
# Set up webshop environment
./setup.sh -d all

# Set up data for SFT
cd ../../
mkdir -p data/train/webshop
mkdir -p data/train/alfworld
mkdir -p data/train/sciworld

mkdir -p data/train/webshop/explore
mkdir -p data/train/webshop/guided_explore

mkdir -p data/train/alfworld/explore
mkdir -p data/train/alfworld/guided_explore

mkdir -p data/train/sciworld/explore
mkdir -p data/train/sciworld/guided_explore