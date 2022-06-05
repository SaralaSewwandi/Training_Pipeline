## Set Up Commands
1.
git clone https://github.com/SaralaSewwandi/Training_Pipeline.git

2.
conda create -n project_env python=3.6

3.
cd Training_Pipeline/

4.
conda activate project_env

5.
pip install -r ./requirements.txt 

## Test Cases

python script.py --data_root ./image_datasets/jpg_images/

python script.py --data_root ./image_datasets/png_images/

python script.py --data_root ./image_datasets/gif_images/

python script.py --data_root ./image_datasets/bmp_images/

python script.py --data_root ./image_datasets/mixed_images/
