export

 CUDA_VISIBLE_DEVICES="0"
export

 PYTHONPATH="."
python functions/main.py --dataset=seq-miniimg --model=icarl_lipschitz --buffer_size=2000 --lr=0.1 --pre_epochs=200 --datasetS=tinyimgR --wd_reg=0 --method=localrobustness > output.txt
