@REM set CUDA_VISIBLE_DEVICES="0"
@REM set PYTHONPATH="."
python functions/main.py --dataset=seq-miniimg --model=icarl_lipschitz --buffer_size=50 --lr=0.1 --pre_epochs=200 --datasetS=tinyimgR --wd_reg=0.5 --loss_reg=0.8 --linear_reg=0