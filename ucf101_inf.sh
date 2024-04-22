NGPUs=$1

python src/scripts/calc_metrics_for_dataset.py \
--real_data_path /home/v-shuhuairen/mycontainer/ckpt/magvit2_a100/ucf101_bsz256_lr0.0001_steps100.0K_codebook1_128_5_64.0k_fsq_888555/inf_8fps/gt \
--fake_data_path /home/v-shuhuairen/mycontainer/ckpt/magvit2_a100/ucf101_bsz256_lr0.0001_steps100.0K_codebook1_128_5_64.0k_fsq_888555/inf_8fps/recon \
--mirror 1 --gpus ${NGPUs} --resolution 128 \
--metrics fvd10000_16f \
--verbose 1 --use_cache 1


#--fake_data_path /home/v-shuhuairen/mycontainer/ckpt/torchscale/ucf101_bsz256_steps100000_frm17_small_model_magvit2_only_vloss_no_bos_middle_clip/inf_topk900_topp0.95_cfg0/pred \