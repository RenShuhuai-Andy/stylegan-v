NGPUs=$1

python src/scripts/calc_metrics_for_dataset.py \
--real_data_path /home/v-shuhuairen/fvd/magvit2_a100/ucf101_bsz256_lr0.0001_steps100.0K_codebook1_128_5_64.0k_fsq_888555/inf_8fps/gt.zip \
--fake_data_path /home/v-shuhuairen/fvd/torchscale/ucf101_bsz256_steps100000_frm17_xl_model_magvit2_only_vloss_no_bos_multi_clip_stride3/inf_topk900_topp0.95_cfg0_steps80.0K/pred.zip \
--mirror 1 --gpus ${NGPUs} --resolution 128 \
--metrics fvd10000_16f \
--verbose 1 --use_cache 1

#--fake_data_path /home/v-shuhuairen/fvd/magvit2_a100/ucf101_bsz256_lr0.0001_steps100.0K_codebook1_128_5_64.0k_fsq_888555/inf_8fps/recon.zip \
#--fake_data_path /home/v-shuhuairen/fvd/torchscale/ucf101_bsz256_steps100000_frm17_small_model_magvit2_only_vloss_no_bos_middle_clip/inf_topk900_topp0.95_cfg0/pred.zip \
#--fake_data_path /home/v-shuhuairen/fvd/torchscale/ucf101_bsz256_steps100000_frm17_xl_model_magvit2_only_vloss_no_bos_multi_clip_stride3/inf_topk900_topp0.95_cfg0/pred.zip \
#--fake_data_path /home/v-shuhuairen/fvd/torchscale/ucf101_bsz256_steps100000_frm17_base_model_magvit2_only_vloss_no_bos_multi_clip_stride3/inf_topk900_topp0.95_cfg0/pred.zip \