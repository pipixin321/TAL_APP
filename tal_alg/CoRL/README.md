<<<<<<< HEAD
# train 
## THUMOS14
python main.py --hyp thu

## ActivityNet13
python main_b.py --hyp act


# test
python main.py --mode test --checkpoint 


# tools
1.generate_point_label.py
generate point label from instance label
-input:{input_root}/gt_full.json
-output:point_labels/point_labels_{method}.csv
-args:method theta gamma of gaussian distribution

2.point_label_distribution_analysis.py
plot distibution of point label
-input:point_label_xxx.csv
-output:dist.png

3.model_out_analysis.py
analysis output of model
args: dataset('act'/'thu') subset('train'/'test')
args: match_gt plot_bkg plot_video_score plot_thresh
-input:run_infos(list)
-output:{save_path}/vid_xxxxxxx.png

generate outputs using test.py before above analysis

=======
# CoRL
>>>>>>> main
