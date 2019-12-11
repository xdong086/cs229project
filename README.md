# cs229project

To prepare for the dateset run
python sunrgbd_data.py --gen_v1_data
python sunrgbd_data.py --gen_v1_2dbbox
python sunrgbd_data.py --gen_coco

To train 3D detector run
python train.py --dataset sunrgbd --log_dir log_sunrgbd --model votenet_gt_box2d --use_bbox2d

To evaluate 3D detector run
python eval.py --use_color --dataset sunrgbd --checkpoint_path log_sunrgbd/checkpoint.tar --dump_dir eval_sunrgbd --model votenet --num_point 40000 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

To train 2d Image detector run
python tools/train_net_sunrgbd.py  --config-file datasets/sunrgbd/sunrgbd.yaml

To visualize the result run
python viz.py
