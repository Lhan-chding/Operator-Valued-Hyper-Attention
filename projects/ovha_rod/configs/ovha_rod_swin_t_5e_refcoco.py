_base_ = 'mmdet::mm_grounding_dino/refcoco/grounding_dino_swin-t_finetune_8xb4_5e_refcoco.py'
load_from = None

custom_imports = dict(imports=['ovha_rod'], allow_failed_imports=False)

data_root = 'data/coco/'
val_ann_file = 'mdetr_annotations/finetune_refcoco_val.json'
model = dict(
    type='OVHAGroundingDINO',
    backbone=dict(init_cfg=None),
    seed_operator='rqgo',
    seed_operator_cfg=dict(
        seed_bias_cap=2.0, relation_scales=(0.05, 0.15, 0.30)),
    role_encoder_cfg=dict(role_count=4, num_heads=8),
    bbox_head=dict(
        type='OVHAGroundingDINOHead',
        loss_seed_weight=0.5,
        loss_ref_weight=0.5,
        loss_role_div_weight=0.005,
        seed_target_gamma=1.0))

val_dataset = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=val_ann_file,
    data_prefix=dict(img='train2014/'),
    test_mode=True,
    return_classes=True,
    pipeline=_base_.test_pipeline,
    backend_args=None)
val_dataloader = dict(dataset=dict(_delete_=True, **val_dataset))
val_evaluator = dict(
    _delete_=True,
    type='OVHARefExpMetric',
    ann_file=data_root + val_ann_file,
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))
test_dataloader = val_dataloader
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-4),
    clip_grad=dict(
        max_norm=0.1, norm_type=2, error_if_nonfinite=True),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'backbone': dict(lr_mult=0.05),
            'language_model': dict(lr_mult=0.025),
            'encoder': dict(lr_mult=0.25),
            'decoder': dict(lr_mult=0.25),
            'bbox_head': dict(lr_mult=0.25),
            'bbox_head.referent_head': dict(lr_mult=1.0),
            'role_encoder': dict(lr_mult=1.0),
            'seed_operator': dict(lr_mult=1.0),
        }))

max_epochs = 5
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[3],
        gamma=0.1),
]
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=2026, deterministic=True)
custom_hooks = [
    dict(type='SeedLossWarmupHook', warmup_iters=500),
    dict(type='OperatorDiagnosticsHook', interval=50),
    dict(type='CheckpointProvenanceHook', identity_path=None),
]
