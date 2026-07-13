_base_ = 'mmdet::mm_grounding_dino/refcoco/grounding_dino_swin-t_finetune_8xb4_5e_refcoco.py'
load_from = None
model = dict(
    type='DeterministicGroundingDINO',
    backbone=dict(init_cfg=None))
custom_imports = dict(imports=['ovha_rod'], allow_failed_imports=False)
custom_hooks = [dict(type='OperatorDiagnosticsHook', interval=50)]

data_root = 'data/coco/'
val_ann_file = 'mdetr_annotations/finetune_refcoco_val.json'
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
    type='RefExpMetric',
    ann_file=data_root + val_ann_file,
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))
test_dataloader = val_dataloader
test_evaluator = val_evaluator
max_epochs = 5
