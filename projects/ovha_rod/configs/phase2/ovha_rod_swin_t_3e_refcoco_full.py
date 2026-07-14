_base_ = '../ovha_rod_swin_t_5e_refcoco.py'
load_from = None
resume = False

model = dict(
    train_decoder_operator_only=True,
    decoder_operator_cfg=dict(
        enabled=True,
        enabled_operators=('qsro', 'tq_cato', 'ms_tleo'),
        router_hidden_dim=128,
        adapter_rank=16,
        qsro_query_chunk_size=128,
        use_router=True,
        use_memory=True,
        use_hyper_adapter=True,
        use_rceo=True))

optim_wrapper = dict(
    constructor='TrainableOnlyOptimWrapperConstructor',
    paramwise_cfg=dict(
        custom_keys={
            'decoder_operator': dict(lr_mult=1.0),
        }))

max_epochs = 3
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
        end=3,
        by_epoch=True,
        milestones=[2],
        gamma=0.1),
]
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=3, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        save_last=True))

# CUDA grid_sample backward in MS-TLEO is not bitwise deterministic. Keep the
# fixed seed, record this choice in run identity, and fail on non-finite values.
randomness = dict(seed=2026, deterministic=False)
custom_hooks = [
    dict(type='OperatorDiagnosticsHook', interval=50),
    dict(type='CheckpointProvenanceHook', identity_path=None),
]
