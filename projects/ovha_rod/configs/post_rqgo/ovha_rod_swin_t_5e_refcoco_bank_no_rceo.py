_base_ = '../ovha_rod_swin_t_5e_refcoco.py'

model = dict(
    decoder_operator_cfg=dict(
        enabled=True,
        enabled_operators=('qsro', 'tq_cato', 'ms_tleo'),
        router_hidden_dim=128,
        adapter_rank=16,
        use_router=True,
        use_memory=True,
        use_hyper_adapter=True,
        use_rceo=False))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'decoder_operator': dict(lr_mult=1.0),
        }))
