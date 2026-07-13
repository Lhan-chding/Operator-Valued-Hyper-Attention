_base_ = '../ovha_rod_swin_t_5e_refcoco.py'

model = dict(
    decoder_operator_cfg=dict(
        enabled=True,
        enabled_operators=('qsro',),
        router_hidden_dim=128,
        adapter_rank=16,
        qsro_cfg=dict(query_chunk_size=128),
        tq_cato_cfg=dict(temperature=1.0),
        ms_tleo_cfg=dict(context_scale=1.5),
        rceo_cfg=dict(prior_cap=2.0),
        use_router=True,
        use_memory=True,
        use_hyper_adapter=True,
        use_rceo=True))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'decoder_operator': dict(lr_mult=1.0),
        }))
