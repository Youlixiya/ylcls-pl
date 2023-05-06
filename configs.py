from argparse import Namespace
COVID_config = Namespace(
    num_epochs=1,
    batch_size=64,
    seed=0,
    image_size=224,
    root_path='COVID-19_Radiography_Dataset',
    model_args = dict(
        model_name = 'resnet50',
        pretrained=False,
        num_classes=2
    ),
    optimizer='AdamW',
    lr=1e-3,
    workers=0,
    exp_name='COVID',
    devices=1,
    precision='32',
    ckpt_path=None,
    use_wandb=False,
    wandb_id=None,
    project_name='COVID',

)