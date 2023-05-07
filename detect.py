from models import Classification
from argparse import Namespace
if __name__ == '__main__':
    args = Namespace(
        num_epochs=50,
        batch_size=64,
        seed=0,
        image_size=224,
        root_path='COVID-19_Radiography_Dataset',
        model_args=dict(
            model_name='mobilenetv3_rw',
            pretrained=False,
            num_classes=2
        ),
        optimizer='AdamW',
        lr=1e-3,
        workers=2,
        exp_name='COVID',
        devices=1,
        precision='32',
        ckpt_path=None,
        use_wandb=True,
        wandb_id=None,
        project_name='COVID',
    )
    cls = Classification.load_from_checkpoint(checkpoint_path='COVID/ckpts/epoch=28-val_accuracy=96.88.ckpt', args = args)
    cls.test()