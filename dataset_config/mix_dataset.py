# dataset settings
dataset_type = 'ConcatDataset'
data_root1 = '/18515601223/UnderwaterDiffusion/dataset/DUO/'
data_root2 = '/18515601223/UnderwaterDiffusion/dataset/UIIS/UDW/'
data_root3 = '/18515601223/UnderwaterDiffusion/dataset/AUDD/'
data_root4 = '/18515601223/UnderwaterDiffusion/dataset/DUT-USEG/'

img_norm_cfg = dict(
    mean = [81.236, 113.761, 117.095], std = [60.598, 58.471, 62.821], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=8),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

audd_data = dict(type='AuddDataset',
               ann_file=data_root3 + 'annotations/instances_train.json',
               img_prefix=data_root3 + 'images/train/',
               pipeline=train_pipeline)

dutuseg_data = dict(type='DUTUsegDataset',
               ann_file=data_root4 + 'Annotations/instances_annotations.json',
               img_prefix=data_root4 + 'JPEGImages/',
               pipeline=train_pipeline)

duo_data = dict(type='DUODataset',
               ann_file=data_root1 + 'annotations/instances_train.json',
               img_prefix=data_root1 + 'images/train/',
               pipeline=train_pipeline)

uiis_data = dict(type='UiisDataset',
                ann_file=data_root2 + 'annotations/train.json',
                img_prefix=data_root2 + 'train/',
                pipeline=train_pipeline)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        datasets=[duo_data, uiis_data, dutuseg_data, audd_data],
        pipeline=train_pipeline),
    val=dict(
        type='DUODataset',
        ann_file=data_root1 + 'annotations/instances_test.json',
        img_prefix=data_root1 + 'images/test/',
        pipeline=test_pipeline),
    test=dict(
        type='DUODataset',
        ann_file=data_root1 + 'annotations/instances_test.json',
        img_prefix=data_root1 + 'images/test/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
