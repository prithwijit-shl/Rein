topsalt_type = "LIPDataset"
topsalt_root = "/lus231/ua/export/saltcrawler/uspcjc/dataset_split/"
test_root = "/glb/hou/pt.sgs/data/ml_ai_us/uspcjc/dataset_new/amendment/"
topsalt_crop_size = (512, 512)
topsalt_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1024, 512)),
    dict(type="RandomCrop", crop_size=topsalt_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
topsalt_val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
topsalt_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_topsalt = dict(
    type=topsalt_type,
    data_root=topsalt_root,
    data_prefix=dict(
        img_path="images/train",
        seg_map_path="labels/train",
    ),
    img_suffix=".jpg",
    seg_map_suffix=".png",
    pipeline=topsalt_train_pipeline,
)
val_topsalt = dict(
    type=topsalt_type,
    data_root=topsalt_root,
    data_prefix=dict(
        img_path="images/val",
        seg_map_path="labels/val",
    ),
    img_suffix=".jpg",
    seg_map_suffix=".png",
    pipeline=topsalt_val_pipeline,
)

test_topsalt = dict(
    type=topsalt_type,
    data_root=test_root,
    data_prefix=dict(
        img_path="images",
        seg_map_path="labels",
    ),
    img_suffix=".jpg",
    seg_map_suffix=".png",
    pipeline=topsalt_test_pipeline,
)

train_dataloader = dict(
batch_size=4,
num_workers=1,
persistent_workers=True,
pin_memory=True,
sampler=dict(type="InfiniteSampler", shuffle=True),
dataset=train_topsalt,
)

val_dataloader = dict(
batch_size=1,
num_workers=1,
persistent_workers=True,
sampler=dict(type="DefaultSampler", shuffle=False),
dataset=val_topsalt,
)

test_dataloader = dict(
batch_size=1,
num_workers=1,
persistent_workers=True,
sampler=dict(type="DefaultSampler", shuffle=False),
dataset=test_topsalt,
)

val_evaluator = dict(
type="DGIoUMetric",
iou_metrics=["mIoU"],
output_dir="work_dirs/eval2",
)

test_evaluator = dict(
type="DGIoUMetric",
iou_metrics=["mIoU"],
format_only=True,
output_dir="work_dirs/eval_test2",
)
