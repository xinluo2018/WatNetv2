96.7

### Experiment

#### Top accuracy (patch_size = 512, no pretrained)
| model             | miou   | oa    | update_time |
| ----------------- | ------ | ----- | ----------- |
| UNet              | 92.2%  | 97.3% | 2026.8.10   |
| watnet            | 92.9%  | 97.6% | 2026.8.14   |
| swin_unet         | 94.76% | 98.3% | 2026.8.14   |
| deeplabv3plus_mb2 | 93.3%  | 97.8% | 2026.8.14   |

#### Top accuracy (patch_size = 512, with pretrained backbone)
| model         | miou  | oa     | update_time | pretrained |
| ------------- | ----- | ------ | ----------- | ---------- |
| UNet+resnet50 | 95.3% | 98.5%  | 2026.8.14   | True       |
| UNet+swinv2   | 95.8% | 98.60% | 2026.8.15   | True       |
|               |       |        |             |            |
