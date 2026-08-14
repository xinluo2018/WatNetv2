96.7

### Experiment

#### Top accuracy (patch_size = 512)

| model             | miou  | oa    | update_time |
| ----------------- | ----- | ----- | ----------- |
| UNet              | 93.0% | 97.6% | 2026.8.10   |
| deeplabv3plus_mb2 | %     | %     | 2026.4.8    |
| watnet            | 90.8% | 96.7% |             |
| swin_unet         | %     | %     | 2026.5.4    |

#### Top accuracy (patch_size = 512, pretrained backbone)

| model                | miou  | oa    | update_time | pretrained |
| -------------------- | ----- | ----- | ----------- | ---------- |
| UNet+efficientnet_b0 | 94.7% | 97.8% | 20260727    | True       |
| UNet+swinv2          | %     | 0%    |             | True       |
|                      |       |       |             |            |
