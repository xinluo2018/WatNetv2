
### Experiment

### baseline: 
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


### vmamba segmentation:     
| model         | miou   | oa      | update_time | pretrained |  note          |
| ------------- | -----  | ------  | ----------- | ---------- | --------       |
| mamba_seg     | 91.75% | 97.21%  | 2026.9.17   | no         |very tiny model |
| mamba_seg     | 94.82% | 98.30%  | 2026.10.9   | no         |  tiny model    |
| mamba_seg     | 95.91% | 98.67%  | 2026.10.12   | yes        |  tiny model    |
| mamba_seg     | 96.07% | 98.72%  | 2026.10.12   | yes        |  tiny model    |



