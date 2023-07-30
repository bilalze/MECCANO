from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from demo_net import demo
from test_net import test
from train_netd import train
from visualization import visualize
from argparse import Namespace
from slowfast.datasets.mecanno_videog import Meccano_videog 
import torch


args = Namespace(cfg_files=['/home/GC/babdulrahman/python_work/MECCANO/configs/action_recognition/SLOWFAST_8x8_R50_MECCANO.yaml'], init_method='tcp://localhost:9999', num_shards=1, opts=None, shard_id=0)
for path_to_config in args.cfg_files:
    cfg = load_config(args, path_to_config)
    cfg = assert_and_infer_cfg(cfg)
val_=Meccano_videog(cfg, mode="val")
val_loader=torch.utils.data.DataLoader(
    val_,
    batch_size=1,
    num_workers=1,
    pin_memory=True,)

for cur_iter, (inputs, labels, index, time) in enumerate(
        val_loader
    ):
    print(inputs)
    