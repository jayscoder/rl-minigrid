# RL-Minigrid

场景：https://minigrid.farama.org/environments/minigrid/DoorKeyEnv/

行为树脚本放到：scripts/目录下

- run_bt.py 运行纯粹的行为树
- run_rl.py 运行纯粹的强化学习
- run_rlbt.py 运行强化学习+行为树

```shell
parser.add_argument('--train', action='store_true') # 是否开启训练
parser.add_argument('--render', action='store_true') # 是否开启渲染
parser.add_argument('--track', action='store_true') # 是否开启pybts监控
```

```shell
python run_rlbt.py --train --render --track
```

- render_all_bt.py 将所有scripts的行为树图片生成到scripts/images中

```shell
python render_all_bt.py
```

## 强化学习节点

- RLSwitcher
- RLSelector
- RLSequence
- RLCondition
- RLAction

## 奖励节点

会将奖励放到对应的奖励域中

```shell
<Reward domain="target" reward="1"/>
```



1个


1红色门，1蓝色门
1红色钥匙，1蓝色钥匙

Selector/Sequence/Switcher
> Selector：备选项
> Sequence： 预设动作序列
> Switcher: 自己学习动作序列


