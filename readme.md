这是一个异常生成流水线，覆盖部件缺失、部件破损，部件凸起，部件凹陷、部件旋转、部件平移六种异常类型！
首先你需要安装blender：


破损异常运行指令：

```
blender --background --python render/render_broken.py -- --category Mug --num_views 10 --top_k 3 --samples 64 --seed 42
```

缺失异常运行指令：
```

```


旋转异常运行指令：
```
blender --background --python render/render_rotation_part.py -- --category Mug --num_views 10 --samples 64 --seed 42
```

也可以调整参数：
```
blender --background --python render/render_rotation.py -- --category Mug --num_views 10 --samples 64 --width 512 --height 512 --min_rotation_deg 5 --max_rotation_deg 20 --min_mask_ratio 0.05 --seed 42
```

