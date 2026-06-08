这是一个异常生成流水线，覆盖部件缺失、部件破损，部件凸起，部件凹陷、部件旋转、部件平移六种异常类型！
首先你需要安装blender：


运行指令：

```
blender --background --python render/render_broken.py -- --category Mug --num_views 10 --top_k 3 --samples 64 --seed 42
```