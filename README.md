# Tantic

毕设~
项目背景：https://larkcommunity.feishu.cn/wiki/P2Tnwd1X6iPkmQk8kr1cklymngk

![image-1743238512369](https://kauizhaotan.oss-accelerate.aliyuncs.com/blog/image-1743238512369.png?x-oss-process=style/water)

## 环境依赖

### 2.1 TLS 插件安装

由于 scapy 官方库暂时不支持 tls 解析，因此需要安装插件，可以通过如下命令下载源库：

```bash
pip install scapy==2.4.3    # 下载scapy

git clone https://github.com/kalidasya/scapy-ssl_tls.git
cd scapy-ssl_tls
git fetch
git checkout -b py3-suite remotes/origin/py3_update 
```
1、参考 [手动安装](https://github.com/kalidasya/scapy-ssl_tls/blob/py3_update/README.md#option-3-manual-installation) 方法对依赖库进行安装。
```bash
python -c "import scapy; print scapy.__file__" # 查看scapy安装包环境
cp scapy-ssl_tls/* <scapy_dir>/layers/
code <scapy_dir>/config.py
```

2、 验证配置结果：进入python交互页面，或使用 ipython。
```bash
In [1]: from scapy.all import *
In [2]: TLS
Out[2]: scapy.layers.ssl_tls.SSL
```