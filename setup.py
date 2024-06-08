from setuptools import setup, find_packages

VERSION = '0.0.2' 
DESCRIPTION = 'My first Python package'
LONG_DESCRIPTION = 'My first Python package with a slightly longer description'

# 配置
setup(
       # 名称必须匹配文件名 'verysimplemodule'
        name="xgdl", 
        version=VERSION,
        author="Jiajun Zhu",
        author_email="<zhuconv@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # 需要和你的包一起安装，例如：'caer'
        
        keywords=['geometric deep learning', 'explainable AI', 'graph neural network', 'AI for science'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)