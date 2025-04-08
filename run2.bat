@echo off
title WD14-Batch
color 0a

cd /d %cd%
:: cd /d %~dp0

set DIR=%cd%

:: https://www.python.org/ftp/python/
:: https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip
:: https://bootstrap.pypa.io/get-pip.py

set PATH=%DIR%\python;%DIR%\python\Scripts;%PATH%;
:: set PATH=%DIR%\git\bin;%DIR%\python;%DIR%\python\Scripts;%PATH%;
set PY_LIBS=%DIR%\python\Scripts\Lib;%DIR%\python\Scripts\Lib\site-packages
set PY_PIP=%DIR%\python\Scripts
set PIP_INSTALLER_LOCATION=%DIR%\python\get-pip.py

set HF_HOME=%DIR%\hf
:: set HF_ENDPOINT=https://hf-mirror.com
:: set HUGGINGFACE_HUB_DISABLE_CACHE=1

:: ��װ pip �󲻿�ʹ��ʱ���ɳ��Ա༭ %DIR%\python\pythonXXX._pth ȥ�� import site ��ע��

:: python �ű�����ǰĿ¼��ӵ� sys.path
::      import os
::      import sys
::      sys.path.append(os.path.dirname(os.path.abspath(__file__)))

:: ����ʱ����·��
set PIP_CACHE_DIR=..\cache

:: ���档off ���ã�on ����
:: set PIP_NO_CACHE_DIR=off

:: ������ URL
set PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
:: https://pypi.org/simple
:: https://mirrors.163.com/pypi/simple/
:: https://mirrors.cloud.tencent.com/pypi/simple
:: https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

:: python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
:: python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
:: python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: python -m pip install xformers===0.0.28.post1 --extra-index-url https://download.pytorch.org/whl/cu124

:: ��������� URL
:: set PIP_EXTRA_INDEX_URL=https://pypi.org/simple

:: �����������ʱʱ�䡣��λ���롣
set PIP_TIMEOUT=10

:: ����ϸ�ĵ�����Ϣ
:: set PIP_VERBOSE=1

:: python caption.py --character_tags_first --general_thresh 0.35 --character_thresh 0.85 --hide_rating_tags

cmd /k python caption2.py input --character_tags_first --general_thresh 0.25 --character_thresh 0.85 --hide_rating_tags --include_subdirectories --use_gpu
