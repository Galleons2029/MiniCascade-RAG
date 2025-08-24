# -*- coding: utf-8 -*-
# @Time   : 2025/8/4 13:55
# @Author : Galleons
# @File   : dependency.py

"""
依赖配置脚本
"""
from langfuse import Langfuse
from dotenv import load_dotenv
import os
load_dotenv()


langfuse = Langfuse(
  secret_key=os.getenv('SECRET_KEY'),
  public_key=os.getenv('PUBLIC_KEY'),
  host="http://localhost:3000"
)