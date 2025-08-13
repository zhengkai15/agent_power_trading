from loguru import logger
import os

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, 'app.log')

# 移除默认的控制台输出，添加文件输出
logger.remove()
logger.add(log_file_path, rotation="10 MB", retention="7 days", level="INFO")
logger.add(os.sys.stderr, level="INFO") # 保持控制台输出，但只输出INFO级别及以上

log = logger
