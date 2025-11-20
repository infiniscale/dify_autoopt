import yaml
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from auth.api import DifyAuthClient

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def login_job():
    """定时登录任务"""
    logger.info("开始执行定时登录任务...")
    try:
        result = client.login()
        if result:
            with open("access_token.txt", "w", encoding="utf-8") as f:
                f.write(result["access_token"])

            logger.info("登录成功")
        else:
            logger.error("登录失败")
    except Exception as e:
        logger.error(f"登录任务执行出错: {e}")

# 读取yaml文件
config_path = "config/config.yaml"
try:
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    logger.info(f"配置文件加载成功: {config.get('dify', {}).get('base_url', 'Unknown')}")
except Exception as e:
    logger.error(f"配置文件加载失败: {e}")
    exit(1)

# 使用配置文件中的凭据初始化客户端
client = DifyAuthClient(
    base_url=config['dify']['base_url'],
    email=config["auth"]['username'],
    password=config["auth"]['password']
)
logger.info("正在启动后台调度器...")
scheduler = BackgroundScheduler()
scheduler.add_job(login_job, 'interval', hours=1)
scheduler.start()
logger.info("调度器已启动...")


# 操作dify
