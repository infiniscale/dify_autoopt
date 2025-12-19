import os
import logging
from pathlib import Path
from typing import Optional

import requests
from requests.exceptions import RequestException, Timeout

from src.config.loaders.config_loader import FileSystemReader

logger = logging.getLogger(__name__)


class Token:
    """令牌管理器 - 处理访问令牌的读写"""

    def __init__(self, config_path: str = "config/env_config.yaml"):
        """
        初始化令牌管理器

        Args:
            config_path (str): 配置文件路径，默认为 "config/env_config.yaml"
        """
        try:
            yaml_data = FileSystemReader.read_yaml(Path(config_path))

            auth_config = yaml_data.get("auth")
            if not auth_config:
                raise KeyError("配置文件中缺少auth配置块")

            self.access_token_path = auth_config.get("access_token_path")
            if not self.access_token_path:
                raise ValueError("配置文件中缺少access_token_path配置项")

            dify_config = yaml_data.get("dify")
            if not dify_config:
                raise KeyError("配置文件中缺少dify配置块")

            self.base_url = dify_config.get("base_url")
            if not self.base_url:
                raise ValueError("配置文件中缺少base_url配置项")

            self.timeout = auth_config.get("token_validation_timeout", 10)

        except FileNotFoundError as e:
            logger.error(f"配置文件未找到: {e}")
            raise FileNotFoundError(f"无法找到配置文件: {e}")
        except KeyError as e:
            logger.error(f"配置项缺失: {e}")
            raise
        except Exception as e:
            logger.error(f"初始化令牌管理器失败: {e}")
            raise RuntimeError(f"令牌管理器初始化失败: {e}")

    def rewrite_access_token(self, access_token: str) -> bool:
        """
        重写访问令牌

        Args:
            access_token (str): 访问令牌

        Returns:
            bool: 写入成功返回True，失败返回False

        Raises:
            IOError: 文件写入失败时抛出
        """
        try:
            # 确保目录存在
            token_dir = Path(self.access_token_path).parent
            token_dir.mkdir(parents=True, exist_ok=True)

            with open(self.access_token_path, "w", encoding="utf-8") as f:
                f.write(access_token)

            logger.info(f"访问令牌已成功保存到: {self.access_token_path}")
            logger.debug(f"令牌已保存: {access_token[:4]}****{access_token[-4:]}")
            return True

        except IOError as e:
            logger.error(f"写入访问令牌失败: {e}")
            raise IOError(f"无法写入访问令牌文件: {e}")
        except Exception as e:
            logger.error(f"保存访问令牌时发生未知错误: {e}")
            raise RuntimeError(f"保存访问令牌失败: {e}")

    def get_access_token(self) -> Optional[str]:
        """
        获取访问令牌

        Returns:
            Optional[str]: 访问令牌，失败时返回None

        Raises:
            IOError: 文件读取失败时抛出
        """
        try:
            if not os.path.exists(self.access_token_path):
                logger.warning(f"访问令牌文件不存在: {self.access_token_path}")
                return None

            with open(self.access_token_path, "r", encoding="utf-8") as f:
                access_token = f.read().strip()

            if not access_token:
                logger.warning("访问令牌文件为空")
                return None

            logger.debug(f"读取访问令牌: {access_token[:4]}****{access_token[-4:]}")
            return access_token

        except IOError as e:
            logger.error(f"读取访问令牌失败: {e}")
            raise IOError(f"无法读取访问令牌文件: {e}")
        except Exception as e:
            logger.error(f"读取访问令牌时发生未知错误: {e}")
            raise RuntimeError(f"读取访问令牌失败: {e}")

    def validate_access_token(self) -> bool:
        """
        验证访问令牌是否有效

        Returns:
            bool: 令牌有效返回True，无效返回False

        Raises:
            RequestException: 网络请求失败时抛出
        """
        try:
            if not os.path.exists(self.access_token_path):
                logger.warning("访问令牌文件不存在")
                return False

            access_token = self.get_access_token()
            if not access_token:
                logger.warning("无法读取访问令牌")
                return False

            # 使用Dify的API来验证令牌有效性
            url = f"{self.base_url}/console/api/apps"
            params = {"page": 1, "limit": 30, "name": "", "is_created_by_me": False}
            headers = {"Authorization": f"Bearer {access_token}"}

            logger.debug("开始验证访问令牌有效性...")
            response = requests.get(url, headers=headers, params=params, timeout=self.timeout)

            if response.status_code == 200:
                logger.info("访问令牌验证成功")
                return True
            elif response.status_code == 401:
                logger.warning("访问令牌已过期或无效")
                return False
            else:
                logger.warning(f"令牌验证失败，状态码: {response.status_code}")
                return False

        except Timeout as e:
            logger.error(f"验证访问令牌超时: {e}")
            raise RequestException(f"令牌验证请求超时: {e}")
        except RequestException as e:
            logger.error(f"验证访问令牌网络错误: {e}")
            raise RequestException(f"令牌验证网络请求失败: {e}")
        except Exception as e:
            logger.error(f"验证访问令牌时发生未知错误: {e}")
            raise RuntimeError(f"令牌验证失败: {e}")

    def clear_access_token(self) -> bool:
        """
        清除访问令牌

        Returns:
            bool: 清除成功返回True，失败返回False
        """
        try:
            if os.path.exists(self.access_token_path):
                os.remove(self.access_token_path)
                logger.info("访问令牌已清除")
            return True
        except OSError as e:
            logger.error(f"清除访问令牌失败: {e}")
            return False
