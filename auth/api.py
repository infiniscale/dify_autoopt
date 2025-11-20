import requests
import logging
from typing import Optional, Dict

# 设置模块日志
logger = logging.getLogger(__name__)


class DifyAuthClient:
    def __init__(self, base_url: str, timeout: int = 10, email: str = None, password: str = None):
        """
        初始化 Dify 认证客户端。

        Args:
            base_url (str): Dify 实例地址，如 "http://localhost:3000"
            timeout (int): 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.email = email
        self.password = password

    def login(self) -> Optional[
        Dict[str, str]]:
        """
        用户登录，获取访问令牌。

        Returns:
            dict: {"access_token": "...", "refresh_token": "..."} 或 None（失败）
        """
        url = f"{self.base_url}/console/api/login"
        payload = {
            "email": self.email,
            "password": self.password,
            "language": "zh-Hans",
            "remember_me": True
        }
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data.get("result") == "success":
                return data.get("data")
            else:
                logger.error(f"Login Failed: {data.get('message', 'Unknown error')}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Login Error: {e}")
            return None

    def refresh_token(self, access_token: str) -> Optional[Dict[str, str]]:
        """
        刷新令牌（需当前 access_token 仍可用于认证）。

        Returns:
            dict: {"access_token": "...", "refresh_token": "..."} 或 None
        """
        url = f"{self.base_url}/console/api/refresh-token"
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            response = requests.post(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data.get("result") == "success":
                return data.get("data")
            else:
                logger.error(f"Refresh Failed: {data.get('message', 'Unknown error')}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Refresh Error: {e}")
            return None

    def logout(self, access_token: str) -> bool:
        """
        用户登出，清除服务端 Refresh Token。

        Returns:
            bool: True 表示成功，False 表示失败
        """
        url = f"{self.base_url}/console/api/logout"
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if data.get("result") == "success":
                    return True
            logger.warning(f"Logout Failed: Status: {response.status_code}, Response: {response.text}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Logout Error: {e}")
            return False
