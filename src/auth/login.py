import time
import os
import requests
import logging

from pathlib import Path
from typing import Optional, Dict
from apscheduler.schedulers.background import BackgroundScheduler
from requests.exceptions import RequestException, Timeout, ConnectionError

from src.auth.token_opt import Token
from src.config.bootstrap import get_runtime, bootstrap_from_unified


# 自定义异常类
class AuthenticationError(Exception):
    """认证失败异常"""
    pass


class SessionExpiredError(Exception):
    """会话过期异常"""
    pass


class PermissionDeniedError(Exception):
    """权限验证失败异常"""
    pass


class NetworkConnectionError(Exception):
    """网络连接异常"""
    pass


class ConfigurationError(Exception):
    """配置错误异常"""
    pass


def _get_logger():
    """获取登录模块日志句柄

    优先使用项目内的 loguru 日志管理器；若未初始化或不可用，回退到标准 logging。
    """
    try:
        from src.utils.logger import get_logger as _get, _log_manager
        if _log_manager.is_configured():
            return _get("auth.login")
    except Exception:
        pass
    # 统一名称为 auth.login；已通过 logger 桥接将标准 logging 转发至 loguru
    return logging.getLogger("auth.login")


def _mask_secret(secret: Optional[str], keep: int = 2) -> str:
    """掩码敏感信息，避免在日志中泄露明文。

    规则：
    - None/空：返回 "<empty>"
    - 长度<=keep*2：返回 "<masked>"
    - 其他：保留前 keep 和后 keep 个字符，中间以 **** 替代，并附带长度信息
    """
    try:
        if not secret:
            return "<empty>"
        s = str(secret)
        if len(s) <= keep * 2:
            return "<masked>"
        return f"{s[:keep]}****{s[-keep:]} (len={len(s)})"
    except Exception:
        return "<masked>"


# 默认初始化一次，以便静态分析与基础运行；函数内会在需要时刷新至最新句柄
logger = _get_logger()


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

        Raises:
            AuthenticationError: 认证失败时抛出
            NetworkConnectionError: 网络连接异常时抛出
            Timeout: 请求超时时抛出
            RequestException: 其他请求异常时抛出
        """
        url = f"{self.base_url}/console/api/login"
        payload = {
            "email": self.email,
            "password": self.password,
            "language": "zh-Hans",
            "remember_me": True
        }
        try:
            # 刷新日志句柄，确保使用集中式日志系统
            global logger
            logger = _get_logger()

            # 在 DEBUG 级别输出账户与部分敏感信息（已掩码）
            try:
                masked_pw = _mask_secret(self.password, keep=2)
                logger.debug(
                    "开始登录请求",
                    extra={
                        "email": self.email,
                        "base_url": self.base_url,
                        "timeout": self.timeout,
                        "language": payload['language'],
                        "remember_me": payload['remember_me'],
                        "password": masked_pw,
                    },
                )
            except Exception:
                logger.debug(
                    f"开始登录请求: {self.email} @ {self.base_url} (timeout={self.timeout}, lang={payload['language']}, remember={payload['remember_me']})"
                )
            t0 = time.perf_counter()
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            t1 = time.perf_counter()
            logger.debug(
                f"登录请求完成: status={response.status_code}, duration_ms={(t1 - t0) * 1000:.1f}, keys={list(data.keys())}"
            )
            if data.get("result") == "success":
                # 登录成功日志（不泄露完整令牌）
                returned = data.get("data")
                masked_token = None
                access_token_val = None
                try:
                    if isinstance(returned, dict):
                        token_candidate = returned.get("access_token")
                        access_token_val = token_candidate if isinstance(token_candidate, str) else None
                        if isinstance(token_candidate, str) and len(token_candidate) >= 8:
                            masked_token = f"{token_candidate[:4]}****{token_candidate[-4:]}"
                except Exception:
                    pass
                if masked_token:
                    logger.info(f"登录成功: 用户 {self.email} @ {self.base_url} (token={masked_token})")
                else:
                    logger.info(f"登录成功: 用户 {self.email} @ {self.base_url}")
                # 如需在 DEBUG 日志中输出完整令牌，需显式开启环境变量 DEBUG_EXPOSE_TOKENS=true
                try:
                    expose = str(os.getenv("DEBUG_EXPOSE_TOKENS", "")).lower() in {"1", "true", "yes", "on"}
                    if expose and access_token_val:
                        logger.warning(
                            "DEBUG_EXPOSE_TOKENS 已启用：将仅在 DEBUG 级别输出完整访问令牌（请勿在生产环境开启）"
                        )
                        logger.debug(f"access_token (debug only) = {access_token_val}")
                except Exception:
                    pass
                if isinstance(returned, dict):
                    try:
                        logger.debug(f"登录成功返回字段: {list(returned.keys())}")
                    except Exception:
                        pass
                return returned
            else:
                error_msg = data.get("message", "Unknown error")
                logger.error(f"Authentication Failed: {error_msg}")
                raise AuthenticationError(error_msg)
        except Timeout as e:
            logger.error(f"Login Timeout: {e}")
            raise NetworkConnectionError(f"登录请求超时: {e}")
        except ConnectionError as e:
            logger.error(f"Connection Error: {e}")
            raise NetworkConnectionError(f"无法连接到服务器: {e}")
        except requests.exceptions.HTTPError as e:
            try:
                logger.debug(
                    f"HTTPError detail: status={response.status_code}, body_len={len(getattr(response, 'text', '') or '')}")
            except Exception:
                pass
            if response.status_code == 401:
                raise AuthenticationError("用户名或密码错误")
            elif response.status_code == 403:
                raise PermissionDeniedError("访问被拒绝，权限不足")
            elif response.status_code == 429:
                raise AuthenticationError("请求过于频繁，请稍后再试")
            else:
                logger.error(f"HTTP Error: {e}")
                raise AuthenticationError(f"服务器返回错误: {response.status_code}")
        except RequestException as e:
            logger.error(f"Request Error: {e}")
            raise NetworkConnectionError(f"网络请求失败: {e}")
        except ValueError as e:
            logger.error(f"JSON Decode Error: {e}")
            raise AuthenticationError("服务器响应格式错误")
        except Exception as e:
            logger.error(f"Login Error: {e}")
            raise AuthenticationError(f"登录失败: {e}")

    def logout(self, access_token: str) -> bool:
        """
        用户登出，清除服务端 Refresh Token。

        Args:
            access_token (str): 访问令牌

        Returns:
            bool: True 表示成功，False 表示失败

        Raises:
            AuthenticationError: 认证失败时抛出
            SessionExpiredError: 会话过期时抛出
            NetworkConnectionError: 网络连接异常时抛出
        """
        url = f"{self.base_url}/console/api/logout"
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            global logger
            logger = _get_logger()
            masked = f"{access_token[:4]}****{access_token[-4:]}" if isinstance(access_token, str) and len(
                access_token) >= 8 else "<masked>"
            logger.debug(f"开始登出请求: token={masked}")
            t0 = time.perf_counter()
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            t1 = time.perf_counter()
            logger.debug(f"登出请求完成: status={response.status_code}, duration_ms={(t1 - t0) * 1000:.1f}")
            if response.status_code == 200:
                data = response.json()
                if data.get("result") == "success":
                    logger.info("登出成功")
                    return True
                else:
                    error_msg = data.get("message", "Logout failed")
                    logger.error(f"Logout Failed: {error_msg}")
                    raise AuthenticationError(error_msg)
        except Timeout as e:
            logger.error(f"Logout Timeout: {e}")
            raise NetworkConnectionError(f"登出请求超时: {e}")
        except ConnectionError as e:
            logger.error(f"Logout Connection Error: {e}")
            raise NetworkConnectionError(f"无法连接到服务器: {e}")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise SessionExpiredError("会话已过期，请重新登录")
            elif response.status_code == 403:
                raise PermissionDeniedError("登出权限不足")
            elif response.status_code == 429:
                raise AuthenticationError("请求过于频繁，请稍后再试")
            else:
                logger.error(f"Logout HTTP Error: {e}")
                raise AuthenticationError(f"登出失败: HTTP {response.status_code}")
        except RequestException as e:
            logger.error(f"Logout Request Error: {e}")
            raise NetworkConnectionError(f"登出网络请求失败: {e}")
        except ValueError as e:
            logger.error(f"Logout JSON Decode Error: {e}")
            raise AuthenticationError("登出响应格式错误")
        except Exception as e:
            logger.error(f"Logout Error: {e}")
            raise AuthenticationError(f"登出失败: {e}")

        logger.warning(f"Logout unexpected response: Status: {response.status_code}, Response: {response.text}")
        return False

    def login_job(self):
        """定时登录任务

        Raises:
            根据具体情况抛出相应的异常
        """
        global logger
        logger = _get_logger()
        logger.info("开始执行定时登录任务...")
        try:
            result = self.login()
            if result:
                # 检查result中是否包含access_token - 获取实际的数据结构
                if isinstance(result, dict):
                    try:
                        logger.debug(f"login_job: 登录返回字段: {list(result.keys())}")
                    except Exception:
                        pass
                    # 检查是否是标准的API响应格式 {"result": "success", "data": {...}}
                    if "result" in result and "data" in result and result.get("result") == "success":
                        token_data = result["data"]
                        if not isinstance(token_data, dict) or "access_token" not in token_data:
                            logger.error("登录响应数据格式无效")
                            raise AuthenticationError("登录响应数据格式无效")
                        access_token = token_data["access_token"]
                    else:
                        # 直接检查是否有access_token字段
                        if "access_token" not in result:
                            logger.error("登录结果中缺少access_token")
                            raise AuthenticationError("登录结果中缺少access_token")
                        access_token = result["access_token"]
                else:
                    logger.error("登录结果格式无效")
                    raise AuthenticationError("登录结果格式无效")

                Token().rewrite_access_token(access_token)
                logger.debug(f"已保存访问令牌: {access_token[:4]}****{access_token[-4:]}")
                # 可选：在 DEBUG 级别打印完整令牌，需显式开启 DEBUG_EXPOSE_TOKENS
                try:
                    expose = str(os.getenv("DEBUG_EXPOSE_TOKENS", "")).lower() in {"1", "true", "yes", "on"}
                    if expose:
                        logger.warning(
                            "DEBUG_EXPOSE_TOKENS 已启用：将仅在 DEBUG 级别输出完整访问令牌（请勿在生产环境开启）"
                        )
                        logger.debug(f"access_token (debug only) = {access_token}")
                except Exception:
                    pass
                logger.info("登录成功")

                if Token().validate_access_token():  # 修复拼写错误
                    logger.info("访问令牌有效")
                else:
                    logger.error("访问令牌无效")
                    raise AuthenticationError("访问令牌无效")
            else:
                logger.error("登录失败")
        except AuthenticationError as e:
            logger.error(f"认证失败: {e}")
            raise AuthenticationError(f"认证失败: {e}")
        except NetworkConnectionError as e:
            logger.error(f"网络连接异常: {e}")
            raise NetworkConnectionError(f"网络连接异常: {e}")
        except SessionExpiredError as e:
            logger.error(f"会话过期: {e}")
            raise SessionExpiredError(f"会话过期: {e}")
        except PermissionDeniedError as e:
            logger.error(f"权限不足: {e}")
            raise PermissionDeniedError(f"权限不足: {e}")
        except Exception as e:
            logger.error(f"登录任务执行出错: {e}")
            raise Exception(f"登录任务执行出错: {e}")


def run(config_path: str = "config/config.yaml"):
    """运行认证客户端

    Args:
        config_path (str): 配置文件路径，默认为 "config/env_config.yaml"

    Raises:
        FileNotFoundError: 配置文件未找到时抛出
        KeyError: 配置项缺失时抛出
        ValueError: 配置值无效时抛出
        Exception: 其他运行时异常
    """
    try:
        global logger
        logger = _get_logger()
        # 确保 runtime 可用（若外部尚未引导，则本地引导）
        try:
            rt = get_runtime()
        except Exception:
            rt = bootstrap_from_unified(Path(config_path))

        # 读取统一配置中的 dify/auth
        url = (rt.app.dify or {}).get("base_url")
        auth_cfg = rt.app.auth or {}
        api_key = auth_cfg.get("api_key")
        email = auth_cfg.get("username")
        password = auth_cfg.get("password")

        if not url:
            raise ValueError("配置文件缺少必要的认证信息: dify.base_url")

        # 优先使用 api_key（无需网络请求）
        if api_key:
            Token().rewrite_access_token(api_key)
            logger.info("已使用 API Key 完成认证并注入访问令牌")
            return

        # 回退到用户名/密码登录
        missing_configs = []
        if not email:
            missing_configs.append("auth.username")
        if not password:
            missing_configs.append("auth.password")
        if missing_configs:
            raise ValueError(f"配置文件缺少必要的认证信息: {', '.join(missing_configs)}")

        client = DifyAuthClient(url, email=email, password=password)
        scheduler = BackgroundScheduler()
        scheduler.add_job(client.login_job, 'interval', hours=1, id='dify_login_job')
        scheduler.start()
        logger.info("调度器已启动（用户名/密码模式）...")

    except FileNotFoundError as e:
        logger.error(f"配置文件未找到: {e}")
        raise FileNotFoundError(f"无法找到配置文件: {e}")
    except KeyError as e:
        logger.error(f"配置项缺失: {e}")
        raise KeyError(f"配置文件中缺少必要的配置项: {e}")
    except ValueError as e:
        logger.error(f"配置值无效: {e}")
        raise ValueError(f"配置文件中的值无效: {e}")
    except Exception as e:
        logger.error(f"运行时异常: {e}")
        raise


if __name__ == "__main__":
    """脚本入口点，提供完整的异常处理"""
    try:
        run("../../config/config.yaml")
    except AuthenticationError as e:
        logger.error(f"认证失败: {e}")
        exit(1)
    except NetworkConnectionError as e:
        logger.error(f"网络连接异常: {e}")
        exit(2)
    except ConfigurationError as e:
        logger.error(f"配置错误: {e}")
        exit(3)
    except KeyboardInterrupt:
        logger.info("用户中断程序运行")
        exit(0)
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        exit(4)
