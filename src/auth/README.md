# 身份与权限模块

## 功能概述

负责Dify平台的身份认证、会话管理和API密钥管理，确保系统能够安全地访问和操作Dify平台资源。

## 模块组成

### 1. 登录认证 (login.py)
- 用户名密码认证
- API密钥认证
- 认证状态管理

### 2. 会话管理 (session.py)
- 会话令牌管理
- 自动登出处理
- 会话状态监控

### 3. API密钥管理 (api_key.py)
- API密钥存储
- 密钥有效性验证
- 密钥轮换机制

## 功能特性

- 🔐 多种认证方式支持
- 🔄 自动会话续期
- 🛡️ 安全密钥管理
- ⏰ 认证状态监控
- 🚫 异常状态处理

## 使用示例

```python
# 基础认证
from src.auth import AuthManager

auth = AuthManager()
auth.login_with_credentials(username, password)

# API密钥认证
auth.login_with_api_key(api_key)

# 检查认证状态
if auth.is_authenticated():
    # 执行操作
    pass
```

## 配置参数

```yaml
auth:
  base_url: "https://your-dify-instance.com"
  timeout: 30
  retry_count: 3
  session_expiry: 3600
```

## 错误处理

- 认证失败异常
- 会话过期处理
- 网络连接异常
- 权限验证失败

## 安全考虑

- 密码加密存储
- API密钥安全传输
- 会话令牌定期刷新
- 异常登录检测