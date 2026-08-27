# 安全政策

## 报告漏洞

请不要在公开 Issue 中提交密钥、个人信息、内网地址、日志原文或可复现的真实业务数据。请使用 GitHub 仓库的 **Security → Report a vulnerability** 私密报告渠道。

## 部署安全基线

- 仅从 `.env.example` 复制配置，真实 `.env` 不得提交、打包或分享。
- 为 `JWT_SECRET_KEY`、管理员密码、数据库密码和所有 API Key 使用独立的强随机值。
- 修改 MinIO、MySQL、Redis、Neo4j 等中间件的默认凭据，并限制管理端口的网络暴露。
- 生产环境保持 `DEBUG=false` 和 `MOCK_LLM=false`，并在反向代理层启用 HTTPS。
- 如果密钥曾进入 Git 历史、日志、镜像或聊天记录，应立即在服务提供方处撤销并轮换；仅删除文件不能使旧密钥失效。

## 发布前检查

1. 确认 `git status --ignored` 中 `.env`、数据目录、日志和构建缓存处于忽略状态。
2. 使用密钥扫描工具检查当前快照和 Git 历史。
3. 检查 `.env.example` 仅包含占位符和可公开的默认值。
4. 确认 Docker 构建上下文不包含凭据、业务数据或本地归档。
