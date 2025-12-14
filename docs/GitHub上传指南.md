# GitHub 上传指南

本文档介绍如何将本地项目上传到 GitHub 仓库。

## 🚀 快速开始（三步上传）

### 前提条件

1. 已安装 Git
2. 拥有 GitHub 账号
3. 已配置 Git 用户信息

```bash
# 检查 Git 是否安装
git --version

# 配置 Git 用户信息（首次使用需要）
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱@example.com"
```

---

## 方法 1：先创建 GitHub 仓库（推荐）

### 步骤 1：在 GitHub 上创建新仓库

1. 登录 [GitHub](https://github.com)
2. 点击右上角 `+` → `New repository`
3. 填写仓库信息：
   - **Repository name**: `medical-qa-lora`（或其他名称）
   - **Description**: `中文医疗问答系统 - 基于 Qwen2.5-4B 的 LoRA/QLoRA 微调项目`
   - **Public** 或 **Private**：根据需要选择
   - ⚠️ **不要勾选** "Add a README file"（因为本地已有）
   - ⚠️ **不要勾选** "Add .gitignore"（因为本地已有）
4. 点击 `Create repository`

### 步骤 2：初始化本地仓库并上传

```bash
# 进入项目目录
cd project

# 初始化 Git 仓库
git init

# 添加所有文件到暂存区
git add .

# 提交到本地仓库
git commit -m "Initial commit: 中文医疗问答系统项目"

# 添加远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/你的用户名/medical-qa-lora.git

# 推送到 GitHub（首次推送）
git branch -M main
git push -u origin main
```

### 步骤 3：验证上传

访问你的 GitHub 仓库页面，确认文件已成功上传。

---

## 方法 2：使用 GitHub CLI（更简单）

### 安装 GitHub CLI

```bash
# macOS
brew install gh

# Windows (使用 winget)
winget install --id GitHub.cli

# Linux
# 参考：https://github.com/cli/cli/blob/trunk/docs/install_linux.md
```

### 上传步骤

```bash
# 进入项目目录
cd project

# 初始化 Git 仓库
git init
git add .
git commit -m "Initial commit: 中文医疗问答系统项目"

# 登录 GitHub（首次使用）
gh auth login

# 创建仓库并推送（一条命令完成）
gh repo create medical-qa-lora --public --source=. --push

# 或创建私有仓库
gh repo create medical-qa-lora --private --source=. --push
```

---

## 📝 详细步骤说明

### 1. 检查 .gitignore 文件

确保项目中有 `.gitignore` 文件，避免上传不必要的文件：

```bash
# 查看 .gitignore 内容
cat .gitignore
```

如果没有，创建一个：

```bash
# 创建 .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# 数据和模型文件（太大，不上传）
data/raw/
data/processed/
models/
outputs/
*.bin
*.safetensors
*.pt
*.pth
*.ckpt

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# 日志
*.log
logs/

# 系统文件
.DS_Store
Thumbs.db

# Jupyter Notebook
.ipynb_checkpoints

# 环境变量
.env
.env.local

# TensorBoard
runs/
EOF
```

### 2. 初始化 Git 仓库

```bash
# 进入项目目录
cd project

# 初始化 Git 仓库
git init

# 查看状态
git status
```

### 3. 添加文件到暂存区

```bash
# 添加所有文件
git add .

# 或者选择性添加
git add README.md
git add configs/
git add src/
git add scripts/
git add docs/

# 查看暂存区状态
git status
```

### 4. 提交到本地仓库

```bash
# 提交
git commit -m "Initial commit: 中文医疗问答系统项目"

# 查看提交历史
git log
```

### 5. 关联远程仓库

```bash
# 添加远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/你的用户名/仓库名.git

# 验证远程仓库
git remote -v
```

### 6. 推送到 GitHub

```bash
# 首次推送（设置上游分支）
git branch -M main
git push -u origin main

# 后续推送（简化命令）
git push
```

---

## 🔐 身份验证

### 方法 1：使用 Personal Access Token（推荐）

GitHub 已不再支持密码验证，需要使用 Token：

1. 访问 GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 点击 `Generate new token (classic)`
3. 设置权限：勾选 `repo`（完整仓库访问权限）
4. 生成并复制 Token（只显示一次，请保存）
5. 推送时使用 Token 作为密码：
   ```bash
   Username: 你的用户名
   Password: ghp_xxxxxxxxxxxx（你的 Token）
   ```

### 方法 2：使用 SSH（更安全）

```bash
# 1. 生成 SSH 密钥
ssh-keygen -t ed25519 -C "你的邮箱@example.com"

# 2. 启动 ssh-agent
eval "$(ssh-agent -s)"

# 3. 添加私钥
ssh-add ~/.ssh/id_ed25519

# 4. 复制公钥
cat ~/.ssh/id_ed25519.pub

# 5. 在 GitHub 添加 SSH 密钥
# Settings → SSH and GPG keys → New SSH key
# 粘贴公钥内容

# 6. 测试连接
ssh -T git@github.com

# 7. 使用 SSH 地址添加远程仓库
git remote add origin git@github.com:你的用户名/仓库名.git
```

---

## 📦 后续更新

### 修改文件后推送

```bash
# 1. 查看修改
git status

# 2. 添加修改的文件
git add .

# 3. 提交
git commit -m "描述你的修改"

# 4. 推送
git push
```

### 常用 Git 命令

```bash
# 查看状态
git status

# 查看修改内容
git diff

# 查看提交历史
git log
git log --oneline

# 撤销修改（未暂存）
git checkout -- 文件名

# 撤销暂存
git reset HEAD 文件名

# 创建分支
git branch 分支名
git checkout -b 分支名

# 切换分支
git checkout 分支名

# 合并分支
git merge 分支名

# 拉取远程更新
git pull

# 克隆仓库
git clone https://github.com/用户名/仓库名.git
```

---

## ⚠️ 注意事项

### 1. 不要上传大文件

GitHub 单个文件限制 100MB，建议：

```bash
# 检查大文件
find . -type f -size +50M

# 如果已经提交大文件，需要从历史中删除
git filter-branch --tree-filter 'rm -f 大文件路径' HEAD
```

### 2. 敏感信息保护

不要上传：
- API 密钥
- 密码
- 数据库连接字符串
- 私钥文件

使用环境变量或配置文件（加入 .gitignore）。

### 3. 数据和模型文件

本项目的 `data/` 和 `models/` 目录已在 `.gitignore` 中，不会上传。

用户需要自己运行脚本下载：
```bash
python scripts/download_data.py
python scripts/download_model.py
```

---

## 🎯 完整示例

```bash
# ===== 1. 配置 Git（首次使用）=====
git config --global user.name "张三"
git config --global user.email "zhangsan@example.com"

# ===== 2. 进入项目目录 =====
cd ~/Desktop/python_project/project

# ===== 3. 初始化仓库 =====
git init

# ===== 4. 添加文件 =====
git add .

# ===== 5. 提交 =====
git commit -m "Initial commit: 中文医疗问答系统项目"

# ===== 6. 在 GitHub 创建仓库 =====
# 访问 https://github.com/new
# 创建名为 medical-qa-lora 的仓库

# ===== 7. 关联远程仓库 =====
git remote add origin https://github.com/你的用户名/medical-qa-lora.git

# ===== 8. 推送 =====
git branch -M main
git push -u origin main

# ===== 9. 验证 =====
# 访问 https://github.com/你的用户名/medical-qa-lora
```

---

## 📚 相关资源

- [Git 官方文档](https://git-scm.com/doc)
- [GitHub 官方文档](https://docs.github.com)
- [GitHub CLI 文档](https://cli.github.com/manual/)
- [Git 教程 - 廖雪峰](https://www.liaoxuefeng.com/wiki/896043488029600)

---

## 🆘 常见问题

### Q1: 推送时提示 "Permission denied"

**解决方案**：
- 检查 Token 是否正确
- 或配置 SSH 密钥

### Q2: 推送时提示 "rejected"

**解决方案**：
```bash
# 先拉取远程更新
git pull origin main --rebase

# 再推送
git push
```

### Q3: 如何删除远程仓库的文件

```bash
# 删除本地文件
git rm 文件名

# 提交
git commit -m "删除文件"

# 推送
git push
```

### Q4: 如何修改最后一次提交

```bash
# 修改提交信息
git commit --amend -m "新的提交信息"

# 强制推送（谨慎使用）
git push --force
```

---

**提示**：首次上传建议使用"方法 1"，步骤清晰，便于理解。熟悉后可以使用 GitHub CLI 简化操作。
