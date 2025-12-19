#!/bin/bash
# Git LFS 设置脚本 - 在 Colab 中运行

echo "=== Git LFS 设置 ==="

# 1. 安装 Git LFS
echo "1. 安装 Git LFS..."
apt-get update -qq
apt-get install -y git-lfs

# 2. 初始化 Git LFS
echo "2. 初始化 Git LFS..."
git lfs install

# 3. 查看 LFS 跟踪的文件
echo "3. 查看 LFS 配置..."
git lfs track

# 4. 添加 .gitattributes
echo "4. 添加 .gitattributes..."
git add .gitattributes

# 5. 提交配置
echo "5. 提交 Git LFS 配置..."
git commit -m "配置 Git LFS 用于大文件管理"

echo ""
echo "✓ Git LFS 设置完成！"
echo ""
echo "接下来的步骤："
echo "1. 添加数据和模型: git add data/ models/"
echo "2. 提交: git commit -m '添加数据集和模型'"
echo "3. 推送: git push origin main"
echo ""
echo "注意: GitHub 免费账户 LFS 限制为 1GB 存储 + 1GB/月带宽"
