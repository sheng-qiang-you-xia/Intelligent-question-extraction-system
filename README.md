# 智能题目提取系统 

## 项目简介

这是一个基于深度学习和多模态大模型的智能题目提取系统，能够从各种格式的文档（PDF、PPT、DOC、PNG）中自动识别、提取和结构化题目内容。系统采用YOLO目标检测模型进行文档布局分析，结合视觉语言模型(VLM)和大型语言模型(LLM)实现高精度的题目识别和结构化输出。

## 核心特性

- **多格式支持**: 支持PDF、PPT、PPTX、DOC、DOCX、PNG等多种文档格式
- **智能布局分析**: 基于YOLO v10的文档布局检测，准确识别文本、图片、表格等元素
- **多模态处理**: 结合视觉语言模型和文本大模型，实现图文混合内容的精确提取
- **结构化输出**: 自动将题目转换为标准化的JSON和Markdown格式
- **图片关联**: 智能关联题目与对应的图片、表格内容
- **批量处理**: 支持多页面文档的批量题目提取

## 系统架构

```
输入文档 → 格式转换 → 页面分割 → YOLO布局检测 → VLM内容识别 → LLM结构化 → 输出题目
    ↓           ↓          ↓           ↓            ↓           ↓
  PDF/PPT/DOC → 图像化 → 单页图像 → 元素检测 → 内容提取 → JSON/MD
```

## 安装指南

### 系统要求

- Python 3.8+
- CUDA 11.0+ (推荐使用GPU加速)
- LibreOffice (用于Office文档转换)
- 8GB+ 内存

### 依赖安装

1. **克隆项目**
```bash
git clone <repository-url>
cd rag_for_topic/v2
```

2. **安装Python依赖**
```bash
pip install -r requirements.txt
```

3. **安装系统依赖**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libreoffice poppler-utils

# CentOS/RHEL
sudo yum install libreoffice poppler-utils

# macOS
brew install libreoffice poppler
```

4. **安装doclayout_yolo模块**
```bash
# 需要单独安装doclayout_yolo包
pip install doclayout_yolo
```

### 环境配置

1. **创建环境变量文件**
```bash
cp .env.example .env
```

2. **配置API密钥**
编辑 `.env` 文件，填入以下配置：
```env
API_KEY=your_qwen_api_key
API_URL=https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation
vlm_MODEL=qwen-vl-max
llm_MODEL=qwen-max
```

3. **下载YOLO模型**
确保 `yolo_model/doclayout_yolo_docstructbench_imgsz1024.pt` 文件存在。

## 使用方法

### 基本使用

```python
from AI_generate_questions import DocumentAnalyzer, QwenVLClient, QwenLLMClient
from utils import get_resource
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化客户端
vlm_client = QwenVLClient(
    api_key=os.getenv("API_KEY"),
    api_url=os.getenv("API_URL"),
    model=os.getenv("vlm_MODEL")
)

llm_client = QwenLLMClient(
    api_key=os.getenv("API_KEY"),
    api_url=os.getenv("API_URL"),
    model=os.getenv("llm_MODEL")
)

# 初始化文档分析器
analyzer = DocumentAnalyzer(
    model_path="yolo_model/doclayout_yolo_docstructbench_imgsz1024.pt"
)

# 处理文档
file_name, full_path = get_resource("your_document_url", "output_path")
image_paths = analyzer.pdf_to_images(full_path, "converted_images")

# 提取题目
for i, image_path in enumerate(image_paths):
    image, result = analyzer.detect_layout(image_path)
    analyzer.extract_questions_md(i, result, image, "questions", vlm_client)

# 清洗和结构化题目
analyzer.clean_questions_from_md(llm_client, "questions", "All_Questions")
```

### 命令行使用

```bash
python AI_generate_questions.py
```

### 支持的文件格式

| 格式 | 扩展名 | 转换方式 | 说明 |
|------|--------|----------|------|
| PDF | .pdf | pdf2image | 直接转换为图像 |
| PowerPoint | .ppt, .pptx | LibreOffice → PDF → 图像 | 先转PDF再转图像 |
| Word | .doc, .docx | LibreOffice → PDF → 图像 | 先转PDF再转图像 |
| 图像 | .png | 直接处理 | 无需转换 |

## 输出格式

### JSON格式
```json
[
  {
    "id": "0_Q1",
    "text": "题目内容...",
    "images": ["image_path1.png", "image_path2.png"],
    "tables": ["表格内容1", "表格内容2"]
  }
]
```

### Markdown格式
每个题目生成独立的 `.md` 文件，包含：
- 题目编号
- 题目文本
- 关联图片
- 表格内容

## 项目结构

```
v2/
├── AI_generate_questions.py    # 主程序文件
├── utils.py                    # 工具函数
├── yolo_model/                 # YOLO模型目录
│   ├── doclayout_yolo_docstructbench_imgsz1024.pt
│   └── README.md
├── requirements.txt            # Python依赖
├── .env.example               # 环境变量示例
└── README.md                  # 项目说明
```

## 核心类说明

### DocumentAnalyzer
文档分析器主类，负责：
- 文档格式转换
- 布局检测
- 题目提取
- 内容结构化

### QwenVLClient
视觉语言模型客户端，负责：
- 图像编码
- 视觉内容识别
- 图文混合内容提取

### QwenLLMClient
文本大模型客户端，负责：
- 文本内容清洗
- 题目结构化
- 内容排版优化

## 配置参数

### YOLO检测参数
- `conf`: 置信度阈值 (默认: 0.2)
- `imgsz`: 图像尺寸 (默认: 1024)
- `device`: 计算设备 (默认: "cuda:0")

### 文档处理参数
- `max_pages`: 最大处理页数 (默认: 5)
- `dpi`: PDF转换DPI (默认: 200)

### API参数
- `temperature`: 生成温度 (默认: 0.0)
- `max_tokens`: 最大输出长度

## 常见问题

### Q: LibreOffice转换失败怎么办？
A: 确保LibreOffice已正确安装，并且soffice命令在PATH中可用。

### Q: CUDA内存不足怎么办？
A: 可以调整batch_size或使用CPU模式（修改device参数）。

### Q: API调用失败怎么办？
A: 检查API密钥和网络连接，确保API配额充足。

### Q: 题目识别不准确怎么办？
A: 可以调整YOLO的置信度阈值或优化提示词模板。

## 性能优化

1. **GPU加速**: 使用CUDA加速YOLO检测
2. **批量处理**: 合理设置max_pages参数
3. **缓存机制**: 对重复文档使用缓存
4. **并行处理**: 多页面并行处理

## 贡献指南

1. Fork 本仓库
2. 创建特性分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -m 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 创建 Pull Request

## 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 更新日志

### v2.0.0
- 支持多种文档格式 (PDF, PPT, DOC, PNG)
- 集成YOLO v10布局检测
- 优化题目提取算法
- 改进图文关联逻辑

### v1.0.0
- 基础PDF题目提取功能
- 简单的布局检测

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com

## 致谢

- [YOLO v10](https://github.com/ultralytics/ultralytics) - 目标检测框架
- [Qwen系列模型](https://github.com/QwenLM/Qwen) - 多模态大模型
- [pdf2image](https://github.com/Belval/pdf2image) - PDF转图像工具
- [LibreOffice](https://www.libreoffice.org/) - 文档转换工具
