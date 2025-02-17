# CADR: Image-Text Retrieval Framework Based on Context Anchor Attention and DINOv2
## Introductiom
Cross-modal retrieval aims to bridge the gap between vision and language.The key lies in how to effectively learn the semantic similarity between images and text. For images containing complex scenes, traditional fine-grained alignment methods are difficult to fully capture the association information between visual fragments and text words, leading to problems such as redundancy of visual fragments and alignment error. In this paper, we propose a new image-text retrieval framework based on Context Anchor Attention and DINOv2, known as CADR, to achieve fine-grained alignment. Specifically, we utilize DINOv2 as an image encoder to extract richer feature representations. Additionally, we introduce context anchor attention to implement a long-range context capture mechanism. This mechanism serves to enhance the understanding of the overall semantics of images, enabling us to identify visual patches associated with text more efficiently and accurately. 
## Preparation
We recommended the following dependencies:
- python >= 3.8
- torch >= 1.12.0
- torchvision >= 0.13.0
- transformers >=4.32.0
- opencv-python
- tensorboard

### Environments
```bash
# 克隆项目仓库
git clone https://github.com/your-repo/your-project.git
cd your-project

# 安装 Python 依赖
pip install -r requirements.txt

# 安装 Node.js 依赖
npm install
