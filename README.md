# CADR: Image-Text Retrieval Framework Based on Context Anchor Attention and DINOv2
## Introductiom
Cross-modal retrieval aims to bridge the gap between vision and language.The key lies in how to effectively learn the semantic similarity between images and text. For images containing complex scenes, traditional fine-grained alignment methods are difficult to fully capture the association information between visual fragments and text words, leading to problems such as redundancy of visual fragments and alignment error. In this paper, we propose a new image-text retrieval framework based on Context Anchor Attention and DINOv2, known as CADR, to achieve fine-grained alignment. Specifically, we utilize DINOv2 as an image encoder to extract richer feature representations. Additionally, we introduce context anchor attention to implement a long-range context capture mechanism. This mechanism serves to enhance the understanding of the overall semantics of images, enabling us to identify visual patches associated with text more efficiently and accurately. 

![示例图片](CADR/imgs/模型结构图.png)

## Preparation
We recommended the following dependencies:
- python >= 3.8
- torch >= 1.12.0
- torchvision >= 0.13.0
- transformers >=4.32.0
- opencv-python
- tensorboard

## Datasets
We have prepared the caption files for two datasets in data/ folder, hence you just need to download the images of the datasets. We hope that the final data are organized as follows:
```bash
data
├── coco  # coco captions
│   ├── train_ids.txt
│   ├── train_caps.txt
│   ├── testall_ids.txt
│   ├── testall_caps.txt
│   └── id_mapping.json
│
├── f30k  # f30k captions
│   ├── train_ids.txt
│   ├── train_caps.txt
│   ├── test_ids.txt
│   ├── test_caps.txt
│   └── id_mapping.json
│
├── flickr30k-images # f30k images
│
├── coco-images # coco images
│   ├── train2014
│   └── val2014
```

## Datasets
First, we set up the arguments, detailed information about the arguments is shown in arguments.py.
```bash
--dataset: the chosen datasets, e.g., f30k and coco.
--data_path: the root path of datasets, e.g., data/.
--multi_gpu: whether to use the multiple GPUs (DDP) to train the models.
--gpu-id, the chosen GPU number, e.g., 0-7.
--logger_name, the path of logger files, e.g., runs/f30k_test or runs/coco_test
```

Then, we run the ```bash train.py``` for model training. 
```bash
### f30k
python train.py --dataset f30k --gpu-id 0

### coco
python train.py --dataset coco --gpu-id 0
```

## Evaluation
Run eval.py to evaluate the trained models on f30k or coco datasets, and you need to specify the model paths.
```bash
python eval.py --dataset f30k --data_path data/ --gpu-id 0
python eval.py --dataset coco --data_path data/ --gpu-id 1
```



