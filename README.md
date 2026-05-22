This codebase integrates the algorithmic frameworks of most state-of-the-art, publicly available models—such as MonoViT, LiteMono, RaDepth, and HRDepth. Its objective is to bring the lightweight integration paradigm currently prevalent in object detection algorithms to the field of monocular depth estimation. By restructuring the code (e.g., through the introduction of a `cfg/options.yaml` file), the configuration of hyperparameters is made transparent and intuitive, thereby simplifying the training workflow and significantly enhancing the ease of integrating and experimenting with different model components (such as various encoders and decoders).

The full code and detailed training process will be announced later.

# Overview
<img width="1675" height="1030" alt="image" src="https://github.com/user-attachments/assets/1ca6169c-3337-4873-ac82-e87381ca8e03" />

<img width="1703" height="878" alt="image" src="https://github.com/user-attachments/assets/94966065-db21-4729-870a-6c350d2d28bd" />

# Kitti Results
<img width="1224" height="906" alt="image" src="https://github.com/user-attachments/assets/5116af9b-3302-4d37-880e-2d5f50aeccc9" />
| Model             | Params | ImageNet Pretrained | Input size | Abs Rel | Sq Rel | RMSE  | RMSE log | delta < 1.25 | delta < 1.25² | delta < 1.25³ |
|-------------------|--------|---------------------|------------|---------|--------|-------|----------|--------------|---------------|---------------|
| [ours(pretrain)](https://pan.baidu.com/s/17TnXWtF8GZH3nHfDQ3zHWQ?pwd=uvr2)  |  4.5M  | yes                 | 640x192    | 0.105   | 0.769  | 4.571 | 0.183    | 0.887        | 0.963        | 0.983         |
| [ours](https://pan.baidu.com/s/1KbY4a6YDDK2FZTyDgfAuEA?pwd=muxb)            | 4.5M   | no               | 640x192    | 0.113  | 0.874  | 4.749| 0.191   | 0.877       | 0.959        | 0.981  
