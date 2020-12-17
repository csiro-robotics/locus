# Locus


This repository is an official implementation of the paper: [Locus: LiDAR-based Place Recognition using Spatiotemporal Higher-Order Pooling](https://arxiv.org/abs/2011.14497)

## Method overview.

![](./utils/docs/pipeline.png)



## Usage

### Set up environment
- Create [conda](https://docs.conda.io/en/latest/) environment with [Open3D](http://www.open3d.org/docs/release/) and [tensorflow-1.8](https://www.tensorflow.org/) with python 3.6:
```bash
conda create --name locus_env python=3.6
conda activate locus_env
pip install -r requirements.txt
```
- Set up [python-pcl](https://github.com/strawlab/python-pcl). See ```utils/setup_python_pcl.txt```
- Segment feature extraction uses the pre-trained model from [ethz-asl/segmap](https://github.com/ethz-asl/segmap). Download and copy the relevant content in [segmap_data](http://robotics.ethz.ch/~asl-datasets/segmap/segmap_data/) into ```~/.segmap/```:
```bash
./utils/get_segmap_data.bash
```
- Download the [KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and set the path in ```config.yml```.

### Segmentation
- Extract and save Euclidean point segments: 
```bash
python ./segmentation/extract_segments.py 
```
- Generate and save SegMap-CNN features for all segments: 
```bash
python ./segmentation/generate_segment_features.py 
```

### Descriptor Generation
- Generate and save Locus descriptor:
```bash
python ./descriptor_generation/locus_descriptor.py 
```

### Online segmentation and description
- Segment and generate Locus descriptor for each scan in selected a sequence:
```bash
python main.py --seq '06'
```

### Evaluation
- Sequence-wise place-recognition using extracted descriptors:
```bash
python ./evaluation/place_recognition.py  --seq  '06' 
```
- Evaluation of place-recognition performance using Precision-Recall curves (multiple sequences):  
```bash
python ./evaluation/pr_curve.py 
```

## Citation

If you find this work usefull in your research, please consider citing:

```
@article{vidanapathirana2020locus,
  title={Locus: LiDAR-based Place Recognition using Spatiotemporal Higher-Order Pooling},
  author={Vidanapathirana, Kavisha and Moghadam, Peyman and Harwood, Ben and Zhao, Muming and Sridharan, Sridha and Fookes, Clinton},
  year={2020},
  eprint={arXiv preprint arXiv:2011.14497}
}
```

## Acknowledgment
Functions from 3rd party have been acknowledged at the respective function definitions or readme files. This project was mainly inspired by the following: [ethz-asl/segmap](https://github.com/ethz-asl/segmap) and [irapkaist/scancontext](https://github.com/irapkaist/scancontext).

## Contact
For questions/feedback, 
 ```
 kavisha.vidanapathirana@data61.csiro.au
 ```
