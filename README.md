# Music Source Separation Universal Training Code

音轨分离模型导出

## 环境搭建
```
conda create -n msst python=3.12
conda activate msst
pip install -r requirements.txt
```

## 导出ONNX

```
python export_onnx.py --onnx mel_band_roformer_depth4.onnx --model_type mel_band_roformer --config mel_band_roformer_depth4.yaml --ckpt mel_band_roformer_depth4_6.4.ckpt --input_audio Spring.wav 
```

其中config和ckpt需要从[此处](https://github.com/ml-inory/Music-Source-Separation-Training/blob/main/docs/mel_roformer_experiments.md)获取，input_audio请自行准备

运行完成后生成onnx和calibration_dataset

## 导出axmodel

```
pulsar2 build --input mel_band_roformer_depth4.onnx --config mel_band_roformer.json --output_dir mel_band_roformer --output_name mel_band_roformer_depth4.axmodel
```