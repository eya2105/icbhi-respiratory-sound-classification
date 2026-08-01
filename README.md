# Respiratory Sound Classification — ICBHI 2017

This repository contains a university research project on respiratory sound classification using the ICBHI 2017 Challenge dataset.

The project explores different training strategies to improve classification performance, especially recall, which is very important in medical tasks because missing an abnormal sound can be more harmful than a false alarm.

## Project Goal

The goal of this work is to classify respiratory sound cycles into four classes:

- Normal
- Crackle
- Wheeze
- Both

The main objective is to improve the model’s ability to detect abnormal cases while keeping strong overall performance.

## Dataset

This project uses the **ICBHI 2017 Respiratory Sound Database**.

The dataset contains real clinical recordings collected with different devices and in noisy environments.  
This makes the task challenging because of:

- class imbalance,
- background noise,
- different recording devices,
- overlap between sound classes.

Official dataset:
[ICBHI 2017 Challenge Dataset](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)

## Method

The project is based on an **Audio Spectrogram Transformer (AST)** model and tests several techniques, including:

- baseline training,
- weighted sampling,
- threshold tuning,
- focal loss,
- test-time augmentation,
- offline data augmentation/Online data augmentation,
- patient-level cross-validation.

## Results

The repository includes multiple experiments and compares their results.

The best configuration is highlighted in the report.

## Repository Structure

- `notebooks/` — Jupyter notebooks used for experiments
- `scripts/` — Python scripts for reusable training and evaluation code
- `reports/` — project report 


## Project Context

This project was completed as a **university research assignment**.  
It focuses on experimentation, model comparison, and reproducible analysis rather than building a production system.


## References

- ICBHI 2017 Challenge Dataset
- Audio Spectrogram Transformer (AST)
- Related literature on respiratory sound classification: **"Geometry-Aware Optimization for Respiratory Sound Classification: Enhancing Sensitivity with SAM-Optimized Audio Spectrogram Transformers"** ([arXiv:2512.22564](https://arxiv.org/abs/2512.22564))

