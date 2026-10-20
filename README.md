# MacroHFT-Pro: Enhanced Context-Aware Reinforcement Learning for High-Frequency Trading

[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%252B-orange)](https://pytorch.org)
[![Architecture](https://img.shields.io/badge/Architecture-Memory%20Augmented%20RL-blue)](#)
[![Status](https://img.shields.io/badge/Status-Production%20Refactored-success)](#)

---

## 📌 Executive Summary

This repository houses an advanced, industrial-grade refactoring and enhancement of the **MacroHFT** (KDD 2024) framework for High-Frequency Trading (HFT) and Limit Order Book (LOB) execution. 

While the baseline architecture establishes memory-augmented context-aware reinforcement learning for market simulation, this enhanced framework introduces optimized feature engineering, rigorous risk-adjusted reward shaping, and a modular MLOps architecture designed to mitigate catastrophic tail-risk and policy drift in non-stationary regimes.

---

## 🚀 Key Architectural Enhancements vs. Original Baseline

1. **Advanced Microstructural Feature Engineering:** Refined feature extraction pipelines focusing on Order Flow Imbalance (OFI), multi-level volume profiles, and volatility-adaptive low-pass filtering.
2. **Risk-Adjusted Reward Shaping:** Integrated strict penalties for maximum drawdown, slippage, and inventory holding costs directly into the agent's core optimization objective, prioritizing homeostatic stability over naive return chasing.
3. **Optimized Memory & Attention:** Enhanced temporal context windows to capture micro-structural price dynamics across high-frequency tick streams with minimized latency overhead.
4. **Production-Grade MLOps Layout:** Completely modularized into clean `src/` packages, declarative YAML configurations, and robust unit-testing suites (`pytest`).

---

## 🏗️ Repository Architecture

```text
MacroHFT-Enhanced/
├── assets/                  # Architecture diagrams & performance benchmark charts
├── configs/                 # Hierarchical YAML configuration files
├── data/                    # Data pipelines & sample mock datasets
│   └── processed/           # Sample schema data for pipeline verification
├── src/                     # Core Production Source Code
│   ├── environment/         # LOB simulation & market state spaces
│   ├── models/              # Memory-augmented networks & attention mechanisms
│   ├── agents/              # Reinforcement learning training & execution loops
│   └── execution/           # Backtesting engine & risk management constraints
├── tests/                   # Automated unit & integration testing suite (pytest)
├── scripts/                 # CLI entry-points for training and out-of-sample backtesting
├── .gitignore               # Strict exclusion of raw proprietary data & model weights
├── requirements.txt         # Pinned system dependencies
└── README.md                # Project Documentation

---

## 🙏 Acknowledgements & Baseline Attribution

This project builds upon and enhances the open-source architecture introduced by the researchers of **MacroHFT** (KDD 2024). We express our gratitude to the original authors for their foundational work in memory-augmented context-aware reinforcement learning for high-frequency trading:

> **MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading**  
> *Chuqiao Zong, Chaojie Wang, Molei Qin, Lei Feng, Xinrun Wang, and Bo An*  
> *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2024)*  
> [arXiv:2406.14537 [cs.LG]](https://arxiv.org/abs/2406.14537)
