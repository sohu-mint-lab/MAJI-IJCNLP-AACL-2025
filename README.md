# MAJI: A Multi-Agent Workflow for Augmenting Journalistic Interviews

[![Conference](https://img.shields.io/badge/IJCNLP--AACL-2025-blue)](https://2025.aaclnet.org/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://openreview.net/pdf?id=lMFDNS5ZDi)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for the paper **"MAJI: A Multi-Agent Workflow for Augmenting Journalistic Interviews"**, accepted to **IJCNLP-AACL 2025**.

MAJI is a system designed to act as a creative partner for journalists during interviews. It employs a structured, multi-agent workflow based on the divergent-convergent thinking model to generate insightful, coherent, and original follow-up questions in real-time.

---

##  Repository Structure

- `dc_agents/`: Core implementations of different DCAgent versions.
- `data`: Pipeline data storage.
- `eval/`: Automated evaluator scripts and metrics calculation.
- `maji_newsinterview_pipeline.py`: Main entry point for running the generation pipeline.
- `run_evaluations_newsinterview.py`: Script for performing systematic evaluations.

##  Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API Key (or compatible LLM provider)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sohu-mint-lab/MAJI-IJCNLP-AACL-2025.git
   cd MAJI-IJCNLP-AACL-2025
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   export API_KEY="your-openai-api-key"
   export BASE_URL="your-api-base-url" # Optional
   ```

## 🛠 Usage

### 1. Running the Generation Pipeline
To generate follow-up questions using the MAJI workflow:
```bash
python maji_newsinterview_pipeline.py --workers 2 --limit 10
```

### 2. Running Evaluations
To evaluate the generated responses against baselines:
```bash
python run_evaluations_newsinterview.py --workers 2
```

## 📖 Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{
anonymous2025maji,
title={{MAJI}: A Multi-Agent Workflow for Augmenting Journalistic Interviews},
author={Anonymous},
booktitle={Submitted to ACL Rolling Review - July 2025},
year={2025},
url={https://openreview.net/forum?id=lMFDNS5ZDi},
note={under review}
}
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
