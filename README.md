<!--
 * @Author: zihao zihao-lee@outlook.com
 * @Date: 2024-06-24 22:03:09
 * @LastEditors: zihao zihao-lee@outlook.com
 * @LastEditTime: 2024-06-24 22:20:58
 * @FilePath: \MT-Distilation\README.md
 * @Description: 
 * 
 * Copyright (c) 2024 by zihao, All Rights Reserved. 
-->
# Exploring Indirect Knowledge Transfer in Multilingual Machine Translation Through Targeted Distillation

Course project of LDA-T313 Approaches to Natural Language Understanding

Zihao Li & Chao Wang

### Overview
This repository contains the implementation of the project "Exploring Indirect Knowledge Transfer in Multilingual Machine Translation Through Targeted Distillation". The project aims to investigate the efficiency of cross-linguistic knowledge transfer in multilingual Neural Machine Translation (NMT) using knowledge distillation techniques.

### Objectives
The study focuses on two main objectives:
1. **Cross-Linguistic Knowledge Transfer**: Evaluate how effectively student models trained on one language perform in translating other related languages within the same language family.
2. **Correlation of Language Similarity with Transfer Effectiveness**: Investigate whether the effectiveness of cross-linguistic knowledge transfer correlates with the degree of linguistic similarity among languages.


### Methodology
#### Teacher Models
We utilize two pre-trained multilingual NMT models from the Helsinki-NLP OPUS-MT project:
- `opus-mt-tc-big-gmq-en`: Translates from Danish, Norwegian, Swedish to English.
- `opus-mt-tc-big-zle-en`: Translates from Belarusian, Russian, Ukrainian to English.
#### Training Datasets
Our training datasets are derived from the NLLB corpus, filtered for quality using Opusfilter. Each dataset contains 5 million parallel sentences for each language pair involving English.
#### Distillation Process
The distillation process uses the outputs of pre-trained teacher models as the target translations for training smaller student models.
```bash
train.sh
```
#### Model Configurations
| Parameter                        | Teacher | Student |
|----------------------------------|---------|---------|
| Embedding dimension              | 1024    | 256     |
| Attention heads                  | 16      | 8       |
| Feed forward network dimension   | 4096    | 2048    |
| Hidden layers                    | 6       | 3       |

#### Evaluation Metrics
The models are evaluated using BLEU and COMET metrics to measure translation accuracy and fluency.

#### Testing Datasets
We used the Tatoeba Translation Challenge and FLORES-200 datasets for evaluating the student models.

#### Key Findings
1. **General Performance**: Student models show reduced performance when translating languages they were not directly trained on, with a pronounced decline in the East Slavic languages.
2. **Lexical Similarity Impact**: Models trained on languages with closer lexical ties to the target language demonstrated enhanced translation accuracy, particularly evident in the North Germanic languages.