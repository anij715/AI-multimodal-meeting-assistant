# AI-multimodal-meeting-assistant: A Multimodal Approach to Insight-Driven Meeting Summarization

This repository contains the data corpus and evaluation benchmark from the Master's thesis, "A Multimodal Approach to Insight-Driven Meeting Summarization: Leveraging Speech and Visual Feedback for Enhanced Data Insight Capture," completed by Rizul Sharma at the University of Cincinnati.

## Project Overview
This research explores the most effective methods for an AI agent to comprehend spoken conversations about data visualizations in meeting environments. The core contributions of this work are a novel multi-modal conversational corpus and a diagnostic evaluation framework designed to test an AI's understanding of these discussions. 

This repository makes two key components available to the research community:

* The Multi-Modal Conversational Corpus: A collection of 72 transcript-chart pairs from mock online meeting sessions where participants discussed data visualizations.
* The Diagnostic Evaluation Benchmark: A set of 318 questions designed to assess an AI's comprehension of the conversations in the corpus, based on a dual-axis framework inspired by Bloom's Taxonomy. 

## The Multi-Modal Conversational Corpus
To facilitate research into AI-driven analysis of collaborative data discussions, we provide a unique corpus with the following features:

* Content: The corpus consists of 72 unique pairs, each containing:
  * A transcript segment from a spoken conversation between two participants discussing a data visualization.
  * The corresponding data visualization image (as a PNG file).
  * The Python source code used to generate the visualization. 


* Origin: The data was collected from a study involving 18 participants in 9 paired sessions, discussing a series of 8 charts related to a movies dataset. The sessions were designed to mimic real-world online meetings and encourage organic, unscripted dialogue. 


* Intentional Discrepancies: To elicit a range of authentic reactions, some visualizations were created with intentional discrepancies, such as mismatched titles, hidden logarithmic scales, or data processing nuances. 

## The Diagnostic Evaluation Benchmark
This benchmark provides a robust method for evaluating an AI's comprehension of the corpus data.


* Structure: It contains 318 questions, with each question corresponding to a specific transcript-chart pair. 

* Dual-Axis Framework: The questions are labeled along two axes:
  * Comprehension Level: Categorizes questions by the cognitive depth required, from foundational recall to deep analytical reasoning.
  * Area of Weakness: Pinpoints specific topics of discussion, such as "Chart Layout," "Data Provenance," or "Participant Hypothesis," to diagnose model failures. 

This framework moves beyond simple accuracy scores to provide a diagnostic understanding of a model's performance, identifying not just if a model fails, but why. 

## How to Use This Repository
The data and benchmark questions in this repository can be used to:

* Train and evaluate AI agents designed for meeting summarization and data analysis.
* Benchmark the performance of various Large Language Models (LLMs) and Vision Language Models (VLMs) on their ability to understand multi-modal conversational data.
* Conduct further research into challenges like knowledge conflict in hybrid models and the grounding of spoken language to visual data. 

The corpus is organized into transcript-chart pairs, and the benchmark questions are provided with their corresponding labels and correct answers. You can find the Python code for the 4-pipeline AI agent used in the original thesis in this repository. 

## Citation
If you use the corpus or the evaluation benchmark in your research, please cite this GitHub repo along with the following thesis:

```latex
@mastersthesis{sharma2025multimodal,
  title={A Multimodal Approach to Insight-Driven Meeting Summarization: Leveraging Speech and Visual Feedback for Enhanced Data Insight Capture},
  author={Sharma, Rizul},
  school={University of Cincinnati},
  year={2025},
  month={July}
}
```
By citing this work, you acknowledge the effort involved in creating and validating these resources and help support the dissemination of this research. We hope this dataset and benchmark will be a valuable contribution to the community. 
