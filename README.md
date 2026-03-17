\section*{Abstract}

While open-source language models perform well on general natural language processing tasks, they frequently struggle with the specialized vocabulary, complex reasoning, and unique document structures required in the legal and medical fields. Consequently, their reliability in these high-stakes domains is limited.
\\\\
This project explores how domain-specific supervised fine-tuning can effectively adapt open-source models to these specialized fields. A pretrained open-source model is evaluated both before and after undergoing supervised fine-tuning on medical and legal datasets.
The evaluation focuses on two core tasks—question answering (F1) and text generation (ROUGE) to assess both factual comprehension and generative proficiency within each specific domain.
\\\\
By utilizing standard quantitative metrics and controlled before-and-after comparisons, the project aims to; quantify the concrete benefits of domain adaptation, analyze any generalization trade-offs (e.g., whether improving specialized knowledge degrades general performance), and offer practical guidance on when supervised fine-tuning is a worthwhile strategy for specialized NLP applications.


\section*{Introduction}

Pretrained language models are foundational to modern natural language processing (NLP), demonstrating exceptional capability across a variety of general tasks due to their exposure to massive, diverse datasets. However, their performance often drops significantly when applied to specialized fields like law and medicine. These domains require an understanding of highly precise terminology, rigid document structures, and nuanced, domain-specific reasoning that general training fails to capture.
\\\\
Domain-specific fine-tuning—continuing a model's training on specialized data—is a proven method for bridging this gap, helping models align with niche vocabularies and contexts. Despite its popularity, there is a lack of systematic evaluation regarding how well fine-tuning actually works across varying domains and tasks. This is especially true for open-source models operating under realistic computational limits.
\\\\
To address this gap, this project conducts an empirical evaluation of supervised, domain-specific fine-tuning on an open-source language model using medical and legal datasets. The study measures performance on question answering and text generation. The model's capabilities are rigorously tested and compared both before and after fine-tuning.Using controlled environments and quantitative metrics, the project aims to; precisely quantify performance gains, analyze the broader effects of domain specialization, and deliver data-driven insights for effectively deploying fine-tuned models in specialized NLP applications.


\section*{Problem statement}

While pretrained open-source language models excel across general natural language processing tasks, they frequently struggle with the complex terminology, nuanced reasoning requirements, and rigid document structures inherent to specialized fields like law and medicine. Because these unique linguistic characteristics are not adequately captured in general training data, unadapted models risk producing inaccurate or unreliable outputs in these high-stakes areas.
\\\\
Supervised fine-tuning is the standard method used to align general models with these niche domains. However, despite its widespread practical use, its actual impact on downstream task performance is rarely evaluated systematically. Furthermore, critical trade-offs involving over-specialization, loss of general knowledge, and training efficiency often go unmeasured, particularly for open-source models operating under realistic computational constraints.
\\\\
To address this research gap, this experiment empirically investigates the efficacy of supervised fine-tuning by evaluating an open-source model before and after adaptation on legal and medical datasets. By assessing the model's performance specifically in question answering and text generation, the experiment aims to accurately quantify domain-specific gains, analyze cross-domain generalization effects, and deliver actionable insights into optimizing open language models for specialized applications.


\section*{Hypothesis}
Pretrained general-purpose language models are optimized for broad linguistic coverage but are not inherently designed to perform well on specialized domains such as legal and medical text. These domains require precise terminology, structured reasoning, and adherence to domain-specific conventions that are often underrepresented in general training corpora. As a result, general-purpose models are expected to exhibit lower performance on domain-specific question answering and text generation tasks when used without adaptation.
\\\\
This project hypothesizes that applying supervised domain-specific fine-tuning techniques will significantly improve model performance on specialized tasks. In particular, instruction-based supervised fine-tuning using parameter-efficient approaches such as Low-Rank Adaptation (LoRA) is expected to enhance task alignment by allowing the model to adjust its representations based on domain-relevant examples. These techniques enable targeted adaptation while minimizing computational cost, making them suitable for practical deployment on small open-source models.
\\\\
Furthermore, the project hypothesizes that effective fine-tuning can substantially reduce the performance gap between general-purpose pretrained models and domain-specific task requirements. By aligning the model’s learned representations with legal and medical data distributions, fine-tuned models are expected to demonstrate higher accuracy in question answering and improved quality in text generation compared to their baseline counterparts. The experiments are designed to empirically validate these hypotheses through controlled before-and-after performance comparisons.

\section*{Used Models   }
\begin{itemize}
    \item Phi-3-mini-4k-instruct
    \item llama-3-8b-Instruct
    \item mistral-7b-instruct-v0.3
\end{itemize}


\section*{Related Works}
The introduction of the Transformer architecture by Vaswani et al. fundamentally transformed sequence modeling through its exclusive use of self-attention mechanisms. Building upon this foundation, subsequent research has consistently proven that adapting Transformer-based models to specific disciplines yields substantial performance gains.
\\\\
In healthcare, for instance, models like BioBERT, ClinicalBERT, and PubMedBERT have demonstrated that fine-tuning on biomedical data significantly improves capabilities in inference and question answering tasks. Likewise, within the legal sector, models such as Legal-BERT and CaseLaw-BERT successfully tailor Transformer representations to navigate complex, highly structured legal documents, achieving superior performance on industry-specific benchmarks.
\\\\
Together, these foundational advancements highlight the clear need for a rigorous empirical evaluation of supervised fine-tuning strategies across diverse domains and tasks—a research gap that this experiment is specifically designed to address.

\textbf{Attention Is All You Need}\\
Advances in Neural Information Processing Systems (NeurIPS)\\
Foundational paper introducing the Transformer architecture.\cite{vaswani2017attention}
\\\\
\textbf{BioBERT: a pre-trained biomedical language representation model for biomedical text mining}\\
Bioinformatics\\
Seminal work demonstrating domain adaptation of Transformers for medical text.\cite{lee2020biobert}
\\\\
\textbf{Domain-Specific Language Model Pretraining for Biomedical NLP}\\
Proceedings of the ACL\\
Shows benefits of domain-specific training for biomedical NLP tasks\cite{gu2021domain}
\\\\
\textbf{LEGAL-BERT: The Muppets straight out of Law School}\\
Proceedings of EMNLP\\
Canonical Transformer-based model for legal domain adaptation.\cite{chalkidis2020legalbert}
\\\\
\textbf{Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks}\\
Proceedings of ACL\\
Provides strong empirical evidence for continued pretraining and fine-tuning.\cite{gururangan2020dont}
\\\\
\textbf{PubMedQA: A Dataset for Biomedical Research Question Answering}\\
Proceedings of EMNLP-IJCNLP\\
Widely used benchmark for evaluating medical question answering.\cite{jin2019pubmedqa}
\\\\

\section*{Proposed Solution}
The solution implements a controlled experimental approach to evaluate the effectiveness of domain-specific supervised fine-tuning for open-source language models. A pretrained open model is selected as the baseline and evaluated on legal(LegalQAEval) and medical datasets (PubMedQA, MedQuAD) to establish initial performance on question answering (F1) and text generation (ROUGE) tasks. This baseline assessment provides a reference point for measuring the impact of domain adaptation.
\\\\
The model is then fine-tuned separately on domain-specific corpora for the legal and medical domains using supervised fine-tuning techniques LoRA and QLoRA. Fine-tuning is performed under consistent architectural and training configurations to isolate the effect of domain-specific data. This process allows the model to adapt its internal representations to specialized vocabulary, structure, and reasoning patterns present in each domain.
\\\\
Performance after fine-tuning is evaluated using standardized quantitative metrics and compared directly against baseline results. By analyzing before-and-after performance across tasks and domains, the solution quantifies the benefits of fine-tuning while also examining trade-offs related to generalization and training cost. The results provide practical insight into the effectiveness of domain adaptation for open language models in specialized applications.

\section*{Experiments}
This project implements a supervised fine-tuning strategy centered on instruction-based learning combined with parameter-efficient fine-tuning (PEFT) techniques. Specifically, Low-Rank Adaptation (LoRA) adapters are employed to fine-tune open-source Transformer-based language models for domain-specific tasks in the legal and medical domains. Instruction-based fine-tuning reformulates training data into instruction–response pairs, enabling the model to learn how to generate task-appropriate outputs rather than merely predicting the next token. This approach is well-suited for both question answering and text generation tasks, while LoRA ensures that the computational overhead remains manageable by updating only a small subset of model parameters within the attention layers.
\\\\
The experimental design begins with a baseline evaluation to establish the performance of pretrained models prior to any domain adaptation. In this experiment, models are evaluated out-of-the-box on legal and medical question answering datasets, as well as on a fixed set of domain-specific text generation prompts. Performance is measured using standard metrics such as accuracy and F1 score for question answering, along with automated or lightweight human evaluation for text generation quality. This baseline experiment provides a critical reference point for quantifying the impact of domain-specific fine-tuning and highlights inherent limitations of general-purpose pretrained models when applied to specialized domains.
\\\\
Subsequent experiments focus on domain-specific fine-tuning and in-domain evaluation. Separate fine-tuning runs are conducted for the legal and medical domains using instruction-based supervised learning with LoRA adapters. After fine-tuning, each model is evaluated on corresponding in-domain tasks to measure performance gains relative to the baseline. Legal-domain experiments assess improvements in legal question answering accuracy and legal text generation quality, while medical-domain experiments mirror this setup for medical question answering and generation tasks. Together, these experiments enable a controlled empirical comparison of how supervised fine-tuning reshapes model behavior across domains, providing insights into domain specialization, performance improvements, and trade-offs associated with efficient adaptation techniques.
\\\\
\textbf{Instruction-Based Supervised Fine-Tuning}\\
Objective:\\
To enable the language model to learn task-oriented response behavior by explicitly training it to follow instructions and generate appropriate answers for domain-specific question answering and text generation tasks.
\\\\
Method:\\
Training data is reformatted into an instruction–response structure, where each example consists of an instruction or question paired with a corresponding target answer. This format encourages the model to learn how to respond to prompts rather than simply predicting the next token in a sequence. Instruction-based supervised fine-tuning is applied consistently across legal and medical datasets to support both question answering and text generation tasks.
\\\\
Purpose:\\
This approach improves task alignment by teaching the model how to generate structured, context-aware responses that are appropriate for domain-specific applications. Instruction-based fine-tuning has been shown to be particularly effective for improving performance on interactive tasks such as question answering and controlled text generation.\\
\\\\
\textbf{Parameter-Efficient Fine-Tuning Using LoRA}\\
Objective:\\
To adapt pretrained language models to specialized domains while minimizing computational cost and reducing the risk of overfitting.
\\\\
Method:\\
Parameter-efficient fine-tuning (PEFT) is implemented using Low-Rank Adaptation (LoRA). Instead of updating all model parameters, LoRA injects low-rank adapter matrices into selected attention layers of the Transformer architecture. During training, only these adapter parameters are updated, while the original model weights remain frozen.
\\\\
Purpose:\\
This technique enables efficient domain adaptation by significantly reducing memory usage and training time, making fine-tuning feasible for small open-source models and limited computational resources. LoRA is widely used and well-validated in academic research, providing a robust and reproducible mechanism for domain-specific supervised fine-tuning.


\subsection*{Baseline Evaluation (Before Fine-Tuning)}
Objective:\\
To understand how pretrained open-source language models perform on legal and medical tasks before any domain-specific training is applied.\\\\
Method:\\
Each selected model is evaluated in its original, pretrained form on the following tasks:
\begin{itemize}
    \item Legal question answering datasets
    \item Medical question answering datasets
    \item A fixed set of domain-specific text generation prompts
\end{itemize}
No additional training or adaptation is performed during this stage.\\\\
Evaluation Metrics:\\
\begin{itemize}
    \item Question answering accuracy and F1 score
    \item Text generation quality, measured using automatic similarity-based metrics and supplemented with limited human judgment for correctness and clarity
\end{itemize}
Purpose:\\
This experiment establishes a baseline reference point and highlights inherent differences in domain knowledge and reasoning ability across models prior to fine-tuning. These results are used for comparison in all subsequent experiments.

\subsection*{Legal Domain Fine-Tuning and In-Domain Evaluation}
Objective:\\
To evaluate the effect of legal domain fine-tuning on legal question answering and text generation performance.\\\\
Method:\\
Each model is fine-tuned on legal domain training data using instruction-based supervised fine-tuning with LoRA adapters. The adapted models are then evaluated on legal QA datasets and legal text generation prompts that were not seen during training.\\\\
Evaluation Metrics:\\
\begin{itemize}
    \item Legal QA accuracy and F1 score
    \item Legal text generation quality
\end{itemize}
Purpose:\\
This experiment measures the extent to which legal domain fine-tuning improves in-domain performance relative to the baseline established in Experiment 1.

\subsection*{Medical Domain Fine-Tuning and In-Domain Evaluation}
Objective:\\
To evaluate the effect of medical domain fine-tuning on medical question answering and text generation performance.\\\\
Method:\\
Each model is fine-tuned on medical domain training data using the same instruction-based fine-tuning approach. The resulting models are evaluated on medical QA datasets and medical text generation prompts.\\\\
Evaluation Metrics:\\
\begin{itemize}
    \item Medical QA accuracy and F1 score
    \item Medical text generation quality
\end{itemize}
Purpose:\\
This experiment mirrors Experiment 2 for the medical domain, enabling a direct comparison of how domain-specific adaptation affects performance across different specialized domains.

\subsection*{Cross-Domain Generalization Analysis}
Objective:\\
To examine whether fine-tuning on one domain improves specialization at the cost of performance in another domain.\\\\
Method:\\
\begin{itemize}
    \item Models fine-tuned on legal data are evaluated on medical QA datasets
    \item Models fine-tuned on medical data are evaluated on legal QA datasets
\end{itemize}
The results are compared against baseline performance from Experiment 1.\\\\
Evaluation Metrics:\\
Cross-domain QA accuracy and F1 score relative to baseline\\\\
Purpose:\\
This experiment investigates potential specialization trade-offs and domain interference introduced by fine-tuning, providing insight into how domain adaptation affects generalization.

\subsection*{Quantization Effects and Resource Efficiency (Standard LoRA vs. QLoRA)}
Objective:\\
To examine the trade-off between computational resource efficiency and downstream task performance when applying 4-bit quantization (QLoRA) compared to standard 16-bit LoRA fine-tuning.\\\\
Method:\\
We will fine-tune a large language model on the selected domain-specific corpus using two configurations:
\begin{itemize}
    \item Baseline (Standard LoRA): The base model is loaded in 16-bit precision (FP16 or BF16), and LoRA adapters are trained under standard precision settings.
    \item Experimental (QLoRA): The base model is loaded in 4-bit precision using the NormalFloat4 (NF4) data type with double quantization. LoRA adapters are trained in 16-bit precision while the quantized base model weights remain frozen.
\end{itemize}
During training, we will record peak GPU memory usage and measure the final adapter size.\\\\
Evaluation Metrics:
\begin{itemize}
    \item Performance Metrics: Accuracy and F1 score on the domain validation set to assess potential degradation due to quantization
\end{itemize}
Purpose:\\
This experiment evaluates whether QLoRA enables efficient domain adaptation of large language models on limited hardware without significant performance degradation. The results will help determine whether 4-bit quantization provides a practical pathway for deploying domain-adapted models on consumer-grade GPUs while maintaining competitive task accuracy.

\section*{Experiment Set-Up and Data-Set Details}
This project uses open-source, domain-specific datasets to evaluate the effects of supervised fine-tuning on open language models in the legal and medical domains. All datasets are publicly available, well-documented, and commonly used in academic research, making them suitable for systematic evaluation and reproducibility. The selected datasets include both long-form domain text and structured question–answer pairs, allowing the study to assess model performance across different task types.\\\\
To ensure consistency across experiments, datasets are preprocessed using standardized formatting and prompt structures. Training and evaluation splits provided by the datasets are used wherever available to avoid data leakage. All datasets are used strictly for research purposes.
\\\\
\textbf{Legal Domain Datasets:}
\\\\
To support evaluation on legal question answering tasks, the project additionally leverages LegalQAEval, an open dataset hosted on Hugging Face. This dataset provides question–answer pairs grounded in legal content, enabling standardized evaluation of model performance before and after fine-tuning.
\\\\
Legal dataset sources:
\\
\begin{itemize}
    \item LegalQAEval (Hugging Face): https://huggingface.co/datasets/isaacus/LegalQAEval\\\\
\end{itemize}
\textbf{Medical Domain Datasets:}
\\
For the medical domain, the project uses PubMedQA, accessed via the BigBio collection on Hugging Face. PubMedQA consists of biomedical research questions paired with supporting PubMed abstracts and labeled answers. This dataset is widely used for evaluating medical question answering systems and provides a reliable benchmark for domain-specific adaptation.\\
In addition, MedQuAD (Medical Question Answering Dataset) is used to supplement the medical evaluation. MedQuAD contains curated medical question–answer pairs sourced from trusted medical websites, offering diverse coverage of clinical and biomedical topics.\\\\
Medical dataset sources:\\

\begin{itemize}
    \item PubMedQA (BigBio): \url{https://huggingface.co/datasets/bigbio/pubmed_qa}\\
    \item MedQuAD (Hugging Face): \url{https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset}\\\\
\end{itemize}


\section*{Evaluation Metrics}
The evaluation framework is designed to quantitatively assess the impact of domain adaptation by comparing model performance before and after fine-tuning. The chosen metrics rigorously test for correctness, relevance, and clarity across two primary tasks: question answering and text generation.
\\\\
For question-answering tasks in both domains, performance is measured using F1 score. F1 score evaluates partial correctness, balancing precision and recall to account for answers that are semantically accurate but vary in phrasing or overlap.
\\\\
For text generation tasks, the experiment employs automated evaluation metrics such as ROUGE. ROUGE measures n-gram overlap against reference texts, making it highly effective for assessing content coverage and relevance in summarization or explanation tasks. 


\subsection*{Evaluation Metrics Summary}
\begin{center}
\begin{tabular}{ c c }
 Task & Metric  \\ 
 Legal QA & F1 \\  
 Medical QA & F1 \\
 Text Generation & ROUGE
\end{tabular}
\end{center}

\subsection*{Metric Justification}
Accuracy is used as a primary evaluation metric for question answering tasks in both the legal and medical domains. It provides a straightforward measure of the proportion of correctly answered questions and is particularly suitable for classification-style or short-answer QA tasks. Accuracy enables clear before-and-after comparison and allows direct measurement of performance gains resulting from domain-specific fine-tuning.
\\\\
F1 score complements accuracy by capturing partial correctness in cases where predicted answers may not exactly match ground-truth responses but still contain relevant information. This is especially important for legal and medical question answering, where responses may vary in phrasing while remaining semantically correct. The F1 score provides a more nuanced evaluation of model performance by balancing precision and recall.
\\\\
ROUGE is employed to evaluate text generation quality by measuring n-gram overlap between generated outputs and reference texts. It is widely used in summarization and explanation tasks and serves as an effective indicator of content relevance and coverage. ROUGE allows systematic comparison of generation quality before and after fine-tuning in domain-specific contexts.
\\\\
BLEU is optionally used as a supplementary metric for text generation tasks that involve more structured outputs. BLEU focuses on precision-based n-gram matching and is useful when evaluating shorter or more formulaic generated responses. While BLEU alone may not fully capture semantic quality, it provides additional quantitative insight when used alongside ROUGE.
\\\\
\\\\
\\\\
\\\\
\section*{Results}
\subsection*{QLora}
Llama
\\\\
\includegraphics[scale = .2]{LlamaQloraPubmedComparison.png}
\includegraphics[scale = .2]{LlamaQloraPubmedImprovement.png}
\includegraphics[scale = .2]{LlamaQloraMedQuadComparison.png}
\includegraphics[scale = .2]{LlamaQloraMedquadImprovement.png}
\includegraphics[scale = .2]{LlamaQloraLegalComparison.png}
\includegraphics[scale = .2]{LlamaQloraLegalImprovement.png}
Mistral
\\\\
\includegraphics[scale = .2]{MistralQloraPubmedComparison.png}
\includegraphics[scale = .2]{MistralQloraPubmedImprovement.png}
\includegraphics[scale = .2]{MistralQloraMedQuadComparison.png}
\includegraphics[scale = .2]{MistralQloraMedquadImprovement.png}
\includegraphics[scale = .2]{MistralQloraLegalComparison.png}
\includegraphics[scale = .2]{MistralQloraLegalImprovement.png}
Phi
\\\\
\includegraphics[scale = .2]{PhiQloraPubmedComparison.png}
\includegraphics[scale = .2]{PhiQloraPubmedImprovement.png}
\includegraphics[scale = .2]{PhiQloraMedQuadComparison.png}
\includegraphics[scale = .2]{PhiQloraMedquadImprovement.png}
\includegraphics[scale = .2]{PhiQloraLegalComparison.png}
\includegraphics[scale = .2]{PhiQloraLegalImprovement.png}

\subsection*{Lora}
Llama
\includegraphics[scale = .25]{LlamaLoraPubmedComparison.png}
\includegraphics[scale = .2]{LlamaLoraPubmedImprovement.png}
\includegraphics[scale = .2]{LlamaLoraMedQuadComparison.png}
\includegraphics[scale = .2]{LlamaLoraMedquadImprovement.png}
\includegraphics[scale = .2]{LlamaLoraLegalComparison.png}
\includegraphics[scale = .2]{LlamaLoraLegalImprovement.png}
Mistral
\\\\
\includegraphics[scale = .2]{MistralLoraPubmedComparison.png}
\includegraphics[scale = .2]{MistralLoraPubmedImprovement.png}
\includegraphics[scale = .2]{MistralLoraMedQuadComparison.png}
\includegraphics[scale = .2]{MistralLoraMedquadImprovement.png}
\includegraphics[scale = .2]{MistralLoraLegalComparison.png}
\includegraphics[scale = .2]{MistralLoraLegalImprovement.png}
Phi
\\\\
\includegraphics[scale = .2]{PhiLoraPubmedComparison.png}
\includegraphics[scale = .2]{PhiLoraPubmedImprovement.png}
\includegraphics[scale = .2]{PhiLoraMedQuadComparison.png}
\includegraphics[scale = .2]{PhiLoraMedquadImprovement.png}
\includegraphics[scale = .2]{PhiLoraLegalComparison.png}
\includegraphics[scale = .2]{PhiLoraLegalImprovement.png}
\subsection*{RAM Usage}
QLoRA vs LoRA
\\\\
\includegraphics[scale = .6]{QLoRARAM.png}
\includegraphics[scale = .6]{LoRARAM.png}
\section*{Limitations}
While this project provides valuable empirical evidence for the efficacy of domain adaptation, several limitations should be considered when interpreting the results:
\begin{itemize}
    \item \textbf{Computational Constraints:} The study was specifically designed to operate under realistic computational limits using small open-source models (Phi, Llama, and Mistral). While parameter-efficient techniques like LoRA and QLoRA made this possible, they may not fully capture the performance potential achievable with full-parameter fine-tuning on larger-scale hardware. Compute costs limit what can be done in the experiment.
    Domain Specialization Trade-offs: Initial findings suggest that as models become highly specialized in a niche field (e.g., legal or medical), there is a risk of "catastrophic forgetting" or degradation in general-purpose knowledge. This study focused primarily on in-domain gains, and a more exhaustive analysis of general knowledge loss remains an area for further investigation.
    \item \textbf{Metric Constraints:} The evaluation relied heavily on automated metrics like F1 and ROUGE. While these are standard for quantifying factual comprehension and content overlap, they may not fully account for the clinical accuracy or legal nuance that a human expert would identify.
    \item \textbf{Data Scope:} The experiments were conducted using specific datasets (LegalQAEval, PubMedQA, and MedQuAD). While these are high-quality benchmarks, they represent only a subset of the vast linguistic diversity found across the entire legal and medical professions.
    \item \textbf{Quantization Effects:} Although QLoRA (4-bit quantization) was utilized to maintain resource efficiency, any form of quantization inherently involves a trade-off in precision. The extent to which this precision loss affects extremely complex, multi-step reasoning in high-stakes environments requires deeper exploration.

\end{itemize}
\section*{Conclusion}
This project conducted a rigorous empirical evaluation of domain adaptation in open-source language models, specifically focusing on the high-stakes legal and medical fields. Our research addressed the critical gap in systematic assessment regarding the efficacy of supervised fine-tuning (SFT) under realistic computational constraints.
\\\\
By implementing instruction-based supervised fine-tuning with parameter-efficient techniques like \textbf{LoRA} and \textbf{QLoRA}, we successfully demonstrated that general-purpose models (Phi, Llama, and Mistral) can be effectively aligned with specialized domains. The experiments validated our primary hypothesis: that domain-specific adaptation significantly reduces the performance gap between general-purpose pretraining and the nuanced requirements of specialized tasks.
\textbf{Key Takeaways:}
\begin{itemize}
    \item \textbf{Performance Gains:} Across both medical (PubMedQA, MedQuAD) and legal (LegalQAEval) datasets, the fine-tuned models showed measurable improvements in both F1 scores for question answering and ROUGE scores for text generation.
    \item \textbf{Efficiency:} The use of QLoRA (4-bit quantization) proved to be a practical pathway for achieving high-performance adaptation on consumer-grade hardware without significant degradation in task accuracy. LoRA generally uses 66\% more compute than QLoRA with similar results . A PEFT training routing with QLoRA will use ~5g of VRAM and the same routine with LoRA will use ~16g VRAM.
    \item \textbf{Specialization vs. Generalization:} The cross-domain analysis provided critical insights into the trade-offs of specialization, highlighting how targeted training reshapes model behavior while navigating the challenges of domain interference.
\end{itemize}
In summary, this study confirms that supervised fine-tuning is a worthwhile and essential strategy for deploying open-source models in specialized NLP applications. The data-driven insights gathered here offer a practical guide for researchers and practitioners looking to optimize large language models for complex, domain-specific reasoning tasks.
\section*{Future Work}
Several directions can extend the findings of this project. 
\begin{itemize}
    \item Extending experiments to additional open-source architectures such as LLaMA-3.2, Qwen-2.5, and Mistral at different parameter scales would allow cross-architecture comparisons and test whether larger models benefit more from LoRA adaptation. 
    \item Systematically varying the LoRA rank (r = 4, 8, 16, 32) would help identify the optimal trade-off between adapter capacity and overfitting risk on small datasets. 
    \item Comparing LoRA against alternative parameter-efficient methods such as AdaLoRA and IA3 would determine whether adaptive rank allocation or learned scaling vectors offer advantages for domain adaptation. 
    \item Investigating how performance scales with training set size using the larger PubMedQA artificial subset (211K samples) and the full MedQuAD corpus (47K samples) would clarify whether the current gains are limited by data volume. 
    \item Conducting manual expert review of model outputs to assess factual accuracy and hallucination rates would address a key limitation of relying solely on automated metrics.
\end{itemize}

\section*{GitHub}
\hyperlink{}{https://github.com/NateyLB/CMPE252FinalProject.git}

\section*{Contributions}
Gunanidhi Ramakrishnan: Domain and dataset selection, dataset preprocessing, baseline evaluation (Experiment 1), LoRA fine-tuning on MedQUAD (Experiment 2), and results visualization.
\\\\
Nathan Howland: LoRA fine-tuning on LegalQAEval (Experiment 3), cross-domain generalization analysis (Experiment 4), LoRA vs QLoRA comparison (Experiment 5), and final report preparation.
\\\\
Ravikumar Komandur Narayanan: Literature review, experiment design, LoRA fine-tuning on PubMedQA (Experiment 2), training configuration, and results analysis.
