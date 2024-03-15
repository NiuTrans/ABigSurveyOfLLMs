 # A Survey of LLM Surveys

Large language models (LLMs) are making sweeping advances across many fields of artificial intelligence. As a result, research interest and progress in LLMs have exploded. There are now hundreds of research papers on LLMs published in various conferences or posted to open-access archives every day. Given the significant growth in LLM-related papers, this work compiles surveys on LLMs to provide a comprehensive overview of the field. Most of these surveys have been published or posted in the past few years, so this collection is relatively new. We hope that our compilation can be helpful for people who want to get a quick understanding of the field.

<!-- :new: We add a NEW category of large language models! [<a href="#large-language-models">Large Language Models</a>] -->

## Outline
<!-- + Large Language Modeling
    + <a href="#alignment">Alignment</a>
    + <a href="#data">Data</a>
    + <a href="#evaluation">Evaluation</a>
    + <a href="#societal-implications">Societal Implications</a>
    + <a href="#safety">Safety</a>
    + <a href="#science-of-LMs">Science of LMs</a>
    + <a href="#compute-efficient-lms">Compute Efficient LMs</a>
    + <a href="#engineering-for-large-lms">Engineering for Large LMs</a>
    + <a href="#learning-algorithms">Learning Algorithms</a>
    + <a href="#inference-algorithms">Inference Algorithms</a>
    + <a href="#human-mind-brain-philosophy-laws-and-LMs">Human Mind, Brain, Philosophy, Laws and LMs</a>
    + <a href="#lms-for-everyone">LMs for Everyone</a>
    + <a href="#lms-and-the-world">LMs and The world</a>
    + <a href="#lms-and-embodiment">LMs and Embodiment</a>
    + <a href="#lms-and-interactions">LMs and Interactions</a>
    + <a href="#lms-with-tools-and-code">LMs with Tools and Code</a>
    + <a href="#lms-on-diverse-modalities-and-novel-applications">LMs on Diverse Modalities and Novel Applications</a> -->

+ [General Surveys](#section1)
+ [Transformers](#section2)
+ [Alignment](#section3)
+ [Prompt Learning](#section4)
	+ [In-context Learning](#section5)
	+ [Chain of Thought](#section6)
	+ [Prompt Engineering](#section7)
	+ [Reasoning](#section8)
+ [Data](#section9)
+ [Evaluation](#section10)
+ [Societal Issues](#section11)
+ [Safety](#section12)
	+ [Source Detection](#section13)
	+ [Security](#section14)
+ [Misinformation](#section15)
	+ [Hallucinations](#section16)
	+ [Factuality](#section17)
+ [Attributes of LLMs](#section18)
+ [Efficient LLMs](#section19)
+ [Learning Methods for LLMs](#section20)
+ [Multimodal LLMs](#section21)
+ [Knowledge Based LLMs](#section22)
	+ [Retrieval-Augmented LLMs](#section23)
	+ [Knowledge Editing](#section24)
+ [Extension of LLMs](#section25)
	+ [LLMs with Tools](#section26)
	+ [LLMs and Interactions](#section27)
+ [Long Sequence LLMs](#section28)
+ [LLMs Applications](#section29)
	+ [Education](#section30)
	+ [Law](#section31)
	+ [Healthcare](#section32)
	+ [Games](#section33)
	+ [NLP Tasks](#section34)
	+ [Software Engineering](#section35)
	+ [Recommender Systems](#section36)
	+ [Graphs](#section37)
	+ [Other](#section38)



<!-- To reduce class imbalance, we separate some of the hot sub-topics from the original categorization of ACL and ICML submissions. E.g., Named Entity Recognition is a first-level area in our categorization because it is the focus of several surveys. -->

<!-- ## Statistics

We show the number of paper in each area in Figures 1-2.

<p align="center"><img src="https://s2.loli.net/2023/05/26/DUa43miWf5NFlZx.png" width="70%" height="70%"/></p>

<p align="center">Figure 1: # of papers in each NLP area.</p>

Also, we plot paper number as a function of publication year (see Figure 3).

<p align="center"><img src="https://s2.loli.net/2023/05/26/7tMmcRO1lK9N5hF.png" width="70%" height="70%"/></p>

<p align="center">Figure 3: # of papers vs publication year.</p>

In addition, we generate word clouds to show hot topics in these surveys (see Figures 4-5).

<p align="center"><img src="https://s2.loli.net/2023/05/26/6RqNCKBwsEZtA3H.png" width="60%" height="60%"/></p>

<p align="center">Figure 4: The word cloud for NLP.</p> -->

<!-- <details><summary>Definitions</summary>

- **Training Speed**: the number of training samples processed per second during the training. (bs=4, cutoff_len=1024)
- **Rouge Score**: Rouge-2 score on the development set of the [advertising text generation](https://aclanthology.org/D19-1321.pdf) task. (bs=4, cutoff_len=1024)
- **GPU Memory**: Peak GPU memory usage in 4-bit quantized training. (bs=1, cutoff_len=1024)
- We adopt `pre_seq_len=128` for ChatGLM's P-Tuning and `lora_rank=32` for LLaMA-Factory's LoRA tuning.

| Version                  | Time       | Update Content                                               |
| ------------------------ | ---------- | ------------------------------------------------------------ |
| V1                       | 2023/03/31 | The initial version.                                         |
</details> -->

## Survey List

#### General Surveys<a id="section1"></a>

- **Large Language Models: A Survey**, arXiv 2024.02 [[Paper](https://arxiv.org/abs/2402.06196)]

- **A Comprehensive Survey of AI-Generated Content (AIGC): A History of Generative AI from GAN to ChatGPT**, arXiv 2023.03 [[Paper](https://arxiv.org/abs/2303.04226)]

- **A Survey of Large Language Models**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2303.18223)] [[GitHub](https://github.com/RUCAIBox/LLMSurvey)]

- **Challenges and Applications of Large Language Models**, arXiv 2023.07 [[Paper](https://arxiv.org/abs/2307.10169)]

- **Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond**, arXiv 2023.04 [[Paper](https://arxiv.org/abs/2304.13712)] [[GitHub](https://github.com/Mooler0410/LLMsPracticalGuide)]

- **A Survey on Large Language Models: Applications, Challenges, Limitations, and Practical Usage**, TechRxiv 2023.07 [[Paper](https://www.techrxiv.org/doi/full/10.36227/techrxiv.23589741.v1)] [[GitHub](https://github.com/anas-zafar/LLM-Survey)]

- **A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT**, arXiv 2023.05 [[Paper](https://arxiv.org/abs/2302.09419)]

#### Transformers<a id="section2"></a>

- **A survey of transformers**, arXiv 2022.10 [[Paper](https://www.sciencedirect.com/science/article/pii/S2666651022000146)]

- **Introduction to Transformers: an NLP Perspective**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.17633)]

- **Efficient Transformers: A Survey**, arXiv 2022.12 [[Paper](https://dl.acm.org/doi/full/10.1145/3530811)]

- **A Practical Survey on Faster and Lighter Transformers**, arXiv 2023.07 [[Paper](https://dl.acm.org/doi/abs/10.1145/3586074)]

- **Attention Mechanism, Transformers, BERT, and GPT: Tutorial and Survey**, arXiv 2020.12 [[Paper](https://osf.io/preprints/osf/m6gcn)]

#### Alignment<a id="section3"></a>

- **Bridging the Gap: A Survey on Integrating (Human) Feedback for Natural Language Generation**, arXiv 2023.06 [[Paper](https://arxiv.org/abs/2305.00955)]

- **AI Alignment: A Comprehensive Survey**, arXiv 2024.02 [[Paper](https://arxiv.org/abs/2310.19852)]

- **Large Language Model Alignment: A Survey**, arXiv 2023.09 [[Paper](https://arxiv.org/abs/2309.15025)]

- **From Instructions to Intrinsic Human Values -- A Survey of Alignment Goals for Big Models**, arXiv 2023.09 [[Paper](https://arxiv.org/abs/2308.12014)] [[GitHub](https://github.com/ValueCompass/Alignment-Goal-Survey)]

- **Aligning Large Language Models with Human: A Survey**, arXiv 2023.07 [[Paper](https://arxiv.org/abs/2307.12966)] [[GitHub](https://github.com/GaryYufei/AlignLLMHumanSurvey)]

- **Instruction Tuning for Large Language Models: A Survey**, arXiv 2023.08 [[Paper](https://arxiv.org/abs/2308.10792)]

- **A Comprehensive Survey on Instruction Following**, arXiv 2024.01 [[Paper](https://arxiv.org/abs/2303.10475v7)] [[GitHub](https://github.com/RenzeLou/awesome-instruction-learning)]

#### Prompt Learning<a id="section4"></a>

###### In-context Learning<a id="section5"></a>

- **A Practical Survey on Zero-shot Prompt Design for In-context Learning**, ranlp 2023.09 [[Paper](https://aclanthology.org/2023.ranlp-1.69/)]

- **A Survey on In-context Learning**, arXiv 2023.06 [[Paper](https://arxiv.org/abs/2301.00234)]

###### Chain of Thought<a id="section6"></a>

- **A Survey of Chain of Thought Reasoning: Advances, Frontiers and Future**, arXiv 2023.10 [[Paper](https://arxiv.org/abs/2309.15402)] [[GitHub](https://github.com/zchuz/CoT-Reasoning-Survey)]

- **Towards Better Chain-of-Thought Prompting Strategies: A Survey**, arXiv 2023.10 [[Paper](https://doi.org/10.48550/arXiv.2310.04959)]

- **Igniting Language Intelligence: The Hitchhiker's Guide From Chain-of-Thought Reasoning to Language Agents**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.11797)] [[GitHub](https://github.com/Zoeyyao27/CoT-Igniting-Agent)]

###### Prompt Engineering<a id="section7"></a>

- **Prompting Frameworks for Large Language Models: A Survey**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.12785)] [[GitHub](https://github.com/lxx0628/Prompting-Framework-Survey)]

- **Unleashing the potential of prompt engineering in Large Language Models: a comprehensive review**, arXiv 2023.10 [[Paper](https://arxiv.org/abs/2310.14735)]

###### Reasoning<a id="section8"></a>

- **Towards Reasoning in Large Language Models: A Survey**, arXiv 2022.12 [[Paper](https://arxiv.org/abs/2212.10403)] [[GitHub](https://github.com/jeffhj/LM-reasoning)]

- **A Survey of Reasoning with Foundation Models**, arXiv 2023.12 [[Paper](https://arxiv.org/abs/2312.11562)] [[GitHub](https://github.com/reasoning-survey/Awesome-Reasoning-Foundation-Models)]

#### Data<a id="section9"></a>

- **Data Management For Large Language Models: A Survey**, arXiv 2023.12 [[Paper](https://arxiv.org/abs/2312.01700)] [[GitHub](https://github.com/ZigeW/data_management_LLM)]

- **A Survey on Data Selection for Language Models**, arXiv 2024.02 [[Paper](https://arxiv.org/abs/2402.16827)]

- **Datasets for Large Language Models: A Comprehensive Survey**, arXiv 2024.02 [[Paper](https://arxiv.org/abs/2402.18041)] [[GitHub](https://github.com/lmmlzn/Awesome-LLMs-Datasets)]

- **Large Language Models for Data Annotation: A Survey**, arXiv 2024.02 [[Paper](https://arxiv.org/abs/2402.13446)] [[GitHub](https://github.com/Zhen-Tan-dmml/LLM4Annotation)]

- **A Survey on Data Selection for LLM Instruction Tuning**, arXiv 2024.02 [[Paper](https://arxiv.org/abs/2402.05123)]

#### Evaluation<a id="section10"></a>

- **Evaluating Large Language Models: A Comprehensive Survey**, arXiv 2023.10 [[Paper](https://arxiv.org/abs/2310.19736)] [[GitHub](https://github.com/tjunlp-lab/Awesome-LLMs-Evaluation-Papers)]

- **A Survey on Evaluation of Large Language Models**, arXiv 2023.07 [[Paper](https://arxiv.org/abs/2307.03109)] [[GitHub](https://github.com/MLGroupJLU/LLM-eval-survey)]

- **Baby steps in evaluating the capacities of large language models**, arXiv 2023.06 [[Paper](https://www.nature.com/articles/s44159-023-00211-x)]

#### Societal Issues<a id="section11"></a>

- **A Survey on Fairness in Large Language Models**, arXiv 2023.08 [[Paper](https://arxiv.org/abs/2308.10149)]

- **Large Language Models as Subpopulation Representative Models: A Review**, arXiv 2023.10 [[Paper](https://arxiv.org/abs/2310.17888)]

- **Perception, performance, and detectability of conversational artificial intelligence across 32 university courses**, SCI REP-UK 2023.08 [[Paper](https://www.nature.com/articles/s41598-023-38964-3)]

- **Should chatgpt be biased? challenges and risks of bias in large language models**, arXiv 2023.04 [[Paper](https://arxiv.org/abs/2304.03738)]

- **Bias and Fairness in Large Language Models: A Survey**, arXiv 2023.09 [[Paper](https://arxiv.org/abs/2309.00770)] [[GitHub](https://github.com/i-gallegos/Fair-LLM-Benchmark)]

#### Safety<a id="section12"></a>

###### Source Detection<a id="section13"></a>

- **A Survey on Detection of LLMs-Generated Content**, arXiv 2023.10 [[Paper](https://arxiv.org/abs/2310.15654)] [[GitHub](https://github.com/Xianjun-Yang/Awesome_papers_on_LLMs_detection)]

- **A Survey on LLM-generated Text Detection: Necessity, Methods, and Future Directions**, arXiv 2023.10 [[Paper](https://arxiv.org/abs/2310.14724)] [[GitHub](https://github.com/NLP2CT/LLM-generated-Text-Detection)]

- **Detecting ChatGPT: A Survey of the State of Detecting ChatGPT-Generated Text**, arXiv 2023.09 [[Paper](https://arxiv.org/abs/2309.07689)]

- **The Science of Detecting LLM-Generated Texts**, arXiv 2023.02 [[Paper](https://arxiv.org/abs/2303.07205)]

###### Security<a id="section14"></a>

- **Survey of Vulnerabilities in Large Language Models Revealed by Adversarial Attacks**, arXiv 2023.10 [[Paper](https://arxiv.org/abs/2310.10844)]

- **A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly**, arXiv 2023.12 [[Paper](https://arxiv.org/abs/2312.02003)]

- **Tricking LLMs into Disobedience: Formalizing, Analyzing, and Detecting Jailbreaks**, arXiv 2023.05 [[Paper](https://arxiv.org/abs/2305.14965)]

- **A Survey of Safety and Trustworthiness of Large Language Models through the Lens of Verification and Validation**, arXiv 2023.05 [[Paper](https://arxiv.org/abs/2305.11391)]

#### Misinformation<a id="section15"></a>

###### Hallucinations<a id="section16"></a>

- **Can Knowledge Graphs Reduce Hallucinations in LLMs? : A Survey**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.07914)]

- **A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.05232)] [[GitHub](https://github.com/LuckyyySTA/Awesome-LLM-hallucination)]

- **A Survey of Hallucination in “Large” Foundation Models**, arXiv 2023.09 [[Paper](https://arxiv.org/abs/2309.05922)] [[GitHub](https://github.com/vr25/hallucination-foundation-model-survey)]

- **Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models**, arXiv 2023.09 [[Paper](https://arxiv.org/abs/2309.01219)] [[GitHub](https://github.com/HillZhang1999/llm-hallucination-survey)]

- **Cognitive Mirage: A Review of Hallucinations in Large Language Models**, arXiv 2023.09 [[Paper](https://arxiv.org/abs/2309.06794)] [[GitHub](https://github.com/hongbinye/Cognitive-Mirage-Hallucinations-in-LLMs)]

- **Augmenting LLMs with Knowledge: A survey on hallucination prevention**, arXiv 2023.09 [[Paper](https://arxiv.org/abs/2309.16459)]

- **A Comprehensive Survey of Hallucination Mitigation Techniques in Large Language Models**, arXiv 2024.01 [[Paper](https://arxiv.org/abs/2401.01313)]

###### Factuality<a id="section17"></a>

- **Trustworthy LLMs: a Survey and Guideline for Evaluating Large Language Models' Alignment**, arXiv 2023.08 [[Paper](https://arxiv.org/abs/2308.05374)]

- **A Survey on Factuality in Large Language Models: Knowledge, Retrieval and Domain-Specificity**, arXiv 2023.10 [[Paper](https://arxiv.org/abs/2310.07521)] [[GitHub](https://github.com/wangcunxiang/LLM-Factuality-Survey)]

- **Give Me the Facts! A Survey on Factual Knowledge Probing in Pre-trained Language Models**, arXiv 2023.10 [[Paper](https://arxiv.org/abs/2310.16570)]

#### Attributes of LLMs<a id="section18"></a>

- **Explainability for Large Language Models: A Survey**, arXiv 2023.09 [[Paper](https://arxiv.org/abs/2309.01029)]

- **The Mystery and Fascination of LLMs: A Comprehensive Survey on the Interpretation and Analysis of Emergent Abilitie**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.00237)]

- **From Understanding to Utilization: A Survey on Explainability for Large Language Models**, arXiv 2024.01 [[Paper](https://arxiv.org/abs/2401.12874)]

- **A Survey of Large Language Models Attribution**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.03731)] [[GitHub](https://github.com/HITsz-TMG/awesome-llm-attributions)]

- **A Survey of Language Model Confidence Estimation and Calibration**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.08298)]

- **Shortcut Learning of Large Language Models in Natural Language Understanding**, COMMUN ACM 2023.12  [[Paper](https://dl.acm.org/doi/10.1145/3596490)]

- **Automatically Correcting Large Language Models: Surveying the landscape of diverse self-correction strategies**, arXiv 2023.08 [[Paper](https://arxiv.org/abs/2308.03188)] [[GitHub](https://github.com/teacherpeterpan/self-correction-llm-papers)]

#### Efficient LLMs<a id="section19"></a>

- **Efficient Large Language Models: A Survey**, arXiv 2023.12 [[Paper](https://arxiv.org/abs/2312.03863)] [[GitHub](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)]

- **LLM Inference Unveiled: Survey and Roofline Model Insights**, arXiv 2024.03 [[Paper](https://arxiv.org/abs/2402.16363)]

- **Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems**, arXiv 2023.12 [[Paper](https://arxiv.org/abs/2312.15234)]

- **A Survey on Model Compression for Large Language Models**, arXiv 2023.08 [[Paper](https://arxiv.org/abs/2308.07633)]

- **A Comprehensive Survey of Compression Algorithms for Language Models**, arXiv 2024.01 [[Paper](https://arxiv.org/abs/2401.15347)]

- **A Survey on Knowledge Distillation of Large Language Models**, arXiv 2024.02 [[Paper](https://arxiv.org/abs/2402.13116)]

- **The Efficiency Spectrum of Large Language Models: An Algorithmic Survey**, arXiv 2023.10 [[Paper](https://arxiv.org/abs/2310.10844)] [[GitHub](https://github.com/tding1/Efficient-LLM-Survey)]

- **Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment**, arXiv 2023.12 [[Paper](https://arxiv.org/abs/2312.12148)]

- **Model Compression and Efficient Inference for Large Language Models: A Survey**, arXiv 2024.02 [[Paper](https://arxiv.org/abs/2402.09748)]

- **Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding**, arXiv 2024.01 [[Paper](https://arxiv.org/abs/2401.07851)] [[GitHub](https://github.com/hemingkx/SpeculativeDecodingPapers)]

- **A Survey on Hardware Accelerators for Large Language Models**, arXiv 2024.01 [[Paper](https://arxiv.org/abs/2401.09890)]

#### Learning Methods for LLMs<a id="section20"></a>

- **Knowledge Unlearning for LLMs: Tasks, Methods, and Challenges**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.15766)]

- **Continual Learning with Pre-Trained Models: A Survey**, arXiv 2024.01 [[Paper](https://arxiv.org/abs/2401.16386)] [[GitHub](https://github.com/sun-hailong/LAMDA-PILOT)]

- **Continual Learning for Large Language Models: A Survey**, arXiv 2024.02 [[Paper](https://arxiv.org/abs/2402.01364)]

#### Multimodal LLMs<a id="section21"></a>

- **Vision-Language Instruction Tuning: A Review and Analysis**, arXiv 2023,11 [[Paper](https://arxiv.org/abs/2311.08172)] [[GitHub](https://github.com/palchenli/VL-Instruction-Tuning)]

- **Large Language Models Meet Computer Vision: A Brief Survey**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.16673)]

- **Foundational Models Defining a New Era in Vision: A Survey and Outlook**, arXiv 2023.07 [[Paper](https://arxiv.org/abs/2307.13721)] [[GitHub](https://github.com/awaisrauf/Awesome-CV-Foundational-Models)]

- **Video Understanding with Large Language Models: A Survey**, arXiv 2023.12 [[Paper](https://arxiv.org/abs/2312.17432)] [[GitHub](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding)]

- **Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook**, arXiv 2023.10 [[Paper](https://arxiv.org/abs/2310.10196)] [[GitHub](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)]

- **Sparks of large audio models: A survey and outlook**, arXiv 2023.08 [[Paper](https://arxiv.org/abs/2308.12792)] [[GitHub](https://github.com/EmulationAI/awesome-large-audio-models)]

- **How to Bridge the Gap between Modalities: A Comprehensive Survey on Multimodal Large Language Model**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.07594)]

- **A Survey on Multimodal Large Language Models**, arXiv 2023.06 [[Paper](https://arxiv.org/abs/2306.13549)]

- **Multimodal Large Language Models: A Survey**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.13165)]

#### Knowledge Based LLMs<a id="section22"></a>

###### Retrieval-Augmented LLMs<a id="section23"></a>

- **Building trust in conversational ai: A comprehensive review and solution architecture for explainable, privacy-aware systems using llms and knowledge graph**, arXiv 2023.08 [[Paper](https://arxiv.org/abs/2308.13534)]

- **A Survey on Retrieval-Augmented Text Generation**, arXiv 2022.02 [[Paper](https://arxiv.org/abs/2202.01110)]

- **Retrieval-Augmented Generation for Large Language Models: A Survey**, arXiv 2023.12 [[Paper](https://arxiv.org/abs/2312.10997)] [[GitHub](https://github.com/Tongji-KGLLM/RAG-Survey)]

###### Knowledge Editing<a id="section24"></a>

- **Trends in Integration of Knowledge and Large Language Models: A Survey and Taxonomy of Methods, Benchmarks, and Applications**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.05876)]

- **Knowledge Editing for Large Language Models: A Survey**, arXiv 2023.10 [[Paper](https://arxiv.org/abs/2310.16218)]

- **Editing Large Language Models: Problems, Methods, and Opportunities**, arXiv 2023.05 [[Paper](https://arxiv.org/abs/2305.13172)]

#### Extension of LLMs<a id="section25"></a>

###### LLMs with Tools<a id="section26"></a>

- **Foundation Models for Decision Making: Problems, Methods, and Opportunities**, arXiv 2023.03 [[Paper](https://arxiv.org/abs/2303.04129)]

- **Augmented Language Models: a Survey**, arXiv 2023.02 [[Paper](https://arxiv.org/abs/2302.07842)]

- **A Survey on Language Models for Code**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.07989)] [[GitHub](https://github.com/codefuse-ai/Awesome-Code-LLM)]

- **Pitfalls in Language Models for Code Intelligence: A Taxonomy and Survey**, arXiv 2023.10 [[Paper](https://arxiv.org/abs/2310.17903)] [[GitHub](https://github.com/yueyueL/ReliableLM4Code)]

- **Large Language Models Meet NL2Code: A Survey**, arXiv 2022.12 [[Paper](https://arxiv.org/abs/2212.09420)]

###### LLMs and Interactions<a id="section27"></a>

- **Large Language Models for Robotics: A Survey**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.07226)]

- **A Survey on Multimodal Large Language Models for Autonomous Driving**, WACV workshop 2023.11 [[Paper](https://openaccess.thecvf.com/content/WACV2024W/LLVM-AD/papers/Cui_A_Survey_on_Multimodal_Large_Language_Models_for_Autonomous_Driving_WACVW_2024_paper.pdf)]

- **LLM4Drive: A Survey of Large Language Models for Autonomous Driving**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.01043v3)] [[GitHub](https://github.com/Thinklab-SJTU/Awesome-LLM4AD)]

- **A Survey on Large Language Model based Autonomous Agents**, arXiv  2023.08 [[Paper](https://arxiv.org/abs/2308.11432)] [[GitHub](https://github.com/Paitesanshi/LLM-Agent-Survey)]

- **The Rise and Potential of Large Language Model Based Agents: A Survey**, arXiv 2023.09 [[Paper](https://arxiv.org/abs/2309.07864)] [[GitHub](https://github.com/WooooDyy/LLM-Agent-Paper-List)]

- **Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives**, arXiv 2023.12  [[Paper](https://arxiv.org/abs/2312.11970)]

- **Large Multimodal Agents: A Survey**, arXiv 2024.02 [[Paper](https://arxiv.org/abs/2402.15116)] [[GitHub](https://github.com/jun0wanan/awesome-large-multimodal-agents)]

- **Role play with large language models**, arXiv 2023.11 [[Paper](https://www.nature.com/articles/s41586-023-06647-8)]

#### Long Sequence LLMs<a id="section28"></a>

- **Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.12351)]

- **Length Extrapolation of Transformers: A Survey from the Perspective of Position Encoding**, arXiv 2023.12 [[Paper](https://arxiv.org/abs/2312.17044)]

#### LLMs Applications<a id="section29"></a>

###### Education<a id="section30"></a>

- **ChatGPT and Beyond: The Generative AI Revolution in Education**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.15198)]

- **ChatGPT and large language models in academia: opportunities and challenges**, arXiv 2023.07 [[Paper](https://link.springer.com/article/10.1186/s13040-023-00339-9)]

- **ChatGPT for good? On opportunities and challenges of large language models for education**, arXiv 2023.04 [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1041608023000195)]

###### Law<a id="section31"></a>

- **Large Language Models in Law: A Survey**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2312.03718)]

- **A short survey of viewing large language models in legal aspect**, arXiv 2023.03 [[Paper](https://arxiv.org/abs/2303.09136)]

###### Healthcare<a id="section32"></a>

- **A Survey of Large Language Models in Medicine: Progress, Application, and Challenge**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.05112)] [[GitHub](https://github.com/AI-in-Health/MedLLMsPracticalGuide)]

- **Large Language Models Illuminate a Progressive Pathway to Artificial Healthcare Assistant: A Review**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.01918)] [[GitHub](https://github.com/mingze-yuan/Awesome-LLM-Healthcare)]

- **Large AI Models in Health Informatics: Applications, Challenges, and the Future**, arXiv 2023.03 [[Paper](https://arxiv.org/abs/2303.11568)] [[GitHub](https://github.com/Jianing-Qiu/Awesome-Healthcare-Foundation-Models)]

- **A SWOT (Strengths, Weaknesses, Opportunities, and Threats) Analysis of ChatGPT in the Medical Literature: Concise Review**, JMIR 2023.11 [[Paper](https://www.jmir.org/2023/1/e49368/)]

- **ChatGPT in Healthcare: A Taxonomy and Systematic Review**, Computer Methods and Programs in Biomedicine 2024.01 [[Paper](https://www.sciencedirect.com/science/article/pii/S0169260724000087)]

- **A review of the explainability and safety of conversational agents for mental health to identify avenues for improvement**, NCBI 2023.10 [[Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10601652/)]

- **Towards a Psychological Generalist AI: A Survey of Current Applications of Large Language Models and Future Prospects**, arXiv 2023.12 [[Paper](https://arxiv.org/abs/2312.04578)]

- **Large Language Models in Mental Health Care: a Scoping Review**, arXiv 2024.01 [[Paper](https://arxiv.org/abs/2401.02984)]

- **The utility of ChatGPT as an example of large language models in healthcare education, research and practice: Systematic review on the future perspectives and**, arXiv 2023.12 [[Paper](https://www.medrxiv.org/content/10.1101/2023.02.19.23286155v1)]

- **The imperative for regulatory oversight of large language models (or generative AI) in healthcare**, arXiv 2023.07 [[Paper](https://www.nature.com/articles/s41746-023-00873-0)]

- **A Survey of Large Language Models for Healthcare: from Data, Technology, and Applications to Accountability and Ethics**, arXiv 2023.10 [[Paper](https://arxiv.org/abs/2310.05694)] [[GitHub](https://github.com/KaiHe-better/LLM-for-Healthcare)]

- **The Shaky Foundations of Clinical Foundation Models: A Survey of Large Language Models and Foundation Models for EMRs**, arXiv 2023.03 [[Paper](https://arxiv.org/abs/2303.12961)]

###### Games<a id="section33"></a>

- **Large Language Models and Games: A Survey and Roadmap**, arXiv 2024.02 [[Paper](https://arxiv.org/abs/2402.18659)]

- **Large Language Models and Video Games: A Preliminary Scoping Review**, arXiv 2024.03 [[Paper](https://arxiv.org/abs/2403.02613)]

###### NLP Tasks<a id="section34"></a>

- **Large Language Models for Information Retrieval: A Survey**, arXiv 2023.08 [[Paper](https://arxiv.org/abs/2308.07107)] [[GitHub](https://github.com/RUC-NLPIR/LLM4IR-Survey)]

- **Large Language Models for Generative Information Extraction: A Survey**, arXiv 2023.12 [[Paper](https://arxiv.org/abs/2312.17617)] [[GitHub](https://github.com/quqxui/Awesome-LLM4IE-Papers)]

- **Recent Advances in Natural Language Processing via Large Pre-Trained Language Models: A Survey**, arXiv 2021.11 [[Paper](https://arxiv.org/abs/2111.01243)]

- **If LLM Is the Wizard, Then Code Is the Wand: A Survey on How Code Empowers Large Language Models to Serve as Intelligent Agents**, arXiv 2024.01 [[Paper](https://arxiv.org/abs/2401.00812)]

###### Software Engineering<a id="section35"></a>

- **Large Language Models for Software Engineering: Survey and Open Problems**, arXiv 2023.10 [[Paper](https://arxiv.org/abs/2310.03533)]

- **Large Language Models for Software Engineering: A Systematic Literature Review**, arXiv 2023.08 [[Paper](https://arxiv.org/abs/2308.10620)]

- **Software Testing with Large Language Models: Survey, Landscape, and Vision**, arXiv 2023.07 [[Paper](https://arxiv.org/abs/2307.07221)]

###### Recommender Systems<a id="section36"></a>

- **Foundation Models for Recommender Systems: A Survey and New Perspectives**, arXiv 2024.02 [[Paper](https://arxiv.org/abs/2402.11143)]

- **User Modeling in the Era of Large Language Models: Current Research and Future Directions**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2312.11518)] [[GitHub]( https://github.com/TamSiuhin/LLM-UM-Reading)]

- **A Survey on Large Language Models for Personalized and Explainable Recommendations**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.12338)]

- **Large Language Models for Generative Recommendation: A Survey and Visionary Discussions**, arXiv 2023.09 [[Paper](https://arxiv.org/abs/2309.01157)]

- **A Survey on Large Language Models for Recommendation**, arXiv 2023.05 [[Paper](https://arxiv.org/abs/2305.19860)] [[GitHub](https://github.com/WLiK/LLM4Rec)]

- **How Can Recommender Systems Benefit from Large Language Models: A Survey**, arXiv 2023.06 [[Paper](https://arxiv.org/abs/2306.05817)] [[GitHub](https://github.com/CHIANGEL/Awesome-LLM-for-RecSys/)]

###### Graphs<a id="section37"></a>

- **A Survey of Graph Meets Large Language Model: Progress and Future Directions**, arXiv 2023.11 [[Paper](https://arxiv.org/abs/2311.12399)]

- **Large Language Models on Graphs: A Comprehensive Survey**, arXiv 2023.12 [[Paper](https://arxiv.org/abs/2312.02783)] [[GitHub](https://github.com/PeterGriffinJin/Awesome-Language-Model-on-Graphs)]

- **The Contribution of Knowledge in Visiolinguistic Learning: A Survey on Tasks and Challenges**, arXiv 2023.03 [[Paper](https://arxiv.org/abs/2303.02411)]

###### Other<a id="section38"></a>

- **Large Language Models in Finance: A Survey**, ICAIF 2023.11 [[Paper](https://dl.acm.org/doi/10.1145/3604237.3626869)]

- **Mathematical Language Models: A Survey**, arXiv 2023.12 [[Paper](https://arxiv.org/abs/2312.07622)]

- **Recent applications of AI to environmental disciplines: A review**, SCI TOTAL ENVIRON 2023.10 [[Paper](https://www.sciencedirect.com/science/article/pii/S0048969723063325?casa_token=sbh1pxIYyAgAAAAA:f3WytHabl8udc5v8OhRunnwHEemEAwNafzAcP2reVdGKMAJ-4EcJIxwKO4gdE8ozb6ZibbcY2_4)]

- **Opportunities and Challenges of Applying Large Language Models in Building Energy Efficiency and Decarbonization Studies: An Exploratory Overview**, arXiv 2023.12 [[Paper](https://arxiv.org/abs/2312.11701)]

- **When Large Language Models Meet Citation: A Survey**, arXiv 2023.09 [[Paper](https://arxiv.org/abs/2309.09727)]

- **A Survey of Text Watermarking in the Era of Large Language Models**, arXiv 2023.12 [[Paper](https://arxiv.org/abs/2312.07913)]

- **The future of gpt: A taxonomy of existing chatgpt research, current challenges, and possible future directions**, SSRN 2023.04 [[Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4413921)]

- **Summary of ChatGPT-Related Research and Perspective Towards the Future of Large Language Models**, Meta-Radiology 2023.09 [[Paper](https://www.sciencedirect.com/science/article/pii/S2950162823000176)]





<!-- >Feel free to let me know the missing papers (issue or pull request). -->


<!-- ## Star History

<a href="https://star-history.com/#NiuTrans/ABigSurvey&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=HqWu-HITCS/Awesome-LLM-Survey&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=HqWu-HITCS/Awesome-LLM-Survey&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=HqWu-HITCS/Awesome-LLM-Survey&type=Date" />
  </picture>
</a> -->
<!-- ## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NiuTrans/ABigSurvey&type=Date)](https://star-history.com/#NiuTrans/ABigSurvey&Date) -->

<!-- ## Team Members

The project is maintained by

*Junhao Ruan*$^{[1]}$, *Long Meng*$^{[1]}$, *Weiqiao Shan*$^{[1]}$, *Tong Xiao*, *Jingbo Zhu*


*Natural Language Processing Lab., School of Computer Science and Engineering, Northeastern University*

*NiuTrans Research*

Please feel free to contact us if you have any questions (libei_neu [at] outlook.com). -->

## Acknowledgements

We would like to thank the people who have contributed to this project. The core contributors are

*Junhao Ruan, Long Meng, Weiqiao Shan, Tong Xiao, Jingbo Zhu*

