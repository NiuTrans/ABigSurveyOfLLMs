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
<!-- - [A Survey of Surveys (LLMs)](#a-survey-of-surveys-llms)
	- [Categorization](#categorization)
	- [📜 The LLMs Paper List](#-the-llms-paper-list)
			- [﻿General](#general)
			- [Alignment](#alignment)
					- [Human alignment](#human-alignment)
					- [Prompt engineering](#prompt-engineering)
			- [Data](#data)
			- [Evaluation](#evaluation)
			- [Societal Issues](#societal-issues)
			- [Safety](#safety)
					- [Detection](#detection)
					- [Security](#security)
			- [Misinformation](#misinformation)
					- [Hallucinations](#hallucinations)
					- [Factuality](#factuality)
			- [Attributes of LMs](#attributes-of-lms)
			- [Efficient LMs](#efficient-lms)
					- [Compute efficient](#compute-efficient)
					- [System](#system)
			- [Learning Methods for LMs](#learning-methodsfor-lms)
			- [Multimodality LMs](#multimodality-lms)
			- [Knowledge of LMs](#knowledge-of-lms)
					- [Retrieval-augmented](#retrieval-augmented)
					- [Knowledge editing](#knowledge-editing)
			- [Extension of LMs](#extension-of-lms)
					- [LMs with tools](#lms-withtools)
					- [LMs and interactions](#lms-andinteractions)
			- [Long Sequence Generation of LMs](#long-sequence-generation-of-lms)
			- [LM Applications](#lm-applications)
					- [Interdisciplinary field](#interdisciplinary-field)
					- [CS field](#cs-field)
					- [Other field](#other-field)
	- [⭐️ Star History](#️-star-history)
	- [Team Members](#team-members)
	- [Acknowledge](#acknowledge) -->

+ [General Surveys](#section1)
+ [Alignment](#section2)
+ [In-context Learning](#section3)
	+ [Prompt Learning](#section4)
	+ [Chain of Thought](#section5)
+ [Data](#section6)
+ [Evaluation](#section7)
+ [Societal Issues](#section8)
+ [Safety](#section9)
	+ [Source Detection](#section10)
	+ [Security](#section11)
+ [Misinformation](#section12)
	+ [Hallucinations](#section13)
	+ [Factuality](#section14)
+ [Attributes of LLMs](#section15)
+ [Efficient LLMs](#section16)
+ [Learning Methods for LLMs](#section17)
+ [Multimodal LLMs](#section18)
+ [Knowledge Based LLMs](#section19)
	+ [Retrieval-Augmented](#section20)
	+ [Knowledge Editing](#section21)
+ [Extension of LLMs](#section22)
	+ [LLMs with Tools](#section23)
	+ [LLMs and Interactions](#section24)
+ [Long Sequence LLMs](#section25)
+ [LLMs Applications](#section26)
	+ [Education](#section27)
	+ [Law](#section28)
	+ [Health](#section29)
	+ [Finance](#section30)
	+ [Game](#section31)
	+ [NLP Tasks](#section32)
	+ [Software Engineering](#section33)
	+ [Recommender Systems](#section34)
	+ [Other](#section35)



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

#### ﻿General Surveys<a id="section1"></a>

1. **Large Language Models: A Survey** arXiv (2024.02)

	*Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, Jianfeng Gao* [[Paper](https://arxiv.org/abs/2402.06196)]

2. **A Comprehensive Survey of AI-Generated Content (AIGC): A History of Generative AI from GAN to ChatGPT** arXiv (2023.03)

	*Yihan Cao, Siyu Li, Yixin Liu, Zhiling Yan, Yutong Dai, Philip S. Yu, Lichao Sun* [[Paper](https://arxiv.org/abs/2303.04226)]

3. **A Survey of Large Language Models** arXiv (2023.11)

	*Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen* [[Paper](https://arxiv.org/abs/2303.18223)] [[GitHub](https://github.com/RUCAIBox/LLMSurvey)]

4. **A Survey of GPT-3 Family Large Language Models Including ChatGPT and GPT-4** arXiv (2023.10)

	*Katikapalli Subramanyam Kalyan* [[Paper](https://arxiv.org/abs/2310.12321)]

5. **Challenges and Applications of Large Language Models** arXiv (2023.07)

	*Jean Kaddour, Joshua Harris, Maximilian Mozes, Herbie Bradley, Roberta Raileanu, Robert McHardy* [[Paper](https://arxiv.org/abs/2307.10169)]

6. **Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond** arXiv (2023.04)

	*Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Bing Yin, Xia Hu* [[Paper](https://arxiv.org/abs/2304.13712)] [[GitHub](https://github.com/Mooler0410/LLMsPracticalGuide)]

7. **A Survey on Large Language Models: Applications, Challenges, Limitations, and Practical Usage** TechRxiv (2023.07)

	*Muhammad Usman Hadi ,qasem al tashi ,Rizwan Qureshi ,Abbas Shah ,amgad muneer ,Muhammad Irfan ,Anas Zafar ,Muhammad Bilal Shaikh ,Naveed Akhtar ,Jia Wu ,Seyedali Mirjalili ,Mubarak Shah* [[Paper](https://www.techrxiv.org/doi/full/10.36227/techrxiv.23589741.v1)] [[GitHub](https://github.com/anas-zafar/LLM-Survey)]

8. **The future of gpt: A taxonomy of existing chatgpt research, current challenges, and possible future directions** SSRN (2023.04)

	*Shahab Saquib Sohaila, Faiza Farhatb, Yassine Himeurc, Mohammad Nadeemd, Dag Oivind Madsene, Yashbir Singhf, Shadi Atallac and Wathiq Mansoorc* [[Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4413921)]

9. **A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT** arXiv (2023.05)

	*Ce Zhou, Qian Li, Chen Li, Jun Yu, Yixin Liu, Guangjing Wang, Kai Zhang, Cheng Ji, Qiben Yan, Lifang He, Hao Peng, Jianxin Li, Jia Wu, Ziwei Liu, Pengtao Xie, Caiming Xiong, Jian Pei, Philip S. Yu, Lichao Sun* [[Paper](https://arxiv.org/abs/2302.09419)]

10. **Science in the age of large language models** arXiv (2023.04)

	*Abeba Birhane* [[Paper](https://www.nature.com/articles/s42254-023-00581-4)]

#### Alignment<a id="section2"></a>

1. **Bridging the Gap: A Survey on Integrating (Human) Feedback for Natural Language Generation** arXiv (2023.06)

	*Patrick Fernandes, Aman Madaan, Emmy Liu, António Farinhas, Pedro Henrique Martins, Amanda Bertsch, José G. C. de Souza, Shuyan Zhou, Tongshuang Wu, Graham Neubig, André F. T. Martins* [[Paper](https://arxiv.org/abs/2305.00955)]

2. **AI Alignment: A Comprehensive Survey** arXiv (2024.02)

	*Jiaming Ji, Tianyi Qiu, Boyuan Chen, Borong Zhang, Hantao Lou, Kaile Wang, Yawen Duan, Zhonghao He, Jiayi Zhou, Zhaowei Zhang, Fanzhi Zeng, Kwan Yee Ng, Juntao Dai, Xuehai Pan, Aidan O'Gara, Yingshan Lei, Hua Xu, Brian Tse, Jie Fu, Stephen McAleer, Yaodong Yang, Yizhou Wang, Song-Chun Zhu, Yike Guo, Wen Gao* [[Paper](https://arxiv.org/abs/2310.19852)]

3. **Large Language Model Alignment: A Survey** arXiv (2023.09)

	*Tianhao Shen, Renren Jin, Yufei Huang, Chuang Liu, Weilong Dong, Zishan Guo, Xinwei Wu, Yan Liu, Deyi Xiong* [[Paper](https://arxiv.org/abs/2309.15025)]

4. **From Instructions to Intrinsic Human Values -- A Survey of Alignment Goals for Big Models** arXiv (2023.09)

	*Jing Yao, Xiaoyuan Yi, Xiting Wang, Jindong Wang, Xing Xie* [[Paper](https://arxiv.org/abs/2308.12014)] [[GitHub](https://github.com/ValueCompass/Alignment-Goal-Survey)]

5. **Aligning Large Language Models with Human: A Survey** arXiv (2023.07)

	*Yufei Wang, Wanjun Zhong, Liangyou Li, Fei Mi, Xingshan Zeng, Wenyong Huang, Lifeng Shang, Xin Jiang, Qun Liu* [[Paper](https://arxiv.org/abs/2307.12966)] [[GitHub](https://github.com/GaryYufei/AlignLLMHumanSurvey)]

6. **Instruction Tuning for Large Language Models: A Survey** arXiv (2023.08)

	*Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, Guoyin Wang* [[Paper](https://arxiv.org/abs/2308.10792)]

#### In-context Learning<a id="section3"></a>

###### Prompt Learning<a id="section4"></a>

1. **A Comprehensive Survey on Instruction Following** arXiv (2024.01)

	*Renze Lou, Kai Zhang, Wenpeng Yin* [[Paper](https://arxiv.org/abs/2303.10475v7)] [[GitHub](https://github.com/RenzeLou/awesome-instruction-learning)]

2. **A Practical Survey on Zero-shot Prompt Design for In-context Learning** ranlp (2023.09)

	*Yinheng Li* [[Paper](https://aclanthology.org/2023.ranlp-1.69/)]

3. **Prompting Frameworks for Large Language Models: A Survey** arXiv (2023.11)

	*Xiaoxia Liu, Jingyi Wang, Jun Sun, Xiaohan Yuan, Guoliang Dong, Peng Di, Wenhai Wang, Dongxia Wang* [[Paper](https://arxiv.org/abs/2311.12785)] [[GitHub](https://github.com/lxx0628/Prompting-Framework-Survey)]

4. **Unleashing the potential of prompt engineering in Large Language Models: a comprehensive review** arXiv (2023.10)

	*Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu* [[Paper](https://arxiv.org/abs/2310.14735)]

5. **A Survey on In-context Learning** arXiv (2023.06)

	*Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, Lei Li, Zhifang Sui* [[Paper](https://arxiv.org/abs/2301.00234)]

6. **Towards Reasoning in Large Language Models: A Survey** arXiv (2022.12)

	*Jie Huang, Kevin Chen-Chuan Chang* [[Paper](https://arxiv.org/abs/2212.10403)] [[GitHub](https://github.com/jeffhj/LM-reasoning)]

###### Chain of Thought<a id="section5"></a>

1. **A Survey of Chain of Thought Reasoning: Advances, Frontiers and Future** arXiv (2023.10)

	*Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng, Ming Liu, Bing Qin, Ting Liu* [[Paper](https://arxiv.org/abs/2309.15402)] [[GitHub](https://github.com/zchuz/CoT-Reasoning-Survey)]

2. **Towards Better Chain-of-Thought Prompting Strategies: A Survey** arXiv (2023.10)

	*Zihan Yu, Liang He, Zhen Wu, Xinyu Dai, Jiajun Chen* [[Paper](https://doi.org/10.48550/arXiv.2310.04959)]

3. **A Survey of Chain of Thought Reasoning: Advances, Frontiers and Future** arXiv (2023.09)

	*Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng, Ming Liu, Bing Qin, Ting Liu* [[Paper](https://arxiv.org/abs/2309.15402)] [[GitHub](https://github.com/zchuz/CoT-Reasoning-Survey)]

4. **Igniting Language Intelligence: The Hitchhiker's Guide From Chain-of-Thought Reasoning to Language Agents** arXiv (2023.11)

	*Zhuosheng Zhang, Yao Yao, Aston Zhang, Xiangru Tang, Xinbei Ma, Zhiwei He, Yiming Wang, Mark Gerstein, Rui Wang, Gongshen Liu, Hai Zhao* [[Paper](https://arxiv.org/abs/2311.11797)] [[GitHub](https://github.com/Zoeyyao27/CoT-Igniting-Agent)]

5. **A Survey of Reasoning with Foundation Models** arXiv (2023.12)

	*Jiankai Sun, Chuanyang Zheng, Enze Xie, Zhengying Liu, Ruihang Chu, Jianing Qiu, Jiaqi Xu, Mingyu Ding, Hongyang Li, Mengzhe Geng, Yue Wu, Wenhai Wang, Junsong Chen, Zhangyue Yin, Xiaozhe Ren, Jie Fu, Junxian He, Wu Yuan, Qi Liu, Xihui Liu, Yu Li, Hao Dong, Yu Cheng, Ming Zhang, Pheng Ann Heng, Jifeng Dai, Ping Luo, Jingdong Wang, Ji-Rong Wen, Xipeng Qiu, Yike Guo, Hui Xiong, Qun Liu, Zhenguo Li* [[Paper](https://arxiv.org/pdf/2312.11562.pdf)] [[GitHub](https://github.com/reasoning-survey/Awesome-Reasoning-Foundation-Models)]

#### Data<a id="section6"></a>

1. **Data Management For Large Language Models: A Survey** arXiv (2023.12)

	*Zige Wang, Wanjun Zhong, Yufei Wang, Qi Zhu, Fei Mi, Baojun Wang, Lifeng Shang, Xin Jiang, Qun Liu* [[Paper](https://arxiv.org/pdf/2312.01700)] [[GitHub](https://github.com/ZigeW/data_management_LLM)]

2. **A Survey on Data Selection for Language Models** arXiv (2024.02)

	*Alon Albalak, Yanai Elazar, Sang Michael Xie, Shayne Longpre, Nathan Lambert, Xinyi Wang, Niklas Muennighoff, Bairu Hou, Liangming Pan, Haewon Jeong, Colin Raffel, Shiyu Chang, Tatsunori Hashimoto, William Yang Wang* [[Paper](https://arxiv.org/pdf/2402.16827.pdf)]

3. **Datasets for Large Language Models: A Comprehensive Survey** arXiv (2024.02)

	*Yang Liu, Jiahuan Cao, Chongyu Liu, Kai Ding, Lianwen Jin* [[Paper](https://arxiv.org/pdf/2402.18041.pdf)] [[GitHub](https://github.com/lmmlzn/Awesome-LLMs-Datasets)]

4. **Large Language Models for Data Annotation: A Survey** arXiv (2024.02)

	*Zhen Tan, Alimohammad Beigi, Song Wang, Ruocheng Guo, Amrita Bhattacharjee, Bohan Jiang, Mansooreh Karami, Jundong Li, Lu Cheng, Huan Liu* [[Paper](https://arxiv.org/abs/2402.13446)] [[GitHub](https://github.com/Zhen-Tan-dmml/LLM4Annotation)]

5. **A Survey on Data Selection for LLM Instruction Tuning** arXiv (2024.02)

	*Jiahao Wang, Bolin Zhang, Qianlong Du, Jiajun Zhang, Dianhui Chu* [[Paper](https://arxiv.org/abs/2402.05123)]

#### Evaluation<a id="section7"></a>

1. **Evaluating Large Language Models: A Comprehensive Survey** arXiv (2023.10)

	*Zishan Guo, Renren Jin, Chuang Liu, Yufei Huang, Dan Shi, Supryadi, Linhao Yu, Yan Liu, Jiaxuan Li, Bojian Xiong, Deyi Xiong* [[Paper](https://arxiv.org/pdf/2310.19736.pdf)] [[GitHub](https://github.com/tjunlp-lab/Awesome-LLMs-Evaluation-Papers)]

2. **A Survey on Evaluation of Large Language Models** arXiv (2023.07)

	*Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, Xing Xie* [[Paper](https://arxiv.org/abs/2307.03109)] [[GitHub](https://llm-eval.github.io/)]

3. **Baby steps in evaluating the capacities of large language models** arXiv (2023.06)

	*MC Frank* [[Paper](https://www.nature.com/articles/s44159-023-00211-x)]

#### Societal Issues<a id="section8"></a>

1. **A Survey on Fairness in Large Language Models** arXiv (2023.08)

	*Yingji Li, Mengnan Du, Rui Song, Xin Wang, Ying Wang* [[Paper](https://arxiv.org/abs/2308.10149)]

2. **Large Language Models as Subpopulation Representative Models: A Review** arXiv (2023.10)

	*Gabriel Simmons, Christopher Hare* [[Paper](https://arxiv.org/abs/2310.17888)]

3. **Perception, performance, and detectability of conversational artificial intelligence across 32 university courses** SCI REP-UK (2023.08)

	*Hazem Ibrahim, Fengyuan Liu, Rohail Asim, Balaraju Battu, Sidahmed Benabderrahmane, Bashar Alhafni, Wifag Adnan, Tuka Alhanai, Bedoor AlShebli, Riyadh Baghdadi, Jocelyn J. Bélanger, Elena Beretta, Kemal Celik, Moumena Chaqfeh, Mohammed F. Daqaq, Zaynab El Bernoussi, Daryl Fougnie, Borja Garcia de Soto, Alberto Gandolfi, Andras Gyorgy, Nizar Habash, J. Andrew Harris, Aaron Kaufman, Lefteris Kirousis, Korhan Kocak, Kangsan Lee, Seungah S. Lee, Samreen Malik, Michail Maniatakos, David Melcher, Azzam Mourad, Minsu Park, Mahmoud Rasras, Alicja Reuben, Dania Zantout, Nancy W. Gleason, Kinga Makovi, Talal Rahwan, Yasir Zaki* [[Paper](https://www.nature.com/articles/s41598-023-38964-3)]

4. **Should chatgpt be biased? challenges and risks of bias in large language models** arXiv (2023.04)

	*Emilio Ferrara* [[Paper](https://arxiv.org/abs/2304.03738)]

5. **Bias and Fairness in Large Language Models: A Survey** arXiv (2024.03)

	*Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, Nesreen K. Ahmed* [[Paper](https://arxiv.org/abs/2309.00770)] [[GitHub](https://github.com/i-gallegos/Fair-LLM-Benchmark)]

#### Safety<a id="section9"></a>

###### Source Detection<a id="section10"></a>

1. **A Survey on Detection of LLMs-Generated Content** arXiv (2023.10)

	*Xianjun Yang, Liangming Pan, Xuandong Zhao, Haifeng Chen, Linda Petzold, William Yang Wang, Wei Cheng* [[Paper](https://arxiv.org/abs/2310.15654)] [[GitHub](https://github.com/Xianjun-Yang/Awesome_papers_on_LLMs_detection)]

2. **A Survey on LLM-generated Text Detection: Necessity, Methods, and Future Directions** arXiv (2023.10)

	*Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Derek F. Wong, Lidia S. Chao* [[Paper](https://arxiv.org/abs/2310.14724)] [[GitHub](https://github.com/NLP2CT/LLM-generated-Text-Detection)]

3. **Detecting ChatGPT: A Survey of the State of Detecting ChatGPT-Generated Text** arXiv (2023.09)

	*Mahdi Dhaini, Wessel Poelman, Ege Erdogan* [[Paper](https://arxiv.org/abs/2309.07689)]

4. **The Science of Detecting LLM-Generated Texts** arXiv (2023.02)

	*Ruixiang Tang, Yu-Neng Chuang, Xia Hu* [[Paper](https://arxiv.org/abs/2303.07205)]

###### Security<a id="section11"></a>

1. **Survey of Vulnerabilities in Large Language Models Revealed by Adversarial Attacks** arXiv (2023.1)

	*Erfan Shayegani, Md Abdullah Al Mamun, Yu Fu, Pedram Zaree, Yue Dong, Nael Abu-Ghazaleh* [[Paper](https://arxiv.org/pdf/2310.10844.pdf)]

2. **A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly** arXiv (2023.12)

	*Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo Sun, Yue Zhang* [[Paper](https://arxiv.org/pdf/2312.02003)]

3. **Tricking LLMs into Disobedience: Formalizing, Analyzing, and Detecting Jailbreaks** arXiv (2023.05)

	*Abhinav Rao, Sachin Vashistha, Atharva Naik, Somak Aditya, Monojit Choudhury* [[Paper](https://arxiv.org/abs/2305.14965)]

4. **A Survey of Safety and Trustworthiness of Large Language Models through the Lens of Verification and Validation.** arXiv (2023.08)

	*Xiaowei Huang, Wenjie Ruan, Wei Huang, Gaojie Jin, Yi Dong, Changshun Wu, Saddek Bensalem, Ronghui Mu, Yi Qi, Xingyu Zhao, Kaiwen Cai, Yanghao Zhang, Sihao Wu, Peipei Xu, Dengyu Wu, Andre Freitas, Mustafa A. Mustafa* [[Paper](https://arxiv.org/abs/2305.11391)]

#### Misinformation<a id="section12"></a>

###### Hallucinations<a id="section13"></a>

1. **Can Knowledge Graphs Reduce Hallucinations in LLMs? : A Survey** arXiv (2023.11)

	*Garima Agrawal Tharindu Kumarage Zeyad Alghami Huan Liu* [[Paper](https://arxiv.org/pdf/2311.07914)]

2. **A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions** arXiv (2023.11)

	*Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, Ting Liu* [[Paper](https://arxiv.org/pdf/2311.05232)] [[GitHub](https://github.com/LuckyyySTA/Awesome-LLM-hallucination)]

3. **A Survey of Hallucination in “Large” Foundation Models** arXiv (2023.09)

	*Vipula Rawte, Amit Sheth, Amitava Das* [[Paper](https://arxiv.org/paper/2309.05922)] [[GitHub](https://github.com/vr25/hallucination-foundation-model-survey)]

4. **Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models** arXiv (2023.09)

	*Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, Shuming Shi* [[Paper](https://arxiv.org/abs/2309.01219)] [[GitHub](https://github.com/hongbinye/Cognitive-Mirage-Hallucinations-in-LLMs)]

5. **Cognitive Mirage: A Review of Hallucinations in Large Language Models** arXiv (2023.09)

	*Hongbin Ye, Tong Liu, Aijia Zhang, Wei Hua, Weiqiang Jia* [[Paper](https://arxiv.org/abs/2309.06794)] [[GitHub](https://github.com/hongbinye/Cognitive-Mirage-Hallucinations-in-LLMs)]

6. **Augmenting LLMs with Knowledge: A survey on hallucination prevention** arXiv (2023.09)

	*Konstantinos Andriopoulos, Johan Pouwelse* [[Paper](https://arxiv.org/pdf/2309.16459.pdf)]

7. **A Comprehensive Survey of Hallucination Mitigation Techniques in Large Language Models** arXiv (2024.01)

	*S.M Towhidul Islam Tonmoy, S M Mehedi Zaman, Vinija Jain, Anku Rani, Vipula Rawte, Aman Chadha, Amitava Das* [[Paper](https://arxiv.org/pdf/2401.01313.pdf)]

###### Factuality<a id="section14"></a>

1. **Trustworthy LLMs: a Survey and Guideline for Evaluating Large Language Models' Alignment** arXiv (2023.08)

	*Yang Liu, Yuanshun Yao, Jean-Francois Ton, Xiaoying Zhang, Ruocheng Guo, Hao Cheng, Yegor Klochkov, Muhammad Faaiz Taufiq, Hang Li* [[Paper](https://arxiv.org/abs/2308.05374)]

2. **A Survey on Factuality in Large Language Models: Knowledge, Retrieval and Domain-Specificity** arXiv (2023.10)

	*Cunxiang Wang, Xiaoze Liu, Yuanhao Yue, Xiangru Tang, Tianhang Zhang, Cheng Jiayang, Yunzhi Yao, Wenyang Gao, Xuming Hu, Zehan Qi, Yidong Wang, Linyi Yang, Jindong Wang, Xing Xie, Zheng Zhang, Yue Zhang* [[Paper](https://arxiv.org/abs/2310.07521)] [[GitHub](https://github.com/wangcunxiang/LLM-Factuality-Survey)]

3. **Give Me the Facts! A Survey on Factual Knowledge Probing in Pre-trained Language Models** arXiv (2023.10)

	*Paul Youssef, Osman Alperen Koraş, Meijie Li, Jörg Schlötterer, Christin Seifert* [[Paper](https://arxiv.org/pdf/2310.16570.pdf)]

#### Attributes of LLMs<a id="section15"></a>

1. **Explainability for Large Language Models: A Survey** arXiv (2023.09)

	*Haiyan Zhao, Hanjie Chen, Fan Yang, Ninghao Liu, Huiqi Deng, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Mengnan Du* [[Paper](https://arxiv.org/abs/2309.01029)]

2. **The Mystery and Fascination of LLMs: A Comprehensive Survey on the Interpretation and Analysis of Emergent Abilitie** arXiv (2023.11)

	*Yuxiang Zhou, Jiazheng Li, Yanzheng Xiang, Hanqi Yan, Lin Gui, Yulan He* [[Paper](https://arxiv.org/pdf/2311.00237.pdf)]

3. **From Understanding to Utilization: A Survey on Explainability for Large Language Models** arXiv (2024.01)

	*Haoyan Luo, Lucia Specia* [[Paper](https://arxiv.org/pdf/2401.12874.pdf)]

4. **A Survey of Large Language Models Attribution** arXiv (2023.11)

	*Dongfang Li, Zetian Sun, Xinshuo Hu, Zhenyu Liu, Ziyang Chen, Baotian Hu, Aiguo Wu, Min Zhang* [[Paper](https://arxiv.org/pdf/2311.03731)] [[GitHub](https://github.com/HITsz-TMG/awesome-llm-attributions)]

5. **A Survey of Language Model Confidence Estimation and Calibration** arXiv (2023.11)

	*Jiahui Geng, Fengyu Cai, Yuxia Wang, Heinz Koeppl, Preslav Nakov, Iryna Gurevych* [[Paper](https://arxiv.org/pdf/2311.08298.pdf)]

6. **Shortcut Learning of Large Language Models in Natural Language Understanding** COMMUN ACM (2023.12 )

	*Mengnan Du, Fengxiang He, Na Zou, Dacheng Tao, Xia Hu* [[Paper](https://dl.acm.org/doi/10.1145/3596490)]

#### Efficient LLMs<a id="section16"></a>

1. **Efficient Large Language Models: A Survey** arXiv (2023.12)

	*Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Jiachen Liu, Zhongnan Qu, Shen Yan, Yi Zhu, Quanlu Zhang, Mosharaf Chowdhury, Mi Zhang* [[Paper](https://arxiv.org/abs/2312.03863)] [[GitHub](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)]

2. **LLM Inference Unveiled: Survey and Roofline Model Insights** arXiv (2024.03)

	*Zhihang Yuan, Yuzhang Shang, Yang Zhou, Zhen Dong, Zhe Zhou, Chenhao Xue, Bingzhe Wu, Zhikai Li, Qingyi Gu, Yong Jae Lee, Yan Yan, Beidi Chen, Guangyu Sun, Kurt Keutzer* [[Paper](https://arxiv.org/abs/2402.16363)]

3. **Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems** arXiv (2023.12)

	*Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Hongyi Jin, Tianqi Chen, Zhihao Jia* [[Paper](https://arxiv.org/abs/2312.15234)]

4. **A Survey on Model Compression for Large Language Models** arXiv (2023.08)

	*Xunyu Zhu, Jian Li, Yong Liu, Can Ma, Weiping Wang* [[Paper](https://arxiv.org/abs/2308.07633)]

5. **A Comprehensive Survey of Compression Algorithms for Language Models** arXiv (2024.01)

	*Seungcheol Park, Jaehyeon Choi, Sojin Lee, U Kang* [[Paper](https://arxiv.org/pdf/2401.15347.pdf)]

6. **A Survey on Knowledge Distillation of Large Language Models** arXiv (2024.02)

	*Xiaohan Xu, Ming Li, Chongyang Tao, Tao Shen, Reynold Cheng, Jinyang Li, Can Xu, Dacheng Tao, Tianyi Zhou* [[Paper](https://arxiv.org/pdf/2402.13116.pdf)]

7. **The Efficiency Spectrum of Large Language Models: An Algorithmic Survey** arXiv (2023.12)

	*Erfan Shayegani, Md Abdullah Al Mamun, Yu Fu, Pedram Zaree, Yue Dong, Nael Abu-Ghazaleh* [[Paper](https://arxiv.org/pdf/2310.10844.pdf)] [[GitHub](https://github.com/tding1/Efficient-LLM-Survey)]

8. **Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment** arXiv (2023.12)

	*Lingling Xu, Haoran Xie, Si-Zhao Joe Qin, Xiaohui Tao, Fu Lee Wang* [[Paper](https://arxiv.org/pdf/2312.12148.pdf)]

9. **Model Compression and Efficient Inference for Large Language Models: A Survey** arXiv (2024.02)

	*Wenxiao Wang, Wei Chen, Yicong Luo, Yongliu Long, Zhengkai Lin, Liye Zhang, Binbin Lin, Deng Cai, Xiaofei He* [[Paper](https://arxiv.org/pdf/2402.09748.pdf)]

10. **Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding** arXiv (2024.01)

	*Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, Zhifang Sui* [[Paper](https://arxiv.org/abs/2401.07851)] [[GitHub](https://github.com/hemingkx/SpeculativeDecodingPapers)]

11. **A Survey on Hardware Accelerators for Large Language Models** arXiv (2024.01)

	*Christoforos Kachris* [[Paper](https://arxiv.org/pdf/2401.09890.pdf)]

#### Learning Methods for LLMs<a id="section17"></a>

1. **Knowledge Unlearning for LLMs: Tasks, Methods, and Challenges** arXiv (2023.11)

	*Nianwen Si, Hao Zhang, Heyu Chang, Wenlin Zhang, Dan Qu, Weiqiang Zhang* [[Paper](https://arxiv.org/pdf/2311.15766)]

2. **Continual Learning with Pre-Trained Models: A Survey** arXiv (2024.01)

	*Da-Wei Zhou, Hai-Long Sun, Jingyi Ning, Han-Jia Ye, De-Chuan Zhan* [[Paper](https://arxiv.org/pdf/2401.16386)] [[GitHub](https://github.com/sun-hailong/LAMDA-PILOT)]

3. **Continual Learning for Large Language Models: A Survey** arXiv (2024.02)

	*Tongtong Wu, Linhao Luo, Yuan-Fang Li, Shirui Pan, Thuy-Trang Vu, Gholamreza Haffari* [[Paper](https://arxiv.org/abs/2402.01364)]

4. **If LLM Is the Wizard, Then Code Is the Wand: A Survey on How Code Empowers Large Language Models to Serve as Intelligent Agents** arXiv (2024.01)

	*Ke Yang, Jiateng Liu, John Wu, Chaoqi Yang, Yi R. Fung, Sha Li, Zixuan Huang, Xu Cao, Xingyao Wang, Yiquan Wang, Heng Ji, Chengxiang Zhai* [[Paper](https://arxiv.org/pdf/2401.00812.pdf)]

#### Multimodal LLMs<a id="section18"></a>

1. **Vision-Language Instruction Tuning: A Review and Analysis** arXiv (2023,11)

	*Chen Li, Yixiao Ge, Dian Li, Ying Shan* [[Paper](https://arxiv.org/abs/2311.08172)] [[GitHub](https://github.com/palchenli/VL-Instruction-Tuning)]

2. **Large Language Models Meet Computer Vision: A Brief Survey** arXiv (2023.11)

	*Raby Hamadi* [[Paper](https://arxiv.org/pdf/2311.16673.pdf)]

3. **Foundational Models Defining a New Era in Vision: A Survey and Outlook** arXiv (2023.07)

	*Muhammad Awais, Muzammal Naseer, Salman Khan, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, Fahad Shahbaz Khan* [[Paper](https://arxiv.org/pdf/2307.13721.pdf)] [[GitHub](https://github.com/awaisrauf/Awesome-CV-Foundational-Models)]

4. **Video Understanding with Large Language Models: A Survey** arXiv (2023.12)

	*Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, Ali Vosoughi, Chao Huang, Zeliang Zhang, Feng Zheng, Jianguo Zhang, Ping Luo, Jiebo Luo, Chenliang Xu* [[Paper](https://arxiv.org/pdf/2312.17432.pdf)] [[GitHub](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding)]

5. **Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook** arXiv (2023.10)

	*Ming Jin, Qingsong Wen, Yuxuan Liang, Chaoli Zhang, Siqiao Xue, Xue Wang, James Zhang, Yi Wang, Haifeng Chen, Xiaoli Li, Shirui Pan, Vincent S. Tseng, Yu Zheng, Lei Chen, Hui Xiong* [[Paper](https://arxiv.org/abs/2310.10196)] [[GitHub](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)]

6. **Sparks of large audio models: A survey and outlook** arXiv (2023.08)

	*Siddique Latif, Moazzam Shoukat, Fahad Shamshad, Muhammad Usama, Yi Ren, Heriberto Cuayáhuitl, Wenwu Wang, Xulong Zhang, Roberto Togneri, Erik Cambria, Björn W. Schuller* [[Paper](https://arxiv.org/pdf/2308.12792.pdf)] [[GitHub](https://github.com/EmulationAI/awesome-large-audio-models)]

7. **How to Bridge the Gap between Modalities: A Comprehensive Survey on Multimodal Large Language Model** arXiv (2023.11)

	*Shezheng Song, Xiaopeng Li, Shasha Li, Shan Zhao, Jie Yu, Jun Ma, Xiaoguang Mao, Weimin Zhang* [[Paper](https://arxiv.org/pdf/2311.07594.pdf)]

8. **A Survey on Multimodal Large Language Models** arXiv (2023.06)

	*Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, Enhong Chen* [[Paper](https://arxiv.org/abs/2306.13549)]

9. **Multimodal Large Language Models: A Survey** arXiv (2023.11)

	*Jiayang Wu, Wensheng Gan, Zefeng Chen, Shicheng Wan, Philip S. Yu* [[Paper](https://arxiv.org/pdf/2311.13165.pdf)]

#### Knowledge Based LLMs<a id="section19"></a>

###### Retrieval-Augmented<a id="section20"></a>

1. **Building trust in conversational ai: A comprehensive review and solution architecture for explainable, privacy-aware systems using llms and knowledge graph** arXiv (2023.08)

	*Ahtsham Zafar, Venkatesh Balavadhani Parthasarathy, Chan Le Van, Saad Shahid, Aafaq Iqbal khan, Arsalan Shahid* [[Paper](https://arxiv.org/pdf/2308.13534.pdf)]

2. **A Survey on Retrieval-Augmented Text Generation** arXiv (2022.02)

	*Huayang Li, Yixuan Su, Deng Cai, Yan Wang, Lemao Liu* [[Paper](https://arxiv.org/abs/2202.01110)]

3. **Retrieval-Augmented Generation for Large Language Models: A Survey** arXiv (2024.1)

	*Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo, Meng Wang, Haofen Wang* [[Paper](https://arxiv.org/abs/2312.10997)] [[GitHub](https://github.com/Tongji-KGLLM/RAG-Survey)]

###### Knowledge Editing<a id="section21"></a>

1. **The Contribution of Knowledge in Visiolinguistic Learning: A Survey on Tasks and Challenges** arXiv (2023.03)

	*Maria Lymperaiou, Giorgos Stamou* [[Paper](https://arxiv.org/abs/2303.02411)]

2. **Trends in Integration of Knowledge and Large Language Models: A Survey and Taxonomy of Methods, Benchmarks, and Applications** arXiv (2023.11)

	*Zhangyin Feng, Weitao Ma, Weijiang Yu, Lei Huang, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, Ting liu* [[Paper](https://arxiv.org/pdf/2311.05876.pdf)]

3. **Knowledge Editing for Large Language Models: A Survey** arXiv (2023.1)

	*Song Wang, Yaochen Zhu, Haochen Liu, Zaiyi Zheng, Chen Chen, Jundong Li* [[Paper](https://arxiv.org/pdf/2310.16218.pdf)]

4. **Editing Large Language Models: Problems, Methods, and Opportunities** arXiv (2023.05)

	*Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang* [[Paper](https://arxiv.org/abs/2305.13172)]

#### Extension of LLMs<a id="section22"></a>

###### LLMs with Tools<a id="section23"></a>

1. **Foundation Models for Decision Making: Problems, Methods, and Opportunities** arXiv (2023.03)

	*Sherry Yang, Ofir Nachum, Yilun Du, Jason Wei, Pieter Abbeel, Dale Schuurmans* [[Paper](https://arxiv.org/abs/2303.04129)]

2. **Augmented Language Models: a Survey** arXiv (2023.02)

	*Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ram Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, Edouard Grave, Yann LeCun, Thomas Scialom* [[Paper](https://arxiv.org/abs/2302.07842)]

3. **A Survey on Language Models for Code** arXiv (2023.11)

	*Ziyin Zhang, Chaoyu Chen, Bingchang Liu, Cong Liao, Zi Gong, Hang Yu, Jianguo Li, Rui Wang* [[Paper](https://arxiv.org/pdf/2311.07989)] [[GitHub](https://github.com/codefuse-ai/Awesome-Code-LLM)]

4. **Pitfalls in Language Models for Code Intelligence: A Taxonomy and Survey** arXiv (2023.10)

	*Xinyu She, Yue Liu, Yanjie Zhao, Yiling He, Li Li, Chakkrit Tantithamthavorn, Zhan Qin, Haoyu Wang* [[Paper](https://arxiv.org/pdf/2310.17903.pdf)] [[GitHub](https://github.com/yueyueL/ReliableLM4Code)]

5. **Large Language Models Meet NL2Code: A Survey** arXiv (2022.12)

	*Daoguang Zan, Bei Chen, Fengji Zhang, Dianjie Lu, Bingchao Wu, Bei Guan, Yongji Wang, Jian-Guang Lou* [[Paper](https://arxiv.org/abs/2212.09420)]

###### LLMs and Interactions<a id="section24"></a>

1. **Large Language Models for Robotics: A Survey** arXiv (2023.11)

	*Fanlong Zeng, Wensheng Gan, Yongheng Wang, Ning Liu, Philip S. Yu* [[Paper](https://arxiv.org/abs/2311.07226)]

2. **A Survey on Multimodal Large Language Models for Autonomous Driving** WACV workshop (2023.11)

	*Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Yang Zhou, Kaizhao Liang, Jintai Chen, Juanwu Lu, Zichong Yang, Kuei-Da Liao, Tianren Gao, Erlong Li, Kun Tang, Zhipeng Cao, Tong Zhou, Ao Liu, Xinrui Yan, Shuqi Mei, Jianguo Cao, Ziran Wang, Chao Zheng* [[Paper](https://openaccess.thecvf.com/content/WACV2024W/LLVM-AD/papers/Cui_A_Survey_on_Multimodal_Large_Language_Models_for_Autonomous_Driving_WACVW_2024_paper.pdf)]

3. **LLM4Drive: A Survey of Large Language Models for Autonomous Driving** arXiv (2023.11)

	*Zhenjie Yang, Xiaosong Jia, Hongyang Li, Junchi Yan* [[Paper](https://arxiv.org/abs/2311.01043v3)] [[GitHub](https://github.com/Thinklab-SJTU/Awesome-LLM4AD)]

4. **Automatically Correcting Large Language Models: Surveying the landscape of diverse self-correction strategies** arXiv (2023.08)

	*Liangming Pan, Michael Saxon, Wenda Xu, Deepak Nathani, Xinyi Wang, William Yang Wang* [[Paper](https://arxiv.org/abs/2308.03188)] [[GitHub](https://github.com/teacherpeterpan/self-correction-llm-papers)]

5. **A Survey on Large Language Model based Autonomous Agents** arXiv ( 2023.08)

	*Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, Ji-Rong Wen* [[Paper](https://arxiv.org/abs/2308.11432)] [[GitHub](https://github.com/Paitesanshi/LLM-Agent-Survey)]

6. **The Rise and Potential of Large Language Model Based Agents: A Survey** arXiv (2023.09)

	*Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, Tao Gui* [[Paper](https://arxiv.org/abs/2309.07864)] [[GitHub](https://github.com/WooooDyy/LLM-Agent-Paper-List)]

7. **Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives** arXiv (2023.12 )

	*Chen Gao, Xiaochong Lan, Nian Li, Yuan Yuan, Jingtao Ding, Zhilun Zhou, Fengli Xu, Yong Li* [[Paper](https://arxiv.org/pdf/2312.11970.pdf)]

8. **Large Multimodal Agents: A Survey** arXiv (2024.02)

	*Junlin Xie, Zhihong Chen, Ruifei Zhang, Xiang Wan, Guanbin Li* [[Paper](https://arxiv.org/pdf/2402.15116)] [[GitHub](https://github.com/jun0wanan/awesome-large-multimodal-agents)]

9. **Role play with large language models** arXiv (2023.11)

	*Murray Shanahan, Kyle McDonell & Laria Reynolds* [[Paper](https://www.nature.com/articles/s41586-023-06647-8)]

#### Long Sequence LLMs<a id="section25"></a>

1. **Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey** arXiv (2023.11)

	*Yunpeng Huang, Jingwei Xu, Junyu Lai, Zixu Jiang, Taolue Chen, Zenan Li, Yuan Yao, Xiaoxing Ma, Lijuan Yang, Hao Chen, Shupeng Li, Penghao Zhao* [[Paper](https://arxiv.org/pdf/2311.12351)]

2. **Length Extrapolation of Transformers: A Survey from the Perspective of Position Encoding** arXiv (2023.12)

	*Liang Zhao, Xiaocheng Feng, Xiachong Feng, Bing Qin, Ting Liu* [[Paper](https://arxiv.org/abs/2312.17044)]

#### LLM Applications<a id="section26"></a>

###### Education<a id="section27"></a>

1. **ChatGPT and Beyond: The Generative AI Revolution in Education** arXiv (2023.11)

	*Mohammad AL-Smadi* [[Paper](https://arxiv.org/abs/2311.15198)]

2. **ChatGPT and large language models in academia: opportunities and challenges** arXiv (2023.07)

	*Jesse G. Meyer* [[Paper](https://link.springer.com/article/10.1186/s13040-023-00339-9)]

3. **ChatGPT for good? On opportunities and challenges of large language models for education** arXiv (2023.04)

	*Enkelejda Kasneci , Kathrin Sessler , Stefan Küchemann , Maria Bannert , Daryna Dementieva , Frank Fischer , Urs Gasser , Georg Groh , Stephan Günnemann  Eyke Hüllermeier , Stephan Krusche , Gitta Kutyniok , Tilman Michaeli , Claudia Nerdel , Jürgen Pfeffer , Oleksandra Poquet , Michael Sailer , Albrecht Schmidt , Tina Seidel , Matthias Stadler …Gjergji Kasneci* [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1041608023000195)]

###### Law<a id="section28"></a>

1. **Large Language Models in Law: A Survey** arXiv (2023.11)

	*Jinqi Lai, Wensheng Gan, Jiayang Wu, Zhenlian Qi, Philip S. Yu* [[Paper](https://arxiv.org/abs/2312.03718)]

2. **A short survey of viewing large language models in legal aspect** arXiv (2023.03)

	*Zhongxiang Sun* [[Paper](https://arxiv.org/abs/2303.09136)]

###### Health<a id="section29"></a>

1. **A Survey of Large Language Models in Medicine: Progress, Application, and Challenge** arXiv (2023.11)

	*Hongjian Zhou, Fenglin Liu, Boyang Gu, Xinyu Zou, Jinfa Huang, Jinge Wu, Yiru Li, Sam S. Chen, Peilin Zhou, Junling Liu, Yining Hua, Chengfeng Mao, Chenyu You, Xian Wu, Yefeng Zheng, Lei Clifton, Zheng Li, Jiebo Luo, David A. Clifton* [[Paper](https://arxiv.org/abs/2311.05112)] [[GitHub](https://github.com/AI-in-Health/MedLLMsPracticalGuide)]

2. **Large Language Models Illuminate a Progressive Pathway to Artificial Healthcare Assistant: A Review** arXiv (2023.11)

	*Mingze Yuan, Peng Bao, Jiajia Yuan, Yunhao Shen, Zifan Chen, Yi Xie, Jie Zhao, Yang Chen, Li Zhang, Lin Shen, Bin Dong* [[Paper](https://arxiv.org/abs/2311.01918)] [[GitHub](https://github.com/mingze-yuan/Awesome-LLM-Healthcare)]

3. **Large AI Models in Health Informatics: Applications, Challenges, and the Future** arXiv (2023.03)

	*Jianing Qiu, Lin Li, Jiankai Sun, Jiachuan Peng, Peilun Shi, Ruiyang Zhang, Yinzhao Dong, Kyle Lam, Frank P.-W. Lo, Bo Xiao, Wu Yuan, Ningli Wang, Dong Xu, Benny Lo* [[Paper](https://arxiv.org/abs/2303.11568)] [[GitHub](https://github.com/Jianing-Qiu/Awesome-Healthcare-Foundation-Models)]

4. **A SWOT (Strengths, Weaknesses, Opportunities, and Threats) Analysis of ChatGPT in the Medical Literature: Concise Review** JMIR (2023.11)

	*Daniel Gödde, Author Orcid,  Sophia Nöhl,  Carina Wolf,  Yannick Rupert, Lukas Rimkus,  Jan Ehlers,  Frank Breuckmann,  Timur Sellmann* [[Paper](https://www.jmir.org/2023/1/e49368/)]

5. **ChatGPT in Healthcare: A Taxonomy and Systematic Review** Computer Methods and Programs in Biomedicine (2024.01)

	*Jianning Li, Amin Dada, Behrus Puladi, Jens Kleesiek, Jan Egger* [[Paper](https://www.sciencedirect.com/science/article/pii/S0169260724000087)]

6. **A review of the explainability and safety of conversational agents for mental health to identify avenues for improvement** NCBI (2023.10)

	*Surjodeep Sarkar, Manas Gaur, Lujie Karen Chen, Muskan Garg, Biplav Srivastava* [[Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10601652/)]

7. **Towards a Psychological Generalist AI: A Survey of Current Applications of Large Language Models and Future Prospects** arXiv (2023.12)

	*Tianyu He, Guanghui Fu, Yijing Yu, Fan Wang, Jianqiang Li, Qing Zhao, Changwei Song, Hongzhi Qi, Dan Luo, Huijing Zou, Bing Xiang Yang* [[Paper](https://arxiv.org/abs/2312.04578)]

8. **Large Language Models in Mental Health Care: a Scoping Review** arXiv (2024.02)

	*Yining Hua, Fenglin Liu, Kailai Yang, Zehan Li, Yi-han Sheu, Peilin Zhou, Lauren V. Moran, Sophia Ananiadou, Andrew Beam* [[Paper](https://arxiv.org/abs/2401.02984)]

9. **The utility of ChatGPT as an example of large language models in healthcare education, research and practice: Systematic review on the future perspectives and** arXiv (2023.12)

	*Malik Sallam* [[Paper](https://www.medrxiv.org/content/10.1101/2023.02.19.23286155v1)]

10. **The imperative for regulatory oversight of large language models (or generative AI) in healthcare** arXiv (2023.07)

	*Bertalan Meskó & Eric J. Topol* [[Paper](https://www.nature.com/articles/s41746-023-00873-0)]

11. **A Survey of Large Language Models for Healthcare: from Data, Technology, and Applications to Accountability and Ethics** arXiv (2023.10)

	*Kai He, Rui Mao, Qika Lin, Yucheng Ruan, Xiang Lan, Mengling Feng, Erik Cambria* [[Paper](https://arxiv.org/abs/2310.05694)] [[GitHub](https://github.com/KaiHe-better/LLM-for-Healthcare)]

12. **The Shaky Foundations of Clinical Foundation Models: A Survey of Large Language Models and Foundation Models for EMRs** arXiv (2023.03)

	*Michael Wornow, Yizhe Xu, Rahul Thapa, Birju Patel, Ethan Steinberg, Scott Fleming, Michael A. Pfeffer, Jason Fries, Nigam H. Shah* [[Paper](https://arxiv.org/abs/2303.12961)]

###### Finance<a id="section30"></a>

1. **Large Language Models in Finance: A Survey** ICAIF (2023.11)

	*Yinheng Li, Shaofei Wang, Han Ding, Hang Chen* [[Paper](https://dl.acm.org/doi/10.1145/3604237.3626869)]

###### Game<a id="section31"></a>

1. **Large Language Models and Games: A Survey and Roadmap** arXiv (2024.02)

	*Roberto Gallotta, Graham Todd, Marvin Zammit, Sam Earle, Antonios Liapis, Julian Togelius, Georgios N. Yannakakis* [[Paper](https://arxiv.org/abs/2402.18659)]

2. **Large Language Models and Video Games: A Preliminary Scoping Review** arXiv (2024.03)

	*Penny Sweetser* [[Paper](https://arxiv.org/abs/2403.02613)]

###### NLP Tasks<a id="section32"></a>

1. **Large Language Models for Information Retrieval: A Survey** arXiv (2023.08)

	*Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Haonan Chen, Zhicheng Dou, Ji-Rong Wen* [[Paper](https://arxiv.org/abs/2308.07107)] [[GitHub](https://github.com/RUC-NLPIR/LLM4IR-Survey)]

2. **Large Language Models for Generative Information Extraction: A Survey** arXiv (2023.12)

	*Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, Xian Wu, Yefeng Zheng, Enhong Chen* [[Paper](https://arxiv.org/abs/2312.17617)] [[GitHub](https://github.com/quqxui/Awesome-LLM4IE-Papers)]

3. **Recent Advances in Natural Language Processing via Large Pre-Trained Language Models: A Survey** arXiv (2021.11)

	*Bonan Min, Hayley Ross, Elior Sulem, Amir Pouran Ben Veyseh, Thien Huu Nguyen, Oscar Sainz, Eneko Agirre, Ilana Heinz, Dan Roth* [[Paper](https://arxiv.org/abs/2111.01243)]

###### Software Engineering<a id="section33"></a>

1. **Large Language Models for Software Engineering: Survey and Open Problems** arXiv (2023.10)

	*Angela Fan, Beliz Gokkaya, Mark Harman, Mitya Lyubarskiy, Shubho Sengupta, Shin Yoo, Jie M. Zhang* [[Paper](https://arxiv.org/abs/2310.03533)]

2. **Large Language Models for Software Engineering: A Systematic Literature Review** arXiv (2023.08)

	*Xinyi Hou, Yanjie Zhao, Yue Liu, Zhou Yang, Kailong Wang, Li Li, Xiapu Luo, David Lo, John Grundy, Haoyu Wang* [[Paper](https://arxiv.org/abs/2308.10620)]

3. **Software Testing with Large Language Models: Survey, Landscape, and Vision** arXiv (2024.03)

	*Junjie Wang, Yuchao Huang, Chunyang Chen, Zhe Liu, Song Wang, Qing Wang* [[Paper](https://arxiv.org/abs/2307.07221)]

###### Recommender Systems<a id="section34"></a>

1. **Foundation Models for Recommender Systems: A Survey and New Perspectives** arXiv (2024.02)

	*Chengkai Huang, Tong Yu, Kaige Xie, Shuai Zhang, Lina Yao, Julian McAuley* [[Paper](https://arxiv.org/abs/2402.11143)]

2. **User Modeling in the Era of Large Language Models: Current Research and Future Directions** arXiv (2023.11)

	*Zhaoxuan Tan, Meng Jiang* [[Paper](https://arxiv.org/abs/2312.11518)] [[GitHub]( https://github.com/TamSiuhin/LLM-UM-Reading)]

3. **A Survey on Large Language Models for Personalized and Explainable Recommendations** arXiv (2023.11)

	*Junyi Chen* [[Paper](https://arxiv.org/abs/2311.12338)]

4. **Large Language Models for Generative Recommendation: A Survey and Visionary Discussions** arXiv (2023.09)

	*Lei Li, Yongfeng Zhang, Dugang Liu, Li Chen* [[Paper](https://arxiv.org/abs/2309.01157)]

5. **A Survey on Large Language Models for Recommendation** arXiv (2023.05)

	*Likang Wu, Zhi Zheng, Zhaopeng Qiu, Hao Wang, Hongchao Gu, Tingjia Shen, Chuan Qin, Chen Zhu, Hengshu Zhu, Qi Liu, Hui Xiong, Enhong Chen* [[Paper](https://arxiv.org/abs/2305.19860)] [[GitHub](https://github.com/WLiK/LLM4Rec)]

6. **How Can Recommender Systems Benefit from Large Language Models: A Survey** arXiv (2023.01)

	*Jianghao Lin, Xinyi Dai, Yunjia Xi, Weiwen Liu, Bo Chen, Hao Zhang, Yong Liu, Chuhan Wu, Xiangyang Li, Chenxu Zhu, Huifeng Guo, Yong Yu, Ruiming Tang, Weinan Zhang* [[Paper](https://arxiv.org/abs/2306.05817)] [[GitHub](https://github.com/CHIANGEL/Awesome-LLM-for-RecSys/)]

###### Other<a id="section35"></a>

1. **A Survey of Graph Meets Large Language Model: Progress and Future Directions** arXiv (2023.11)

	*Yuhan Li, Zhixun Li, Peisong Wang, Jia Li, Xiangguo Sun, Hong Cheng, Jeffrey Xu Yu* [[Paper](https://arxiv.org/pdf/2311.12399)]

2. **Large Language Models on Graphs: A Comprehensive Survey** arXiv (2023.12)

	*Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, Jiawei Han* [[Paper](https://arxiv.org/pdf/2312.02783.pdf)] [[GitHub](https://github.com/PeterGriffinJin/Awesome-Language-Model-on-Graphs)]

3. **When Large Language Models Meet Citation: A Survey** arXiv (2023.09)

	*Yang Zhang, Yufei Wang, Kai Wang, Quan Z. Sheng, Lina Yao, Adnan Mahmood, Wei Emma Zhang, Rongying Zhao* [[Paper](https://arxiv.org/abs/2309.09727)]

4. **A Survey of Text Watermarking in the Era of Large Language Models** arXiv (2023.12)

	*Aiwei Liu, Leyi Pan, Yijian Lu, Jingjing Li, Xuming Hu, Xi Zhang, Lijie Wen, Irwin King, Hui Xiong, Philip S. Yu* [[Paper](https://arxiv.org/abs/2312.07913)]

5. **Mathematical Language Models: A Survey** arXiv (2023.12)

	*Wentao Liu, Hanglei Hu, Jie Zhou, Yuyang Ding, Junsong Li, Jiayi Zeng, Mengliang He, Qin Chen, Bo Jiang, Aimin Zhou, Liang He* [[Paper](https://arxiv.org/abs/2312.07622)]

6. **Recent applications of AI to environmental disciplines: A review** SCI TOTAL ENVIRON (2023.10)

	*Aniko Konya, Peyman Nematzadeh* [[Paper](https://www.sciencedirect.com/science/article/pii/S0048969723063325?casa_token=sbh1pxIYyAgAAAAA:f3WytHabl8udc5v8OhRunnwHEemEAwNafzAcP2reVdGKMAJ-4EcJIxwKO4gdE8ozb6ZibbcY2_4)]

7. **Opportunities and Challenges of Applying Large Language Models in Building Energy Efficiency and Decarbonization Studies: An Exploratory Overview** arXiv (2023.12)

	*Liang Zhang, Zhelun Chen* [[Paper](https://arxiv.org/abs/2312.11701)]

8. **Summary of ChatGPT-Related Research and Perspective Towards the Future of Large Language Models** Meta-Radiology (2023.08)

	*Yiheng Liu, Tianle Han, Siyuan Ma, Jiayue Zhang, Yuanyuan Yang, Jiaming Tian, Hao He, Antong Li, Mengshen He, Zhengliang Liu, Zihao Wu, Lin Zhao, Dajiang Zhu, Xiang Li, Ning Qiang, Dingang Shen, Tianming Liu, Bao Ge* [[Paper](https://www.sciencedirect.com/science/article/pii/S2950162823000176)]

9. **AI-Augmented Surveys: Leveraging Large Language Models and Surveys for Opinion Prediction** arXiv (2023.09)

	*Junsol Kim, Byungkyu Lee* [[Paper](https://arxiv.org/abs/2305.09620)]




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

## Acknowledge

We would like to thank the people who have contributed to this project. The core contributors are

*Junhao Ruan, Long Meng, Weiqiao Shan, Tong Xiao, Jingbo Zhu*

