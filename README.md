## Benchmark
#### [(pass@k) Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)


## CoT Effectiveness
#### [The Curse of CoT: On the Limitations of Chain-of-Thought in In-Context Learning](https://arxiv.org/abs/2504.05081)
🧠 核心发现：
1️⃣ 负面效果：在9大ICL基准测试中，CoT及其变体（ReAct/ToT）全面溃败，直接回答准确率平均高出20.4%！
2️⃣ 模态差距：符号推理任务中，CoT表现最差（落后41.9%）；文本推理任务稍好（落后10.4%）
3️⃣ 数据量陷阱：提供的示例越多，CoT相对表现越差
	
💡 机制解码：
▫️ 显隐双重性：CoT实际是"显式推理"（CoT推理步骤）和"隐式推理"（直接预测）的混合体
▫️ 显式推理的“绊脚石”：LLM在思维链中难以从示例中准确归纳规则，生成错误中间步骤（哪怕最终答案对了！）
▫️ 隐式推理的“救场王”：模型其实偷偷靠“直觉”纠偏，但思维链拉长的上下文距离会削弱这种能力！
🚨 专用推理模型（如Deepseek-R1）即使消耗40倍计算资源，准确率仍不及基座模型（Deepseek-V3）直接回答
	
🚀 结论与未来展望
本研究揭示了Chain-of-Thought在模式化上下文学习中的系统性局限：显式推理的脆弱性与隐式推理的鲁棒性形成了鲜明对比。实验表明，当任务依赖于从少量示例中提取明确规则时，直接回答往往比CoT更可靠——这不仅挑战了"CoT必然提升推理"的普遍认知，更凸显了LLM底层推理机制的复杂性。

#### [Reasoning Models Can Be Effective Without Thinking](https://arxiv.org/abs/2504.09858)
近年来，大型语言模型（LLMs）在推理任务上的表现有了显著提升，这主要归功于它们在生成过程中包含了一个显式的、冗长的“思考”过程。例如，DeepSeek-R1、OpenAI的o1等模型在解决复杂任务时，会先生成一个长的思考链，包含反思、回溯和自我验证等步骤，然后再给出最终的解决方案和答案。这种显式的推理过程被认为有助于提高模型的推理能力，但也导致了推理时的计算成本显著增加，包括更多的token使用和更高的延迟。
研究方法
论文提出了一种名为NoThinking的方法，通过简单的提示（prompting）来绕过显式的思考过程。具体来说，NoThinking通过在模型的解码过程中预填充一个空的“思考”块，直接生成最终的解决方案和答案。这种方法不需要对模型进行额外的训练或奖励信号，而是直接利用现有的推理模型。
实验设计
论文在多个具有挑战性的推理数据集上对NoThinking进行了广泛的评估，包括数学问题解决（AIME 2024、AIME 2025、AMC 2023）、编程（LiveCodeBench）、形式化定理证明（MiniF2F、ProofNet）等任务。实验中，作者比较了NoThinking和传统的思考过程（Thinking）在不同token预算下的表现，并使用pass@k指标来衡量模型在生成k个样本时至少有一个正确答案的概率。

#### [Concise Reasoning via Reinforcement Learning]()
标题：
《Concise Reasoning via Reinforcement Learning》——用强化学习调教大模型，短小精悍也能高精度推理！
	
核心问题：
LLM的推理总爱“长篇大论”，但真的需要这么啰嗦吗？论文发现：长≠准，冗余token反而可能拖后腿！
	
关键发现：
1️⃣ 反常识结论：正确回答往往更短！
	
实验数据打脸传统认知：在MATH、AIME等数学基准上，正确回答的平均长度比错误答案短50%+（见表1）。
	
冗余token可能让模型误入“死胡同”（deadends），比如陷入重复代码或无意义多语言输出。
	
2️⃣ RL训练是双刃剑：
	
初期RL训练会拉长回答（因为PPO算法在负奖励下偏好长文本）。
	
但第二阶段RL微调（用少量可解问题）能显著缩短回答，同时保持甚至提升准确率！
	
技术亮点：
✨ 两阶段RL训练法：
	
Phase 1：专攻难题，练就模型“硬核推理力”。
	
Phase 2：用少量简单题“瘦身”，逼模型学会“用最少token搞定问题”。
💡 仅需8道题微调，1.5B/7B模型的回答长度直降54%和40%，MMLU-STEM准确率反而提升12.5%！（见图3&表2）
	
数学硬核分析：
🔍 PPO损失函数动态表明：
	
λ<1时，正确回答（r>0）会驱动模型缩短文本，错误回答（r<0）则鼓励“水字数”。
	
λ=1会导致训练不稳定（见Appendix图5-6），作者建议λ=0.95更鲁棒。
	
实用价值：
✅ 降本增效：短回答=更少计算资源+更快响应，适合落地场景。
✅ 小数据奇迹：仅需4~8道题微调，非推理模型（如Qwen-Math）准确率飙升30%（表4）。
	
争议点：
⚠️ 现有RLHF流程可能过度鼓励“长回答”，需重新审视奖励设计。

#### [START: Self-taught Reasoner with Tools](https://arxiv.org/abs/2503.04625)

这篇论文提出了START模型，通过结合外部工具和自学习技术，有效缓解了现有大型推理模型的局限性。通过引入提示推理和提示拒绝采样微调，START模型在多个挑战性基准上表现出色，成为高级推理任务的领先开源解决方案。

#### [Speculative Thinking: Enhancing Small-Model Reasoning with Large Model Guidance at Inference Time](https://arxiv.org/abs/2504.12329)

🔍 问题背景：当前的推理模型在处理复杂任务时，常因冗长的输出和低效的自反思而导致性能不佳。尽管大型语言模型（LLM）在推理能力上表现出色，但其高昂的推理成本使其在实际应用中难以推广。
	
💡 研究动机：研究团队提出了一种无需额外训练的框架——Speculative Thinking，通过在推理过程中让大型模型在关键节点指导小型模型，从而在不牺牲效率的情况下显著提升小型模型的推理准确性和输出效率。
	
🚀 方法简介：Speculative Thinking框架利用了“\n\n”等结构化线索来识别模型的反思行为。当小型模型遇到需要反思的推理段落时，框架会自动将该任务交给大型模型处理，从而避免冗余的生成，提高推理的准确性和效率。
	
📊 实验设计：研究团队在四个基准数据集上进行了实验，包括AIME 2022–2024、GPQA、MATH500和AMC23。实验结果显示，在大型模型的帮助下，小型模型的准确性得到了显著提升，同时输出长度显著减少。例如，1.5B模型在MATH500数据集上的准确性从83.2%提高到了89.4%，输出长度减少了15.7%。

#### [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)

RL对推理能力的训练不是真的学会更多技能，而是在推理过程中对步骤猜的更准。即便用没有经过任何rl训练的基础模型，采样足够多次，也能达到甚至超过RL推理模型的水平，出现正确的解答。

pass@k 对于很大的k来说，所有的rl算法得到的模型几乎都是一样的。也就是说，给base足够多次尝试的机会，它能几乎做对rl推理模型能做对的【所有】问题，甚至更多。

简单来说就是，如果你有鉴别答案和cot对错的能力，那么为了得到正确的解答，直接用比较强的base玩命采样2048条解答筛选，和用RL没有什么区别，甚至更好。加大采样，众生平等。

#### [Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://arxiv.org/abs/2504.04736)

🔍 问题背景：传统的RLHF（Reinforcement Learning from Human Feedback）和RLAIF（Reinforcement Learning from AI Feedback）方法在处理多步骤任务时表现不佳。多步骤任务需要语言模型进行多个文本生成、推理和环境交互步骤。研究团队提出了一种新的方法——Step-Wise Reinforcement Learning (SWiRL)，来解决这一问题。
	
💡 研究动机：为了提升多步骤任务的性能，研究团队提出了SWiRL。该方法生成多步骤合成数据，并通过逐步强化学习来优化模型，从而改进多步骤推理和工具使用能力。
	
🚀 方法简介：SWiRL方法分为两个阶段。第一阶段生成多步骤合成数据，第二阶段使用逐步强化学习对生成的数据进行优化。在第一阶段，模型生成多步骤轨迹，每个步骤可以是工具调用或最终答案生成。合成数据经过过程过滤后，用于第二阶段的训练。
	
📊 实验设计：研究团队在多个数据集上进行了实验，包括HotPotQA、GSM8K、MuSiQue、CofCA和BeerQA。实验结果表明，SWiRL在这些数据集上的性能比基线方法分别提高了21.5%、12.3%、14.8%、11.1%和15.3%。此外，SWiRL还表现出跨任务的泛化能力，例如在HotPotQA上训练的模型在GSM8K上的零样本性能提高了16.9%。


## Acceleration
#### [Sleep-time Compute: Beyond Inference Scaling at Test-time](https://arxiv.org/abs/2504.13171)

他们创建了两个推理任务的修改版本——Stateful GSM-Symbolic 和 Stateful AIME。他们发现，在这两个任务上，睡眠时计算可以将达到相同准确度所需的测试时计算量减少约 5 倍；通过调整睡眠时计算的规模，他们可以进一步提高这个两个任务的准确度，分别提高 13% 和 18%。



## RAG Innovation
#### [Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks](https://arxiv.org/abs/2412.15605)

缓存增强生成（CAG）消除了检索延迟并最小化了检索错误，同时保持了上下文相关性。
利用长上下文大型语言模型的扩展上下文能力，将所有相关资源预加载到LLM的扩展上下文中，并缓存运行时参数。在推理时，模型利用这些预加载的参数来回答查询，从而实现无检索知识集成。

#### [KBLaM: Knowledge Base augmented Language Model](https://arxiv.org/abs/2410.10450)
直接知识内化：告别 RAG 的检索延迟

线性计算复杂度：突破上下文长度限制

保持原始模型完整性：适配器修改知识处理流程

#####但上述内容效果一般，不如用 qdrant vector database，已经可以非常快了。

## VLM CoT
#### [https://arxiv.org/abs/2411.19488](https://arxiv.org/abs/2411.19488)
🎉 背景
随着大语言模型（LLMs）扩展到多模态任务，链式思维（CoT）提示被引入以增强模型的推理能力。但当前主流的多模态CoT方法仍依赖文本-only的中间推理步骤，这种方式难以精准表达图像中的细节关系，导致理解偏差或信息缺失。为此，如何将视觉信息更有效地融入推理过程，成为多模态推理研究的新挑战。
	
✨ 方法
为了解决多模态推理中的表达粗糙和信息缺失问题，作者提出了一种全新框架：Interleaved-modal Chain-of-Thought（ICoT），其核心创新点包括：
① 图文交替推理（Interleaved-modal Reasoning）：在中间推理步骤中穿插图像区域（视觉片段）和文字解释，使推理更贴近人类思维流程，提升表达的细粒度和准确性；
② Attention-driven Selection（ADS）机制：利用VLM的注意力图，动态选择最相关的图像区域（而非生成新图像），并插入这些视觉Token以辅助文本生成，中间无需额外训练，具有极强的适配性和效率；
③ Plug-and-Play设计：ADS不依赖任何模型参数修改，可直接适用于不同架构的VLM（如Chameleon 和 Qwen2-VL），轻量高效，几乎不引入额外延迟。
	
🏆 实验
作者在多个多模态推理任务上评估ICoT的效果，结果显示该方法在推理能力与可解释性方面均优于现有方案：
① 显著提升性能：在M3CoT、ScienceQA 和 LLaVA-W 三个基准数据集上，ICoT相较其他方法提升准确率最多达 14%；
② 更高解释性：生成的推理过程不仅包含正确答案，更清晰展示了模型的思维路径和视觉依据，显著缓解误解、泛化过度和“幻想”问题；
③ 验证组件有效性：消融实验表明，ADS机制和精细图像片段（Fine-grained Visual Information, FVI）对性能提升至关重要。自动生成示例虽可行，但人工设计示例仍表现更优。

#### [SoTA with Less: MCTS-Guided Sample Selection for Data-Efficient Visual Reasoning Self-Improvement](https://arxiv.org/abs/2504.07934)
一种增强视觉推理的有效方法，其所需训练样本少，纯粹依靠自我改进，且没有知识提炼。
	
他们认为，强化微调（RFT）过程中训练数据的难度至关重要，即使数据集很小，适当的挑战样本也能大幅提高推理能力。因此，主要的挑战仍然是如何准确量化样本难度，以实现有效的数据筛选。
	
为此，他们提出了重新利用蒙特卡洛树搜索（MCTS）的新方法。从他们策划的 70k 个开源训练样本开始，他们引入了一种基于 MCTS 的选择方法，该方法根据 VLM 解决每个问题所需的迭代次数来量化样本难度。MCTS 中这种明确的分步推理方法能让模型思考更长的时间，从而更好地识别真正具有挑战性的样本。他们筛选并保留了 11k 个样本，在 Qwen2.5-VL-7B-Instruct 上执行 RFT，最终形成了 ThinkLite-VL 模型。
	
对 8 个基准的评估结果表明，ThinkLite-VL 在仅使用 11k 个训练样本且未进行知识提炼的情况下，将 Qwen2.5-VL-7B-Instruct 的平均性能提高了 7%，优于所有现有的 7B 级推理 VLM，也优于他们使用经典选择方法（如基于准确性的过滤）的对比基线。值得注意的是，在 MathVista 上，ThinkLite-VL-7B 实现了 SoTA 准确率 75.1，超过了 Qwen2.5-VL-72B、GPT-4o 和 o1。

## 目前工作
1 CUHK的工作代码完全可以。关于全是鸟的数据集，复现结果显著好于实验报告结果，很奇怪。
	
2 o3, o4-mini 在一些随手尝试的任务上要好于4.1


