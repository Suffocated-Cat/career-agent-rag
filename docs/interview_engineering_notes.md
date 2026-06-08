# CareerAgent — 工程决策面试提纲（详解版）

> ⚠️ **本文档由 AI（Claude）基于源码生成，仅供学习参考。** 参数/公式取自代码，但解读、行业对比、"为什么"的论证可能有疏漏或简化——面试前请结合源码与官方资料自行核实，标注"行业全景/延伸"的内容尤其要分清哪些是项目里真有的、哪些是外部知识。

> 目标：每个**架构选择、模型选型、参数取值、阈值、公式**都能讲清"为什么这样 + 调大调小的影响 + 行业替代方案"。
> 所有数字/模型/阈值均来自源码。读法建议：先通读"一、整体架构"建立心智模型，再按模块深挖；每节末尾的 ⚠️ 是高频追问点。

---

# 一、整体架构

## 1.1 分层：四入口 → 确定性内核 → 可选 LLM 层

```
入口层   REST API(/api/v1) · ReAct Agent · MCP server · career-match CLI
           （四种"调用同一套服务"的方式，不是流水线四个阶段）
              │
确定性内核   JD/简历解析 · BM25+向量+hybrid(+rerank) 检索 · 多信号技能匹配
           · 项目相关性排序 · 规则审计   ← 拥有全部分数/排名/结论
              │
可选 LLM 层   字段抽取 · 风险建议 · 把结构化证据叙述成报告（schema 校验 + 确定性兜底）
              │
输出        匹配分 · 技能缺口 · 排序后的经历 · 风险项 · 报告
```

**核心设计原则（整个项目的灵魂，必背）：**
> 确定性内核永远算数字，LLM 只在上面**抽取、建议、叙述**——LLM 解释证据，不取代分数。

**为什么这么分层（讲三点）：**
1. **可信/可审计**：招聘是高风险决策，分数必须可复现、可追溯，不能由不确定的 LLM 生成。
2. **可测试**：分数是确定函数 → 490 个测试能断言精确值；分数若来自 LLM 只能做模糊断言。
3. **降级可用**：每个 LLM 调用都有 deterministic fallback，没有 API key（`--no-llm`）整条链路照常出结果，LLM 是增强非依赖。

**为什么"四入口共享内核"而不是各写各的：** REST / Agent / MCP / CLI 只是**暴露方式**不同，业务逻辑只有一份（`analyze_match` 是 `/match` 端点和 CLI skill 共享的唯一编排点，"so they can't drift apart"）→ 改一处所有入口受益（DRY、单一职责）。

⚠️ **追问：为什么不让 LLM 端到端做匹配？** → 不可复现、不可审计、贵、会幻觉。确定性内核给下界保证，LLM 只在上面加可读性。

## 1.2 贯穿全项目的三个"反 LLM 单点故障"模式
这三个模式在几乎每个 service 里重复出现，是项目的工程主线，讲任何模块都能回扣：
1. **Primary + Fallback 双轨**：规则/确定性算法是主，LLM/embedding 是增强；增强失败就回退主轨。
2. **Schema 校验**：所有 LLM 抽取结果用 Pydantic 校验，校验失败即视为失败 → 回退。
3. **Graceful degradation**：没有 embedding、没有 LLM、没有 DB，系统都能跑（功能降级但不崩）。

## 1.3 关键设计模式（说出模式名显专业）

| 模式 | 在哪 | 解决什么 |
|---|---|---|
| **Strategy（策略）** | `factory.build_retriever("hybrid+rerank")` + 统一 `Retriever` 协议 | 检索后端按字符串可换 → 直接支撑消融实验 |
| **Adapter/统一接口** | 内存 retriever 与 `PgVectorRetriever` 同接口、同 filter-then-rank 行为 | 上层无感切换存储 |
| **Dependency Injection** | LLMClient/Reranker/Embedding 都可注入预建实例 | 测试不打网络、不加载真模型；可共享单例 |
| **Lazy Loading** | 重对象首次用才加载（cross-encoder、OpenAI client、SentenceTransformer、psycopg） | 启动快、按需付内存、离线测试不依赖它们 |
| **Facade** | `skills/career_match.py` 把整条链路包成一个端到端调用 | 给 CLI/最终用户简单入口 |
| **Graceful Degradation** | `llm_support` 每个 LLM 调用带 `fallback=` | 外部依赖挂了系统不崩 |

## 1.4 数据契约：Pydantic 贯穿
JD/Resume/Match/Audit 全是 Pydantic 模型。**为什么：** ①入口校验请求体 ②LLM 抽取结果用 schema 校验（抽歪了降级而非崩）③API 自动生成 OpenAPI 文档 ④MCP 工具用它把 dict ↔ 模型互转。一个模型多处复用。

---

# 二、数据解析层（JDParser / ResumeParser）

被很多人忽略，但其实是"多策略融合"的好例子，能体现工程判断。

## 2.1 三级解析策略（Primary + Fallback + LLM override）
解析一份 JD/简历的字段（title/skills/responsibilities…），按可靠性递进：

```
有 LLM？  ── 是 ──►  LLM 抽取(schema 校验)  ──失败──►  ↓ 回退规则管线
   │否
   ▼
1. 正则切分章节(主)  ──信号不足──►  2. embedding 把段落分类到章节(兜底)
3. 关键词词表匹配技能(主) ──────►  4. embedding 语义发现技能(兜底)
5. 列表项抽取 responsibilities / nice_to_have
```

**为什么这样设计（核心讲法）：**
- **规则优先**：正则/词表**确定、零成本、可解释**，能覆盖 80% 结构化 JD。
- **embedding 兜底**：JD 格式千奇百怪，正则切不出章节时，用段落与"章节描述锚点"的 cosine 相似度软分类。
- **LLM 顶配**：配了 LLM 就用它一把抽全（最灵活），但**仍以规则管线为安全网**——LLM 挂了不影响可用性。
- 一句话："**确定性方法做主力保证可靠，语义/LLM 做增量提升覆盖率，三层都有退路。**"

## 2.2 技能词表（`TECH_SKILLS`，几百个）
手维护的技能词汇表 + 别名映射（`react.js`/`reactjs` → `react`）。
**为什么用词表而不是纯 embedding：** 技能是**封闭、可枚举**的术语集，词表精确、零误报、零成本；embedding 只用来"捞"词表没覆盖的近义表达。两者互补——又是一次"精确 + 语义"的组合（和检索里 BM25+向量同一思想）。

## 2.3 关键阈值（`_embedding_helpers.py`）
| 阈值 | 值 | 含义 | 调高/调低影响 |
|---|---|---|---|
| `SECTION_SIMILARITY_THRESHOLD` | 0.22 | 段落归到某章节的最低 cosine | 高→更多段落进 "preamble" 未分类；低→易错分类 |
| `SKILL_DISCOVERY_THRESHOLD` | 0.42 | embedding 发现技能的最低相似度 | 高→漏掉近义技能(精确率↑召回率↓)；低→误判普通词为技能 |
| `KEYWORD_BOOST` | 0.06 | 段落含章节提示词时给的小加分 | 解决"语义模糊但有明显关键词"的边界 case |

⚠️ **追问：阈值怎么定的？** → 经验值，针对 `all-MiniLM-L6-v2` 这个模型的相似度分布手调；换 embedding 模型要重标。0.22 低是因为段落长、跨章节语义本就接近；技能 0.42 高是因为技能短、需更高门槛防误报（呼应下面 VectorMatcher 的 0.55）。

---

# 三、匹配打分（KeywordMatcher / VectorMatcher）

这是"分数怎么来的"，面试官最爱问"你这个匹配分到底怎么算的"。

## 3.1 多信号加权打分（`keyword_matcher.py`）
最终 `overall_score` 是**多个信号的加权和**，且**按可用信号分支**（权重各自 sum=1.0）：

| 模式 | 公式 |
|---|---|
| 有 embedding + 有经验信号 | `0.75·技能覆盖 + 0.15·经验↔职责对齐 + 0.10·文档语义相似` |
| 有 embedding 无经验信号 | `0.85·技能覆盖 + 0.15·文档语义相似` |
| 纯关键词(无 embedding) | `0.90·技能覆盖` |

**设计要点（必讲）：**
1. **技能覆盖是主信号（0.75~0.90）**：匹配的核心永远是"技能对不对得上"，其余是辅助语境。
2. **为什么纯关键词模式只给 0.90 不给 1.0**："reserve headroom for absent signals"——预留 10% 给拿不到的语义信号，避免无 embedding 时虚高，让分数在不同模式间可比。
3. **技能覆盖是叠加的**：精确匹配 + 在简历原文里找到 + 语义匹配，三种命中都算 → 满覆盖不会因"没有额外语义匹配"被扣分。

⚠️ **追问：为什么不用 LLM 直接打分？** → 同 1.1：要可复现可测试。加权公式每个权重都能解释、能被单测断言。
⚠️ **追问：权重怎么定的？** → 领域先验（技能 > 经验对齐 > 文档相似），不是学出来的；想更严谨可以用带标注的匹配数据做 LTR 学权重（这是"如何改进"的好答案）。

## 3.2 语义匹配的两个阈值（`vector_matcher.py`）
```
DEFAULT_SKILL_THRESHOLD      = 0.55   # 技能↔技能
DEFAULT_EXPERIENCE_THRESHOLD = 0.50   # 职责↔经验
```
**为什么技能阈值(0.55)比经验阈值(0.50)高（很能体现理解）：**
> 技能是短文本(1~3 词)，短文本的 cosine 容易虚高、需更高门槛防误报；经验是长文本，中等 cosine 就已经有意义。**文本越短，相似度门槛要越高。**

**工程实现细节：** 所有文本**一批 encode**，再算 cross-similarity 矩阵 `(n_query, n_cand)`，每个 query 取最高分候选 → 矩阵一次算完，不循环调模型。先跳过精确匹配的技能（那些归关键词处理），只对"还没匹配上的"做语义补救。

## 3.3 项目相关性排序 + min-max 归一（`match_pipeline.py`）
把 JD 的 skills+responsibilities 拼成一个 query，把简历每条经历/项目当文档，跑检索器排序。返回时做 **min-max 归一**：`(score-lo)/(hi-lo)`，最佳=1.0。
**为什么归一化：** 不同检索方法（BM25 无界 / cosine 0~1 / cross-encoder 任意尺度）原始分不可比，归一化后给前端一个统一的 0~1 相对相关度。同时**保留原始分**(`score`)供调试。`span==0`(全同分)时全给 1.0 防除零。

---

# 四、检索体系（核心，最易被深挖）

## 4.1 Embedding 模型：`all-MiniLM-L6-v2`
**是什么：** sentence-transformers 句向量模型，**6 层 Transformer，384 维**，CPU 上跑。

**为什么选它（选型权衡）：**
| 维度 | all-MiniLM-L6-v2（选中） | 更大模型(mpnet 768 / bge-large 1024) |
|---|---|---|
| 维度 | 384 | 768~1024 |
| 速度 | 快，CPU 可接受 | 慢，CPU 吃力 |
| 质量(MTEB) | 同体积里很强 | 高几个点 |
| 存储/索引 | 向量小，省内存省 pgvector 空间 | 翻倍 |

→ "Docker/CPU 上的原型，语料是简历条目+面试题（短文本），384 维是质量/速度/成本的甜点。"

**调大（768/1024 维）影响：** 召回质量略升（长难句/跨域语义），但编码慢一倍、pgvector 向量列翻倍、CPU 延迟明显、`db/connection.py` 里 `EMBED_DIM=384` 和 HNSW 索引都要改。**调小/更轻：** 更快但语义分辨力下降。

**Embedding 模型全景（备追问）：** 更强开源 `bge-large`/`gte-large`/`e5-large`/`mpnet`(768)；多语/中文 `bge-m3`/`text2vec`；API 托管 OpenAI `text-embedding-3-*`/Cohere(省事但联网/付费/数据出境)。选型看：语料语言、文本长度、规模、是否本地化、质量 vs 成本；MTEB 榜单做基准。

⚠️ 384 维够吗？→ 短文本足够；维度是表达力上限不是越高越好，小语料高维反而稀疏。为什么 CPU？→ 原型/可复现优先，上生产切 `EMBEDDING_DEVICE=cuda`（已是配置项）。

## 4.2 相似度：归一化 + 点积 = cosine
**代码（`embedding.py`）：** 每个向量除以模长（归一化成单位向量），再算点积。
- **为什么不用欧氏距离：** 关心**语义方向**而非长度；cosine 对长度不敏感。
- **为什么归一化后用点积：** 归一化后 `A·B` 就等于 cosine，但点积能一次矩阵乘批量算完所有候选 → 快。数学等价但工程更优，很加分。这个技巧在 `_embedding_helpers`、`vector_matcher` 里反复出现。

## 4.3 BM25：`k1=1.5`, `b=0.75`（手写 Okapi BM25）
```
score(D,Q) = Σ_t IDF(t)·( f(t,D)·(k1+1) ) / ( f(t,D) + k1·(1 - b + b·|D|/avgdl) )
IDF(t) = ln( 1 + (N - df + 0.5)/(df + 0.5) )
```
**为什么手写不用库：** 语料小、要零依赖、要能讲清公式。
- **`k1`（词频饱和，典型 1.2–2.0）**：一个词出现多次后收益衰减多快。调大→更看重重复出现；调小→更快饱和。
- **`b`（长度归一，0–1）**：对长文档的惩罚。`b=1` 完全按长度归一，`b=0` 不归一，`0.75` 是 IR 经典经验值。
- **为什么需要 BM25（既然有向量）：** 精确技术词（PyTorch/Docker/K8s）向量会"糊"在一起，BM25 做精确匹配 → 这是 hybrid 存在的理由。

## 4.4 Hybrid 融合：RRF，`rrf_k=60`，`candidate_pool=50`
**结构：** 同语料同时跑 BM25+向量，融合排名。两种策略：
- **RRF（默认）`score = Σ w·1/(rrf_k+rank)`**：按排名融合，不看原始分。
- **weighted**：min-max 归一到 [0,1] 再加权和。

**为什么默认 RRF：** BM25 分数无上界、cosine 0~1，**量纲不同**，直接加权要先归一化、对候选池敏感。RRF 只用排名 → 免归一化、更鲁棒。
**`rrf_k=60`：** RRF 论文经典默认。调小(如10)→ 头部排名主导；调大→ 各排名权重拉平。
**`candidate_pool=50`：** 融合前每路各取多少候选；太小漏召回，太大变慢。
**融合范式全景：** RRF（免归一化、稳）/ 加权归一（可调但敏感）/ LTR 学习排序（要训练数据）。

## 4.5 Reranker：cross-encoder `ms-marco-MiniLM-L-6-v2`
**bi-encoder vs cross-encoder（必考）：**
- 上游 BM25/向量是 **bi-encoder**：query 和 doc **分开**编码 → 可预存向量、便宜，但不交互、粗。
- reranker 是 **cross-encoder**：`(query,doc)` **拼一起**过模型，每个 query token attend 每个 doc token → 准得多，但不能预存、每 pair 实时算 → 贵。

**结构（两阶段检索）：** `便宜检索召回 top-20 → cross-encoder 重排 → top-k`。只重排候选池因为 cross-encoder 是 O(候选数) 实时前向，全库跑不起。
**模型选 ms-marco-MiniLM 原因：** 在 MS MARCO 段落排序任务训练（正是 query→相关段落场景），同样 6 层 MiniLM，延续"小而够用、CPU 能跑"。
**`candidate_pool=20`（factory 默认）影响：** 调大(100)→ 召回更全但 cross-encoder 算 100 pair，**延迟线性涨**；调小→ 快但上游漏的好文档救不回。经典 **recall vs latency** 权衡。
**工程细节：** cross-encoder 很重，**lazy 加载**，首次 rerank 才加载，可注入预建实例。
**重排范式全景：** cross-encoder（准/贵）/ ColBERT 后期交互（token 级但可预存、规模友好）/ LLM-as-reranker（最强最贵）。

---

# 五、RAG 与向量存储

## 5.1 真正的 RAG 路径：interview_prep（`interview_prep.py`）
这是项目里**最纯粹的 RAG**：检索面试题 KB → LLM **grounded** 在检索到的题目上写备考指南，无 LLM 则退化为"列出检索到的题 + 缺口"。

**最值得讲的设计——per-skill 检索：**
> 不用一个大 query 检索，而是**每个技能单独查、各取 top-2、去重后截断到 k**。
> 原因（代码注释）："a single combined query lets a few skills dominate the top-k and starves the rest"——合并 query 会让少数强信号技能霸占 top-k，饿死其它技能。逐技能检索保证**每个要求技能都被覆盖**。这是 RAG 检索质量的一个实战 trick。

## 5.2 pgvector 持久化 + DB schema（`db/connection.py`）
真实建表语句（能背下来很加分）：
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE knowledge_doc (
    id serial PRIMARY KEY, doc_id text UNIQUE, skill text, doc_type text,
    text text, metadata jsonb DEFAULT '{}', embedding vector(384)
);
CREATE INDEX ... USING gin (metadata);                       -- 元数据过滤
CREATE INDEX ... USING hnsw (embedding vector_cosine_ops);   -- 向量近邻
```
**两个索引，对应两种查询：**
- **HNSW + `vector_cosine_ops`**：向量列建 HNSW 近似最近邻索引，cosine 距离算子 `<=>` 查询；这就是"向量检索为什么快"的答案落地。
- **GIN on jsonb**：metadata 建 GIN 索引 → 支持"先按 `role/difficulty/tags` 过滤再向量排序"(filter-then-rank)。
**ingest 流程：** `scripts/ingest_kb.py` 把每条 KB embed 后 upsert（`ON CONFLICT DO UPDATE`）→ 可重复运行不重复插入。
**lazy import：** psycopg/pgvector 在连接函数内才 import → 离线测试和不碰 DB 的代码无需安装它们。

## 5.3 专用向量数据库全景（备追问）

| 方案 | 类型 | 优势 | 劣势 | 适用 |
|---|---|---|---|---|
| **pgvector（选中）** | Postgres 扩展 | 一库存向量+元数据+业务数据；事务/SQL/JOIN；运维零新增；metadata 过滤天然 | 亿级性能/水平扩展不如专用库；ANN 索引选项少 | 中小规模、已有 PG、要过滤 |
| **Pinecone** | 托管 SaaS | 全托管免运维、弹性扩展、低延迟 | 闭源、贵、数据出本地、厂商锁定 | 不想运维、快上生产 |
| **Weaviate** | 开源/托管 | 自带 hybrid、GraphQL、模块化 embedding | 资源占用高、运维门槛 | 要开箱 hybrid |
| **Milvus** | 开源 | 十亿级、多 ANN 索引、GPU、分布式 | 架构重(etcd/MinIO)、运维复杂 | 超大规模、有团队 |
| **Qdrant** | 开源(Rust) | 性能好、过滤强、内存高效、易部署 | 生态较新 | 要性能又不想要 Milvus 重 |
| **Chroma** | 嵌入式 | 极简、本地、原型友好 | 不适合生产规模 | demo/原型 |
| **Faiss** | 库(非DB) | ANN 标杆、极快 | 无持久化/过滤/服务化 | 离线大规模相似度 |

**为什么本项目选 pgvector：** ①规模小（KB 约 145 条文档、~30 个主题）②metadata 过滤是刚需，一条 SQL 搞定 ③已有 Postgres、零新增运维 ④向量与业务数据同库、事务一致。**何时换：** 千万/亿级、要水平扩展、高 QPS → Milvus/Qdrant/Pinecone。

## 5.4 底层必懂：ANN 索引（追问"向量检索为什么快"）
精确最近邻是 O(N) 全扫；生产用**近似最近邻(ANN)**牺牲一点召回换提速：
- **HNSW**（分层图，本项目用）：查得快、召回高、**内存大**——多数库默认。
- **IVF**（倒排+聚类）：只看最近几个桶，省内存，召回看 `nprobe`。
- **Flat**：精确但慢，小数据用。
- **PQ**（乘积量化）：压缩向量省内存、略损精度，常与 IVF 组合(IVF-PQ)。
核心是**召回率 ↔ 速度 ↔ 内存**三角；本项目规模小，HNSW 足够。

---

# 六、Agent 设计（ReAct）

## 6.1 为什么自己手写 ReAct，不用 LangChain/LlamaIndex

| 方案 | 优势 | 劣势 | 本项目为何不选 |
|---|---|---|---|
| **手写 ReAct（选中）** | 完全可控、零黑盒、可逐步测试、prompt 透明 | 自己维护循环/解析/容错 | — |
| LangChain Agent | 生态大、工具多、上手快 | 抽象层厚、版本不稳、调试像拆盲盒、prompt 被隐藏 | 面试要讲清每一步，黑盒讲不清 |
| LlamaIndex | RAG 数据接入强 | 偏数据管线，agent 编排较弱 | 检索已自建 |
| OpenAI 原生 function calling | 模型原生、结构稳 | 绑定特定 provider 工具协议 | 要 provider 无关 |

**一句话：** "我用 STRICT JSON 自定义了 ReAct 协议，provider 无关、每步可解释可测试，不要框架黑盒。"

## 6.2 循环机制（`react_controller.py`）
```
Thought → Action → Observation → Thought → ... → finish → (单独一步) compose 最终答案
```
每步 LLM 返回严格 JSON，三类动作：`<tool>` 调工具 / `ask_user` 暂停问用户 / `finish` 收集完毕。

**值得讲的设计决策：**
1. **`max_steps=8` 步预算**：防无限循环烧 token。调大→处理更复杂任务但成本/延迟/跑飞风险↑；调小→省钱但复杂任务做不完。
2. **工具读写共享 `ReactState`，步骤间只传小输入**：不把整个解析后简历在每步 prompt 来回抄 → **省 token、防上下文爆炸**。
3. **错误即 observation，不抛异常**：工具报错变成 observation 喂回，让 LLM **自我纠错**。这是 ReAct 比固定流水线强的地方——能从错误恢复。
4. **finish 与 compose 分离**：收集证据 / 写答案分开，答案用单独 system prompt（必须 ground 在 observation、不许编）+ **token 流式**输出。
5. **JSON 容错**：解析失败/字段不对 → 塞一条 error step 继续，而非整轮崩；`action_input` 宽松类型，畸形值降级成"无参"。

## 6.3 工具集（12 个，都是 service 的薄包装）
`parse_jd / parse_resume / match / rank_projects / audit / generate_report / kb_search / interview_prep / advise / rewrite_bullet / compare_jds / select_jd`。多 JD 用 `compare_jds` 排序后 `select_jd` 把某个"提升为活跃 JD"——状态机思路管理多目标。

## 6.4 Slash 命令 vs 自由文本（`slash.py`）
- **自由文本** → ReAct agent（灵活、LLM 驱动、贵）。
- **`/match /report /prep /audit /compare`** → **直接调确定性管线**（便宜、可复现、无 LLM 也能用）。
- 两者**共享 session 的 `ReactState`**：slash 算出的结果，后续自由文本提问能看到（反之亦然）。
**设计点：** 给用户"快捷确定路径"和"灵活智能路径"两个档位，按需取舍成本/灵活性。

---

# 七、Agent 记忆设计（高频且体现深度）

## 7.1 短期工作记忆 = `ReactState`（单次 run 内）
所有工具共享的"工作台"：解析后的 JD/resume/match/report、待比较多 JD、KB retriever、embedding service、LLM。工具间传中间结果，避免重复解析、避免大对象进 prompt。**run 结束即弃。**

## 7.2 会话记忆 = `ChatSession`（跨轮，`sessions.py`）
**记忆压缩策略——"滚动摘要 + 最近窗口"：**
```
RECENT_WINDOW = 6   # 最近 3 轮逐字保留
更早对话 → LLM 折叠进 rolling summary；summarized 记录覆盖到第几条
喂给 agent 的上下文 = summary + 最近窗口原文
无 LLM 时 → 降级成截断 2000 字符的纯文本流水账
```

**为什么这样设计（对比四种方案，必考）：**

| 策略 | 做法 | 优势 | 劣势 |
|---|---|---|---|
| 全量历史 | 全塞进 prompt | 不丢信息 | token 爆炸、贵、超窗口 |
| 滑动窗口 | 只留最近 N 条 | 简单省 token | 早期信息丢 |
| **摘要+窗口(本项目)** | 旧的压摘要，新的留原文 | 省 token + 长期事实在 + 近期细节全 | 摘要丢细节、压缩花一次 LLM 调用 |
| 向量记忆(RAG memory) | 历史存向量库按需检索 | 可扩超长对话、按相关性召回 | 复杂、要额外存储、可能召回不全 |

**一句话：** "近期逐字、远期滚动摘要——在 token 成本和信息保真之间取平衡；要扩展就上向量记忆按需召回。"

## 7.3 ask_user 暂停/恢复（可中断可恢复 agent）
agent 中途需要只有用户能给的信息（如简历缺的量化指标），发 `ask_user` → 循环**暂停**，`pending_question/task/steps` 存进 session。用户回复填进最后一步 observation，带累积 steps 再调 `run` → **从断点继续**。能这么做是因为 agent 状态（steps）是纯数据、可序列化。这是 human-in-the-loop agent 标准做法。

---

# 八、LLM 层与输出可控（`llm_client.py` / `llm_support.py`）

## 8.1 客户端：`temperature=0.0`
**为什么 0：** 贪心解码 → 同输入同输出，可复现可测试。本项目 LLM 做抽取/结构化/叙述，要稳定不要创造性。调高(0.7~1.0)→ 多样但抽取不稳、JSON 易跑偏。
**OpenAI-compatible：** 只认 `base_url/api_key/model`，能接任何兼容端点（OpenAI/DeepSeek/本地 vLLM）；对 DeepSeek 关掉 thinking(`extra_body`)。
**工程模式：** client lazy 创建 + 可注入；`UsageTracker` 累计 token → 成本可观测；支持 `stream()` 流式(SSE)。

## 8.2 两个增强函数 + corrective retry（`llm_support.py`，重点）
- `generate_text` → 自由文本，失败回退确定性字符串。
- `generate_model` → JSON 解析进 Pydantic 模型，失败回退确定性对象。

**最值得讲的——corrective retry（纠错重试）：**
> JSON 校验失败时，**不是简单重发**（temperature=0 重发只会复现同样错误），而是把**坏输出 + 具体报错 + 目标 JSON schema** 一起喂回去，并收紧 system prompt 要求"只输出 JSON"，让模型**自我修正**。重试 `retries` 次后仍失败 → 确定性 fallback。
这是把"LLM 当不可靠组件"工程化的典范：可观测错误、有限重试、有兜底、永不抛异常。

**`extract_json` 的鲁棒性：** 能处理纯 JSON、```json 围栏、混在散文里的 JSON（正则定位第一个对象/数组）→ 容忍模型不听话的输出格式。

⚠️ **追问：怎么保证 LLM 输出可控？** 四件套：①schema 校验 ②corrective retry ③确定性 fallback ④temperature=0 ⑤finish/compose 分离（再加一条）。

---

# 九、规则审计（`project_auditor.py`）

"检索回答哪条经历最相关；审计回答这条声明可不可信"——故意用**透明规则不用 LLM**，让每条发现可解释可复现。

**三类发现：**
1. `unsupported_skill`：列了技能但任何经历/项目里都没出现（无证据声明）。
2. `vague_experience`：highlight 声称影响/付出但无量化（无数字/百分比）。
3. `unsupported_project_claim`：项目列了高级技术但描述撑不起。

**两个有意思的工程判断：**
- **两层证据 + 高级声明更严**（`_Evidence`）：普通技能可由"项目技术列表"佐证；但 **advanced 技能（rag/agent/llm/lora/vllm…）必须出现在 prose（经历/项目描述正文）里**——光在技术列表里堆个"RAG"和在技能栏堆一样，都是未经证实的声明。
- **单词 vs 多词不同匹配**：单词技能按 token 相等匹配（防 "java" 命中 "javascript" 这种子串误判）；多词技能按子串匹配。

**风险分公式：** `min(1, Σ严重度权重 / (high权重·单元数))`，按简历规模归一 → 大简历几条小问题不会虚高。严重度权重 high=3/medium=2/low=1。
**LLM 角色：** 只在确定性发现之上**生成"怎么改"的建议**，grounded 在 findings 上，**数字永不受影响**。

---

# 十、评估体系（`eval/`）

## 10.1 检索指标（`metrics.py`，公式要会）
- **Recall@k** = `|相关 ∩ top-k| / |相关|` —— 关心"漏没漏"。
- **MRR** = `1/第一个相关结果的排名` —— 关心"第一个对的排多前"。
- **nDCG@k** = `DCG/IDCG`，`DCG = Σ grade_i / log2(i+2)` —— 带分级相关度、按位置 log 折扣、用理想排序归一到[0,1]，排序质量金标准。

## 10.2 Ablation 消融（`ablation.py`，高加分）
配合 Strategy 的 factory，对 `bm25/vector/hybrid/hybrid+rerank` 各跑同一数据集，输出 Markdown 表对比 Recall@k/MRR/nDCG → **用数据量化"加 rerank/加向量各涨几分"**。讲法："我不是拍脑袋说 hybrid 好，我跑了消融用 nDCG 证明每个组件的边际贡献。"

## 10.3 LLM-as-Judge（`llm_judge.py`）
第二个 LLM 给生成的报告打分，**只对照结构化证据**判：`groundedness`(忠不忠于证据)/`coverage`(用没用全关键证据)/`clarity`，1~5 分，并列出**不被证据支持的声明**（幻觉检查）。
**工程细节：** 走 `generate_model` 硬化路径（schema 校验+纠错重试+fallback）→ judge 失败返回"未评估"verdict 而非抛异常。`evaluated`/`overall` 是派生属性。
⚠️ LLM judge 的坑：会偏好流畅文风、有位置/长度偏置 → 所以 prompt 明确"不要奖励流畅但加了未证实事实的文本"。

## 10.4 延迟度量（`perf.py`）
`LatencyRecorder` 用 context manager 计时，报 **p50/p95/p99/max**。
**为什么看百分位不看均值：** 均值被尾部拉偏，p95/p99 才描述"最差那批用户"的真实体验——SLA/容量规划都看尾延迟。线性插值算分位数。

---

# 十一、MCP（`mcp/`）

**是什么：** 用 FastMCP 把核心能力（parse_jd/parse_resume/match/audit/rank_projects）暴露成 **MCP 工具**，跑成 **stdio server**，任何 MCP host（如 Claude Desktop）能发现并调用。
**设计点：** 工具逻辑放 `tools.py`（纯函数、dict 进 dict 出、零依赖、可单测），`server.py` 只负责用 `@mcp.tool()` 注册 → **逻辑与协议解耦**。embedding service 用 `lru_cache` 建一次，失败回退 None。
**为什么做 MCP：** 同一套确定性内核多一个标准化暴露面，体现"内核与入口分离"的架构价值。

---

# 十二、反复出现的工程模式（讲这些显成熟）
1. **统一接口 + 多实现（Protocol/Strategy）**：retriever/存储全走同接口，可换、可消融。
2. **Primary + Fallback 双轨**：规则主、语义/LLM 增强、增强失败回退。
3. **Lazy loading**：重对象/重依赖首次用才加载，离线测试不需要它们。
4. **依赖注入**：重对象可外部注入预建实例 → 测试不碰网络/真模型。
5. **Schema 校验 + 纠错重试 + 确定性 fallback**：把 LLM 当不可靠组件工程化。
6. **批量向量化 + 矩阵乘**：所有文本一批 encode，cross-similarity 一次矩阵算完。
7. **接口对称**：内存 retriever 与 pgvector 行为对齐(filter-then-rank)，上层无感。

---

# 十三、高频追问速查

| 问题 | 一句话答 |
|---|---|
| 匹配分怎么算的？ | 多信号加权：技能覆盖0.75 + 经验对齐0.15 + 文档相似0.10，按可用信号分支 |
| 为什么纯关键词只给0.90？ | 预留10%给拿不到的语义信号，让不同模式分数可比 |
| 技能阈值0.55为何比经验0.50高？ | 技能是短文本，cosine 易虚高，需更高门槛防误报 |
| 解析怎么做的？ | 规则(主)+embedding(兜底)+LLM(顶配)三级，层层有退路 |
| 为什么不纯 embedding 抽技能？ | 技能是封闭术语集，词表精确零成本，embedding 只捞近义 |
| 为什么 hybrid 不用纯向量？ | BM25 抓精确技术词，向量抓语义改写，互补 |
| RRF 为什么不直接加权分数？ | BM25 无界、cosine 0~1，量纲不同；RRF 只用排名免归一化 |
| rerank 为什么只重排20个？ | cross-encoder 实时算每 pair，全库太贵；recall/latency 权衡 |
| 为什么384维 embedding？ | 短文本足够，CPU/Docker 上质量-速度-成本甜点 |
| 向量检索为什么快？ | HNSW 近似最近邻索引，召回-速度-内存三角权衡 |
| 为什么 pgvector 不用 Pinecone/Milvus？ | 规模小+要 metadata 过滤+零运维；规模上去再换 |
| RAG 检索怎么保证覆盖所有技能？ | per-skill 检索各取 top-2，防少数技能霸占 top-k |
| temperature 为什么0？ | 抽取/叙述要可复现可测试，不要随机 |
| LLM 输出 JSON 错了怎么办？ | 喂回坏输出+报错+schema 纠错重试，仍失败回退确定性对象 |
| 没有 LLM 能跑吗？ | 能，确定性内核算分，LLM 只增强 |
| 为什么不用 LangChain？ | 要可控/可测试/provider 无关，不要黑盒抽象 |
| 对话记忆怎么做？ | 滚动摘要+最近窗口；对比全量/滑窗/向量记忆取舍 |
| Agent 怎么从错误恢复？ | 工具错误转成 observation 喂回，LLM 自我纠错(ReAct 本质) |
| 怎么防 agent 跑飞/烧钱？ | max_steps 预算 + 工具间传小状态省 token + temperature=0 |
| 怎么证明检索方案好？ | factory 换后端做 ablation，用 nDCG/Recall@k 量化 |
| 审计为什么用规则不用 LLM？ | 要可解释可复现；高级声明必须 prose 佐证；LLM 只给改进建议 |
| 延迟为什么看 p95/p99？ | 均值被尾部拉偏，分位数才反映最差用户体验 |
