# CLAUDE.md — 毕业论文项目规则

## 项目背景

本项目是一个**医学大语言模型幻觉评测系统**，基于 MED-HALT 数据集，研究大模型在医学推理（Reasoning）和信息检索（Information Retrieval）任务上的幻觉行为。项目代码将作为毕业论文的实验支撑，对**准确性、可复现性和规范性**要求极高。

### 七个评测任务

| 类型 | 任务 | 说明 |
|------|------|------|
| Reasoning | FCT | 功能性正确性判断 |
| Reasoning | Fake | 虚假题目识别 |
| Reasoning | NOTA | None of the Above 判断 |
| IR | pmid2title | PMID → 论文标题 |
| IR | pubmedlink2title | PubMed URL → 论文标题 |
| IR | title2pubmedlink | 论文标题 → PubMed URL |
| IR | abstract2pubmedlink | 论文摘要 → PubMed URL |

---

## 语言与沟通

- **所有回复使用中文**，包括解释、建议、错误分析。
- 代码注释和文档字符串使用**中文**（变量名、函数名保持英文）。
- 引用文件或代码位置时使用 `文件路径:行号` 格式。

---

## 代码规范

### Python 风格

- 遵循 **PEP 8**，缩进使用 4 个空格，行宽不超过 100 字符。
- 函数、类使用**类型注解**（Type Hints），返回值类型必须标注。
- 公开函数必须有 **docstring**，说明参数、返回值和异常。
- 使用 `pathlib.Path` 处理路径，禁止硬编码路径分隔符。