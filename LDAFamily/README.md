# 用来实现论文 Automatic_Labeling_of_Topic_Models_Using_Text_Summaries.pdf
- 文件目录:  
----- data/ : 用来在数据预处理阶段生成的一些文件，如分割title的结果.   
----- docs/ : 用来存储原始的数据 具体形式在代码里修改，但一定不少于 title， content 这两个字段，还有分词表lexicon.txt ,一行一个词 。还有停词表，也是一行一个词.  
----- ldaresult/ : 用来存储LDA的结果probility.txt主题词概率文件，还有topics.txt,主题词文件  
----- model/: 用来存储训练的tfidf模型和lda模型
- 代码： autonameforLDA.py
