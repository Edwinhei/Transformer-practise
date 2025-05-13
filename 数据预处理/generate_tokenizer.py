import regex
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tokenizers.normalizers import Sequence as NormalizerSequence
import time
from pathlib import Path

# 小型分词器（主要是中英文分词器训练）

# 1. 加载语料到内存
def load_corpus_to_memory(file):
    fileinfo = file
    start_time = time.time()
    if isinstance(file, str):
        with open(Path(__file__).parent.joinpath('./corpus.txt'), "r", encoding="utf-8") as f: 
            fileinfo = f.readlines()
            print(f"已加载{len(fileinfo)}行到内存，耗时{time.time() - start_time:.2f}秒")
            
    return fileinfo


# 2. 训练BPE分词器（完全在内存中）
def train_bpe_tokenizer(corpus, vocab_size=8000, min_frequency=1, special_tokens=["[PAD]","[UNK]","[BOS]","[EOS]","[CLS]", "[SEP]", "[MASK]"]): 
    start_time = time.time()
    
    # 初始化一个BPE分词器
    tokenizer = Tokenizer(models.BPE())
    
    # 设置 normalizer（可选）
    # 如果不需要额外的 normalizer，可以跳过这一步
    tokenizer.normalizer = NormalizerSequence([])
        
    """
    Whitespace 的行为
        英文：Whitespace 对英文非常友好，因为英文单词通常由空格分隔，标点符号也自然分隔了词和句子。
        中文：中文文本没有空格分隔词，Whitespace 会将连续的中文字符视为一个整体，直到遇到空格或标点符号。
        标点符号处理：Whitespace 能正确识别中文标点符号（如 ，、！、。），并将其作为分隔点。这对中文句子分割有一定帮助。
        简单高效：Whitespace 是一种轻量级的预分词方法，计算开销低，适合快速处理大语料。
        
        pre_tokenizers.Whitespace() 对中文分词不完全友好，主要问题在于：
        它无法识别中文词的边界（因为中文没有空格）。
        它依赖标点符号分隔，但如果中文文本没有标点，效果会很差。
        对于英文，Whitespace 非常合适；但对于中文，Whitespace 的预分词方式会导致后续分词（比如 BPE）效果不佳，尤其是在语料中包含大量连续中文文本时。
        
        4. 改进方法：更适合中文的预分词器
            为了让分词器对中文更友好，我们可以调整预分词器的设置，或者结合其他工具。以下是几种改进方法：

            4.1 使用 pre_tokenizers.ByteLevel()
            作用：
            ByteLevel 预分词器将文本按字节（byte）级别处理，直接将每个字符（包括中文字符）拆分为单独的单元。
            对于中文，ByteLevel 会将每个汉字视为一个独立单元（因为中文字符通常是多字节的 UTF-8 编码）。
            示例：
            输入：你好，世界！
            预分词结果：["你", "好", "，", "世", "界", "！"]
            优点：
            按字符拆分，避免了 Whitespace 将连续中文视为一个整体的问题。
            后续 BPE 算法可以根据语料的统计信息合并字符，形成有意义的子词（比如 你好 可能会被合并为一个 token）。
            缺点：
            初始分词粒度较细（按字符），可能增加 BPE 的合并次数，训练时间稍长。
    """
    # 设置预分词器（按空格和标点分词） 对英文友好，对中文并不是很友好
    # tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # 使用 pre_tokenizers.ByteLevel() 将中文按字级别拆分，但是英语单词按字符级别拆分，对中文友好，但对英文不友好（英文适合按词拆分）
    # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    
    """
    UnicodeScripts()：区分中英文脚本（中文用 Han 脚本，英文用 Latin 脚本）。
    Whitespace()：按空格和标点拆分英文（Hello, world! → ["Hello", ",", "world", "!"]）。
    ByteLevel()：按字符拆分中文（你好，世界！ → ["你", "好", "，", "世", "界", "！"]）。
    """
    # 使用 pre_tokenizers.Sequence() 组合 Whitespace 和 ByteLevel，让英文按空格和标点拆分，中文按字符拆分。
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.UnicodeScripts(), 
                                                       pre_tokenizers.Whitespace(), 
                                                       pre_tokenizers.Split(pattern=r"[\u4E00-\u9FFF]", behavior="isolated")])
                                                      #  导致中文乱码，换一个试试
                                                    #    pre_tokenizers.ByteLevel(add_prefix_space=False)])
    
    # 定义BPE训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,# 最小频率，低于此频率的 token 不会加入词汇表
        show_progress=True # 显示训练进度
    )
    
    # 直接在内存中训练
    tokenizer.train_from_iterator(corpus, trainer)
        
    # 设置 post_processor，支持 BERT、GPT 和 T5 规则
    # 使用单一模板，避免多模板问题
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] [BOS] $B [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ]
    )
    
    print(f"BPE 分词器训练完成，耗时 {time.time() - start_time:.2f} 秒")
    return tokenizer

# 3. 保存分词器
def save_tokenizer(tokenizer, path="my_tokenizer"): 
    tokenizer.save(path)
    print(f"分词器已保存到{path}")
    
# 4. 分词测试
def tokenize_text(tokenizer, text):
    encoded = tokenizer.encode(text)
    tokens = encoded.tokens
    # 手动解码 token，确保中文正确显示
    decoded_tokens = [token.encode().decode('utf-8', errors='replace') for token in tokens]
    return decoded_tokens

# 5. 主函数
def gen_tokenizer(corpus, path): 
    # 加载语料到内存
    corpus = load_corpus_to_memory(corpus)
    
    print(f"使用 {len(corpus)} 行进行训练")
    
    # 训练BPE分词器
    tokenizer = train_bpe_tokenizer(corpus, vocab_size=32000)
    
    # 保存分词器
    save_tokenizer(tokenizer, path)

    
if __name__ == "__main__":
    src_path = "corpus.txt"
    tokenizer_path = "translation.json"
    gen_tokenizer(src_path, tokenizer_path)
