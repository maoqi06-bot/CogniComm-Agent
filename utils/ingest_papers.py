import os
import pdfplumber
import re
from pathlib import Path

# 配置路径
# 输入：存放你 ISAC, MIMO, RIS, cvxpy 等 PDF 论文的目录
SOURCE_PDF_DIR = "./my_raw_papers"
# 输出：Agent 的内置知识库目录
OUTPUT_KNOWLEDGE_DIR = "../dm_agent/data/builtin_knowledge"


def clean_text(text):
    """
    对提取的学术文本进行基础清洗：
    1. 修复因换行产生的单词断裂 (e.g., "com-\nmunication" -> "communication")
    2. 将多个空格替换为一个
    3. 移除特殊非打印字符
    """
    if not text:
        return ""

    # 1. 修复跨行连字符单词
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # 2. 移除单行内的换行符，保留段落间的换行符 (假设学术论文段落间至少有一个空行)
    # 这个处理比较粗糙，复杂的 PDF 可能需要更精细的逻辑
    # 简单的做法是把所有单换行换成空格
    text = text.replace('\n', ' ')

    # 3. 移除多余空格
    text = re.sub(r'\s+', ' ', text)

    # 4. 移除由于 PDF 编码问题产生的非法字符
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # 仅保留 ASCII，科研论文通常是英文

    return text.strip()


def convert_pdfs_to_knowledge(source_dir, output_dir):
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        print(f"❌ 错误：源 PDF 目录 '{source_dir}' 不存在。请先创建该目录并放入 PDF 文件。")
        return

    # 支持的文件扩展名
    extensions = ['*.pdf']
    pdf_files = []
    for ext in extensions:
        pdf_files.extend(source_path.glob(ext))

    if not pdf_files:
        print(f"ℹ️ 在 '{source_dir}' 中未找到 PDF 文件。")
        return

    print(f"📂 找到 {len(pdf_files)} 个 PDF 文件，准备开始转换...")

    processed_count = 0
    error_count = 0

    for pdf_file in pdf_files:
        txt_filename = pdf_file.stem + ".txt"
        txt_output_path = output_path / txt_filename

        # 如果 txt 已存在且 PDF 未更新，可选跳过（这里为了简单每次都重写）

        print(f"📖 正在处理: {pdf_file.name} ...")

        try:
            full_text = []
            with pdfplumber.open(pdf_file) as pdf:
                # 学术论文通常第一页有重要元数据，也需要提取
                for page_num, page in enumerate(pdf.pages):
                    # 提取文本
                    page_text = page.extract_text()
                    if page_text:
                        # 可以在这里做一些基于页面的清洗，比如移除页眉页脚（需要知道坐标，较复杂）
                        full_text.append(page_text)

            # 合并所有页文本
            raw_text = "\n".join(full_text)
            # 清洗文本
            final_text = clean_text(raw_text)

            # 保存为 txt 文件
            with open(txt_output_path, "w", encoding="utf-8") as f:
                f.write(final_text)

            print(f"✅ 已成功转换为文本并保存至: {txt_output_path}")
            processed_count += 1

        except Exception as e:
            print(f"❌ 处理 {pdf_file.name} 时出错: {e}")
            error_count += 1

    print("-" * 30)
    print(f"📊 转换完成。成功: {processed_count}, 失败: {error_count}。")
    print(f"💡 现在你可以启动 Agent，RAG 技能会自动加载这些知识。")


if __name__ == "__main__":
    # 1. 手动创建一个存放原始 PDF 的文件夹
    if not os.path.exists(SOURCE_PDF_DIR):
        os.makedirs(SOURCE_PDF_DIR)
        print(f"ℹ️ 已创建 '{SOURCE_PDF_DIR}' 目录。请将你的科研论文 PDF 放入该目录，然后重新运行此脚本。")
    else:
        # 2. 运行转换
        convert_pdfs_to_knowledge(SOURCE_PDF_DIR, OUTPUT_KNOWLEDGE_DIR)