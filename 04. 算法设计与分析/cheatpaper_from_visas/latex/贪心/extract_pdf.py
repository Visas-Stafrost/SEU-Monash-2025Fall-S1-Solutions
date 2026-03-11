from pypdf import PdfReader
import sys

def safe_encode(text):
    """移除可能导致编码问题的字符"""
    result = []
    for char in text:
        try:
            char.encode('utf-8')
            result.append(char)
        except:
            # 替换为空格或删除
            continue
    return ''.join(result)

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        # 过滤掉无法编码的字符
        text += safe_encode(page_text) + "\n"
    return text

if __name__ == "__main__":
    pdf_path = r"D:\workspace\algorithm\yzy\期末复习\贪心\算法设计与分析_05_贪心.pdf"
    text = extract_pdf_text(pdf_path)

    # 写入文件而不是打印，避免编码问题
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print(f"提取完成，共 {len(text)} 个字符")
