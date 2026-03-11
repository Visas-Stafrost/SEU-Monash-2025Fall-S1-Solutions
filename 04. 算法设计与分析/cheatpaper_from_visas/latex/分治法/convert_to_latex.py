"""
将分治法题目文本转换为LaTeX格式
"""

import os
import re
from pathlib import Path

# ============ 配置区 ============
INPUT_FOLDER = './problems_txt'  # 输入文件夹
OUTPUT_FOLDER = './tex'          # 输出文件夹
# =================================

def parse_problem(content):
    """
    解析题目内容，返回结构化数据
    """
    lines = content.split('\n')

    # 提取标题
    title = lines[0].strip() if lines else "未命名题目"

    # 提取各个部分
    sections = {}
    current_section = None
    current_content = []

    for line in lines[1:]:  # 跳过标题
        line = line.strip()
        if not line:
            continue

        # 检测新的章节
        if re.match(r'^\d+\.\s*题目重现', line) or line.startswith('1. 题目重现'):
            if current_section:
                sections[current_section] = current_content
            current_section = '题目重现'
            current_content = []
        elif re.match(r'^\d+\.\s*考点定位', line) or line.startswith('2. 考点定位'):
            if current_section:
                sections[current_section] = current_content
            current_section = '考点定位'
            current_content = []
        elif re.match(r'^\d+\.\s*解题思路', line) or line.startswith('3. 解题思路'):
            if current_section:
                sections[current_section] = current_content
            current_section = '解题思路'
            current_content = []
        elif re.match(r'^\d+\.\s*详细步骤', line) or line.startswith('4. 详细步骤'):
            if current_section:
                sections[current_section] = current_content
            current_section = '详细步骤'
            current_content = []
        elif re.match(r'^\d+\.\s*答案呈现', line) or line.startswith('5. 答案呈现'):
            if current_section:
                sections[current_section] = current_content
            current_section = '答案呈现'
            current_content = []
        elif re.match(r'^\d+\.\s*改编思路', line) or line.startswith('6. 改编思路'):
            if current_section:
                sections[current_section] = current_content
            current_section = '改编思路'
            current_content = []
        elif line.startswith(('（1）', '（2）', '（3）', '(1)', '(2)', '(3)', '一、', '二、', '三、')):
            # 子章节标记
            current_content.append(line)
        else:
            current_content.append(line)

    # 保存最后一个章节
    if current_section:
        sections[current_section] = current_content

    return title, sections


def generate_latex(title, sections, problem_num):
    """
    生成LaTeX代码
    """
    latex_lines = []

    # 标题
    latex_lines.append(f"\\section*{{{problem_num}. {title}}}")
    latex_lines.append("")

    # 题目描述
    if '题目重现' in sections:
        latex_lines.append("\\subsection*{题目描述}")
        latex_lines.append("\\begin{itemize}")
        for line in sections['题目重现']:
            if line.strip() and not line.startswith(('（1）', '（2）', '(1)', '(2)')):
                # 清理行首序号
                cleaned_line = re.sub(r'^[\d+\（\）、]+\s*', '', line)
                latex_lines.append(f"    \\item {cleaned_line}")
        latex_lines.append("\\end{itemize}")
        latex_lines.append("")

    # 考点定位
    if '考点定位' in sections:
        latex_lines.append("\\subsection*{考点}")
        content = ' '.join(sections['考点定位'])
        # 清理
        content = re.sub(r'^\d+\.\s*考点定位\s*', '', content)
        latex_lines.append(content)
        latex_lines.append("")

    # 解题思路
    if '解题思路' in sections:
        latex_lines.append("\\subsection*{思路}")
        latex_lines.append("\\begin{itemize}")
        for line in sections['解题思路']:
            if line.strip():
                # 清理行首序号
                cleaned_line = re.sub(r'^[（\(\d+\）\)]+\s*', '', line)
                # 检测是否是关键点列表
                if re.match(r'^[核心性质分治策略边界判断]', cleaned_line):
                    latex_lines.append(f"    \\item \\textbf{{{cleaned_line[:10]}}}：{cleaned_line[10:]}")
                else:
                    latex_lines.append(f"    \\item {cleaned_line}")
        latex_lines.append("\\end{itemize}")
        latex_lines.append("")

    # 算法框架
    latex_lines.append("\\subsection*{算法}")
    latex_lines.append("\\begin{lstlisting}[language=C++, style=cppstyle]")
    latex_lines.append("// TODO: 添加具体算法代码")
    latex_lines.append("// 根据题目描述实现核心逻辑")
    latex_lines.append("\\end{lstlisting}")
    latex_lines.append("")

    # 详细步骤
    if '详细步骤' in sections:
        latex_lines.append("\\subsection*{步骤}")
        latex_lines.append("\\begin{enumerate}")
        for line in sections['详细步骤']:
            if line.strip() and len(line) > 5:
                cleaned_line = re.sub(r'^[（\(\d+\）\)]+\s*', '', line)
                if cleaned_line:
                    latex_lines.append(f"    \\item {cleaned_line}")
        latex_lines.append("\\end{enumerate}")
        latex_lines.append("")

    return '\n'.join(latex_lines)


def convert_all_problems():
    """转换所有题目"""

    # 创建输出文件夹
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

    # 获取所有txt文件并排序
    txt_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')])

    if not txt_files:
        print(f"在 {INPUT_FOLDER} 文件夹中未找到 .txt 文件")
        return

    print(f"找到 {len(txt_files)} 个题目文件")
    print("=" * 60)

    for i, filename in enumerate(txt_files, 1):
        filepath = os.path.join(INPUT_FOLDER, filename)

        try:
            # 读取题目
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析题目
            title, sections = parse_problem(content)

            # 生成LaTeX
            latex_content = generate_latex(title, sections, i)

            # 保存LaTeX文件
            output_filename = f"problem_{i:02d}.tex"
            output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)

            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(latex_content)

            print(f"已转换: {output_filename} - {title[:30]}")

        except Exception as e:
            print(f"转换 {filename} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("=" * 60)
    print(f"转换完成！共生成 {len(txt_files)} 个LaTeX文件")
    print(f"保存位置: {OUTPUT_FOLDER}/")


if __name__ == '__main__':
    print("=" * 60)
    print("分治法题目转LaTeX工具")
    print("=" * 60)
    print()

    convert_all_problems()

    print()
    print("提示:")
    print("  - 检查生成的LaTeX文件")
    print("  - 可能需要手动调整代码部分")
    print("  - 编译前请确保有main.tex主文件")
