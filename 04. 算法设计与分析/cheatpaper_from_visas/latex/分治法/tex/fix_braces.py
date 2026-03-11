import re
import glob

def fix_all_braces(filepath):
    """全面修复右大括号问题"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result = []
    in_listing = False

    for i, line in enumerate(lines):
        # 检测lstlisting环境
        if '\\begin{lstlisting}' in line:
            in_listing = True
            result.append(line)
            continue
        elif '\\end{lstlisting}' in line:
            in_listing = False
            result.append(line)
            continue

        # 在lstlisting环境中处理
        if in_listing:
            stripped = line.strip()

            # 如果这一行只包含右大括号（可能前后有空格）
            if re.match(r'^\s*}\s*$', line):
                # 将}添加到上一行末尾
                if result and result[-1].strip():
                    # 清理上一行末尾可能已有的多余}
                    result[-1] = re.sub(r'}+', '}', result[-1].rstrip())
                    # 添加一个}
                    result[-1] = result[-1] + '}'
                # 跳过当前行
                continue
            else:
                # 清理当前行中可能存在的多个连续}}
                line = re.sub(r'}+', '}', line)
                result.append(line)
        else:
            result.append(line)

    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(result)

    print(f"Fixed {filepath}")

# 处理所有文件
files = sorted(glob.glob('期末复习/分治法/tex/problem_*.tex'))
for filepath in files:
    fix_all_braces(filepath)

print("All files fixed!")
