import subprocess
import os

os.chdir(r"D:\workspace\algorithm\yzy\期末复习\贪心")

# 直接运行，不捕获输出
result = subprocess.run(
    ["xelatex", "-interaction=nonstopmode", "main.tex"],
    encoding='utf-8',
    errors='ignore'
)

print(f"\n返回码: {result.returncode}")
if result.returncode == 0:
    print("编译成功！")

    # 检查PDF是否生成
    if os.path.exists("main.pdf"):
        size = os.path.getsize("main.pdf")
        print(f"PDF已生成: {size} 字节")
else:
    print("编译失败")

    # 检查log文件
    if os.path.exists("main.log"):
        with open("main.log", "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            # 查找错误行
            for line in content.split('\n'):
                if '!' in line and 'Error' in line:
                    print(line)
