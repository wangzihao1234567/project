import matplotlib.font_manager
import os

# 查找系统中的中文字体
print("查找系统中的中文字体：")
for font in matplotlib.font_manager.fontManager.ttflist:
    if 'YaHei' in font.name or 'SimHei' in font.name:
        print(f"字体名称: {font.name}")
        print(f"字体路径: {font.fname}")
        print()

# 检查字体文件是否存在
font_paths = [
    'C:/Windows/Fonts/msyh.ttf',  # Microsoft YaHei
    'C:/Windows/Fonts/simhei.ttf'  # SimHei
]

for path in font_paths:
    if os.path.exists(path):
        print(f"字体文件存在: {path}")
    else:
        print(f"字体文件不存在: {path}")