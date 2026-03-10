import matplotlib.font_manager

# 列出系统中所有可用的字体
print("系统中可用的字体：")
for font in matplotlib.font_manager.fontManager.ttflist:
    if 'Hei' in font.name or 'YaHei' in font.name or 'Sim' in font.name or 'Microsoft' in font.name:
        print(f"- {font.name}")

# 打印当前的字体配置
import matplotlib.pyplot as plt
print("\n当前字体配置：")
print(f"font.sans-serif: {plt.rcParams['font.sans-serif']}")
print(f"font.family: {plt.rcParams['font.family']}")