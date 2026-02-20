import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 설정 (제공된 테이블 수치 기반)
labels = ['100', '500', '2000', '5000']
x = np.arange(len(labels))  # x축 위치
width = 0.35  # 막대 너비

# Metric 데이터
# acc+cm vs dn+cm 비교 (PSNR, SSIM)
psnr_acc = [23.15, 24.02, 24.52, 24.28]
psnr_dn = [24.53, 24.85, 25.01, 25.09]

ssim_acc = [0.8585, 0.8675, 0.8747, 0.8759]
ssim_dn = [0.8898, 0.8911, 0.8934, 0.8956]

# acc+seg vs dn+seg 비교 (Dice)
dice_acc = [0.6244, 0.6662, 0.6793, 0.6940]
dice_dn = [0.7380, 0.7869, 0.7952, 0.7996]

# 2. 스타일 설정
plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
colors = ['#f28e8e', '#72b6e8'] # 첫 번째 이미지의 핑크, 블루 계열 색상 참고

def format_plot(ax, ylabel, title=None):
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Training samples per task', fontsize=10)
    ax.grid(axis='y', linestyle='-', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, loc='left', pad=10)

# 3. 각 그래프 그리기
# Plot 1: PSNR
ax1.bar(x - width/2, psnr_acc, width, label='acc+cm', color=colors[0])
ax1.bar(x + width/2, psnr_dn, width, label='dn+cm', color=colors[1])
format_plot(ax1, 'PSNR (dB)')
ax1.set_ylim(20, 27) # 데이터 범위에 맞게 조정
ax1.legend(loc='upper left', frameon=False)

# Plot 2: SSIM
ax2.bar(x - width/2, ssim_acc, width, label='acc+cm', color=colors[0])
ax2.bar(x + width/2, ssim_dn, width, label='dn+cm', color=colors[1])
format_plot(ax2, 'SSIM')
ax2.set_ylim(0.8, 0.95)

# Plot 3: Dice Score
ax3.bar(x - width/2, dice_acc, width, label='acc+seg', color=colors[0])
ax3.bar(x + width/2, dice_dn, width, label='dn+seg', color=colors[1])
format_plot(ax3, 'Dice Score')
ax3.set_ylim(0.5, 0.9)
ax3.legend(loc='upper left', frameon=False)

plt.tight_layout()
plt.show()