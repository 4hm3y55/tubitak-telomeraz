# GÜN 1 ÖDEVİ – COLAB’DA 5 SANİYEDE BİTİR
!pip install numpy matplotlib --quiet
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

N_CELLS = 5000
L0 = 10000

np.random.seed(42)
telomeres = np.random.normal(L0, 800, N_CELLS)
telomeres = np.clip(telomeres, 3000, 15000)

plt.figure(figsize=(12,7))
plt.hist(telomeres, bins=60, color='#3498db', edgecolor='black', alpha=0.85)
plt.title('5000 Hücrenin Başlangıç Telomer Uzunluk Dağılımı\n(Harley et al. 1990)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Telomer Uzunluğu (baz çifti)', fontsize=14)
plt.ylabel('Hücre Sayısı', fontsize=14)
plt.axvline(4000, color='red', linestyle='--', linewidth=2, label='Kritik Eşik')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('telomer_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Ortalama: {telomeres.mean():.1f} bp")
print(f"Std     : {telomeres.std():.1f} bp")
print(f"Kritik altı hücre: {np.sum(telomeres < 4000)}")

files.download('telomer_histogram.png')  # otomatik indirir
