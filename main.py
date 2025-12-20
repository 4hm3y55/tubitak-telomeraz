"""
TÜBİTAK 2204-A Projesi
"Telomeraz enzimi potansiyel riskleri ve optimal kullanım tavsiyesi"

v7.3 - tr 
YENİ: 
- Telomeraz kullanımı mutasyon riskini ciddi şekilde artırır
- Düzeltilmiş senesens: Telomeraz YOK -> Hayflick + Telomer limiti
- Diğer stratejiler: Sadece Hayflick limiti (telomer uzayabilir)
- Kanser hücreleri senesense GİRMEZ
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None):
        return iterable

# ===================== BİLİMSEL PARAMETRELER =====================
T0 = 10000
T_KRITIK = 4500
T_KRITIK_VARYASYON = 500
TELOMER_KAYBI = 75
KAYIP_VARYASYONU = 25

TELOMERAZ_ETKINLIGI = 0.8
TELOMERAZ_EKLEME = 60

MUTASYON_ORANI = 0.002  # %0.2 - Çok düşük ama var
KANSER_ESIGI = 6  # Literatüre uygun
MUTASYON_ARTISI_KISA = 2.0  # Kısa telomerde 2x
KANSER_BUYUME_ORANI = 1.5
KANSER_ESIGI_VARYASYON = 1  # ±1

# KRITIK: Telomeraz kullanımı mutasyon riskini ARTTIRIR
TELOMERAZ_MUTASYON_CARPANI = 1.5  # Telomeraz ile mutasyon riski artar

HUCRE_SAYISI = 5000
MAX_NESIL = 80
HAYFLICK_LIMITI = 50
MAX_POPULASYON = 50000
HAYFLICK_LIMITI_VARYASYON = 10

STRATEJI_ETIKETLERI = {
    'hic': 'Telomeraz Yok (Normal)',
    'surekli': 'Sürekli Aktif (Kanser)',
    'erken_patlama': 'Erken Dönem',
    'periyodik': 'Periyodik',
    'optimal': 'Optimal Strateji'
}

# ===================== TELOMERAZ STRATEJİLERİ =====================
def telomeraz_aktivitesi_al(nesil, strateji, **parametreler):
    if strateji == "hic":
        return 0.0
    elif strateji == "surekli":
        return parametreler.get('dose', 1.0)
    elif strateji == "erken_patlama":
        if nesil < 15:
            return 1.0
        return 0.0
    elif strateji == "periyodik":
        periyot = parametreler.get('period', 20)
        sure = parametreler.get('duration', 5)
        doz = parametreler.get('dose', 0.8)
        dongu_konumu = nesil % periyot
        if dongu_konumu < sure:
            return doz
        return 0.0
    elif strateji == "optimal":
        if nesil % 18 < 4:
            return 0.75
        return 0.0
    return 0.0

# ===================== POPÜLASYON BAŞLATMA =====================
def populasyon_baslat(hucre_sayisi):
    populasyon = []
    for _ in range(hucre_sayisi):
        baslangic_telomer = max(8000, np.random.normal(T0, 1000))
        hucre_kritik = max(2000, np.random.normal(T_KRITIK, T_KRITIK_VARYASYON))
        hucre_kanser_esigi = max(3, KANSER_ESIGI + np.random.randint(
            -KANSER_ESIGI_VARYASYON, KANSER_ESIGI_VARYASYON + 1
        ))
        hucre_hayflick = max(30, HAYFLICK_LIMITI + np.random.randint(
            -HAYFLICK_LIMITI_VARYASYON, HAYFLICK_LIMITI_VARYASYON + 1
        ))
        # Son eleman: senesans timer (0 = normal, >0 = remaining gens before death)
        populasyon.append((
            baslangic_telomer, 0, 0, False,
            hucre_kritik, hucre_kanser_esigi, hucre_hayflick, 0
        ))
    return populasyon

# ===================== SİMÜLASYON =====================
def populasyon_simule_et_varyasyonlu(strateji_adi, nesil_sayisi=MAX_NESIL, **strateji_parametreleri):
    populasyon = populasyon_baslat(HUCRE_SAYISI)
    sonuclar = {
        'nesil': [], 'canli_hucre': [], 'ort_telomer': [],
        'kanser_hucre': [], 'yeni_kanser': [], 'senesans_hucre': [],
        'ort_mutasyon': [], 'ort_bolunme': []
    }
    
    for nesil in range(nesil_sayisi):
        telomeraz_aktif = telomeraz_aktivitesi_al(nesil, strateji_adi, **strateji_parametreleri)
        is_hic = (strateji_adi == 'hic')
        
        # Sürekli strateji için kanser işaretlemesi
        if strateji_adi == 'surekli' and telomeraz_aktif > 0.9:
            populasyon = [
                (tel, mut, bol, True, hc_kritik, hc_kans, hc_hf, 0)
                for (tel, mut, bol, _, hc_kritik, hc_kans, hc_hf, _sen) in populasyon
            ]
        
        yeni_populasyon = []
        yeni_kanser_sayisi = 0
        senesans_sayisi = 0
        
        for hucre_veri in populasyon:
            telomer, mutasyon, bolunme, kanser_mi, hucre_kritik, hucre_kanser_esigi, hucre_hayflick, senescent_timer = hucre_veri

            # SENESENS MEKANİZMASI: Senesense giren hücreler bir sonraki nesilde ölür
            if senescent_timer > 0:
                senesans_sayisi += 1
                if senescent_timer > 1:
                    yeni_populasyon.append((
                        telomer, mutasyon, bolunme, kanser_mi, hucre_kritik, 
                        hucre_kanser_esigi, hucre_hayflick, senescent_timer - 1
                    ))
                continue

            # SENESENS TETİKLEYİCİLERİ:
            senesense_gir = False
            
            # KANSER HÜCRELERİ SENESENSE GİRMEZ!
            if not kanser_mi:
                if is_hic:
                    # Telomeraz YOK: Hem telomer hem Hayflick limiti kontrol edilir
                    if telomer < hucre_kritik or bolunme >= hucre_hayflick:
                        senesense_gir = True
                else:
                    # Diğer stratejiler: Sadece telomer limiti (telomeraz telomeri uzatabilir)
                    # Hayflick limiti diğer stratejilerde geçerli değil (telomeraz uzatabilir)
                    if telomer < hucre_kritik:
                        senesense_gir = True
            
            if senesense_gir:
                senesans_sayisi += 1
                yeni_populasyon.append((
                    telomer, mutasyon, bolunme, kanser_mi, hucre_kritik, 
                    hucre_kanser_esigi, hucre_hayflick, 1
                ))
                continue

            # KANSER HÜCRELERİ: Hızlı çoğalma, senesense girmez
            if kanser_mi:
                yavru_sayisi = int(2 * KANSER_BUYUME_ORANI)
                for _ in range(yavru_sayisi):
                    kayip = max(0, np.random.normal(TELOMER_KAYBI * 0.3, KAYIP_VARYASYONU))
                    kazanc = TELOMERAZ_EKLEME * telomeraz_aktif * TELOMERAZ_ETKINLIGI if telomeraz_aktif > 0 else 0
                    yeni_telomer = max(0, telomer - kayip + kazanc)
                    
                    # Kanser hücrelerinde mutasyon
                    yeni_mutasyon = mutasyon
                    mutasyon_olasiligi = MUTASYON_ORANI * 2.0  # Kanser hücrelerinde biraz daha yüksek
                    
                    # Telomeraz kullanımı mutasyonu artırır
                    if telomeraz_aktif > 0:
                        mutasyon_olasiligi *= (1 + telomeraz_aktif * TELOMERAZ_MUTASYON_CARPANI)
                    
                    if np.random.random() < mutasyon_olasiligi:
                        yeni_mutasyon += 1
                    
                    yeni_bolunme = bolunme + 1
                    yeni_populasyon.append((
                        yeni_telomer, yeni_mutasyon, yeni_bolunme, True,
                        hucre_kritik, hucre_kanser_esigi, hucre_hayflick, 0
                    ))
                continue

            # NORMAL HÜCRE BÖLÜNMESI
            for _ in range(2):
                kayip = max(0, np.random.normal(TELOMER_KAYBI, KAYIP_VARYASYONU))
                kazanc = TELOMERAZ_EKLEME * telomeraz_aktif * TELOMERAZ_ETKINLIGI if telomeraz_aktif > 0 else 0
                yeni_telomer = max(0, telomer - kayip + kazanc)

                # MUTASYON HESAPLAMA - Düşük oran ama telomeraz ile artar
                mutasyon_olasiligi = MUTASYON_ORANI  # %0.2
                
                # Kısa telomer mutasyon riskini artırır
                if yeni_telomer < hucre_kritik * 1.5:
                    mutasyon_olasiligi *= MUTASYON_ARTISI_KISA  # 2x
                
                # TELOMERAZ KULLANIMI MUTASYON RİSKİNİ ARTIRIR
                if telomeraz_aktif > 0:
                    mutasyon_olasiligi *= (1 + telomeraz_aktif * TELOMERAZ_MUTASYON_CARPANI)
                    # %0.2 → Tam telomeraz ile %0.5'e çıkar
                
                # 'hic' stratejisinde mutasyon olasılığını daha da azalt
                if is_hic:
                    mutasyon_olasiligi *= 0.1  # Neredeyse hiç kanser

                yeni_mutasyon = mutasyon
                if np.random.random() < mutasyon_olasiligi:
                    yeni_mutasyon += 1

                yeni_kanser_mi = yeni_mutasyon >= hucre_kanser_esigi
                if is_hic:
                    yeni_kanser_mi = False
                elif yeni_kanser_mi and not kanser_mi:
                    yeni_kanser_sayisi += 1

                yeni_bolunme = bolunme + 1
                yeni_populasyon.append((
                    yeni_telomer, yeni_mutasyon, yeni_bolunme, yeni_kanser_mi,
                    hucre_kritik, hucre_kanser_esigi, hucre_hayflick, 0
                ))
        
        populasyon = yeni_populasyon
        
        if len(populasyon) > MAX_POPULASYON:
            hayatta_kalma_agirliklari = [2.0 if hucre[3] else 1.0 for hucre in populasyon]
            toplam_agirlik = sum(hayatta_kalma_agirliklari)
            olasiliklar = [a / toplam_agirlik for a in hayatta_kalma_agirliklari]
            indeksler = np.random.choice(len(populasyon), MAX_POPULASYON, replace=False, p=olasiliklar)
            populasyon = [populasyon[i] for i in indeksler]
        
        if len(populasyon) > 0:
            normal_hucre = [hucre for hucre in populasyon if not hucre[3]]
            kanser_hucre_populasyon = [hucre for hucre in populasyon if hucre[3]]
            toplam_kanser_sayisi = len(kanser_hucre_populasyon)
            
            if len(normal_hucre) > 0:
                telomerler = [hucre[0] for hucre in normal_hucre]
                mutasyonlar_listesi = [hucre[1] for hucre in normal_hucre]
                bolunmeler_listesi = [hucre[2] for hucre in normal_hucre]
                ort_telomer = np.mean(telomerler)
                ort_mutasyon = np.mean(mutasyonlar_listesi)
                ort_bolunme = np.mean(bolunmeler_listesi)
            else:
                telomerler = [hucre[0] for hucre in kanser_hucre_populasyon]
                mutasyonlar_listesi = [hucre[1] for hucre in kanser_hucre_populasyon]
                bolunmeler_listesi = [hucre[2] for hucre in kanser_hucre_populasyon]
                ort_telomer = np.mean(telomerler) if telomerler else 0
                ort_mutasyon = np.mean(mutasyonlar_listesi) if mutasyonlar_listesi else 0
                ort_bolunme = np.mean(bolunmeler_listesi) if bolunmeler_listesi else 0
            
            sonuclar['nesil'].append(nesil)
            sonuclar['canli_hucre'].append(len(populasyon))
            sonuclar['ort_telomer'].append(ort_telomer)
            sonuclar['kanser_hucre'].append(toplam_kanser_sayisi)
            sonuclar['yeni_kanser'].append(yeni_kanser_sayisi)
            sonuclar['senesans_hucre'].append(senesans_sayisi)
            sonuclar['ort_mutasyon'].append(ort_mutasyon)
            sonuclar['ort_bolunme'].append(ort_bolunme)
        else:
            break
    
    return pd.DataFrame(sonuclar)

# ===================== OPTİMİZASYON =====================
def telomeraz_zamanlamasi_optimize():
    sonuclar = []
    periyotlar = range(15, 25, 3)
    sureler = range(3, 7)
    dozlar = [0.7, 0.8, 0.9]
    
    toplam = len(periyotlar) * len(sureler) * len(dozlar)
    ilerleme_cubugu = tqdm(total=toplam)
    
    for periyot in periyotlar:
        for sure in sureler:
            if sure >= periyot:
                continue
            for doz in dozlar:
                veri = populasyon_simule_et_varyasyonlu('periyodik', nesil_sayisi=60,
                                                       period=periyot, duration=sure, dose=doz)
                yasam_suresi = len(veri)
                son_populasyon = veri['canli_hucre'].iloc[-1] if len(veri) > 0 else 0
                ort_kanser = veri['kanser_hucre'].mean() if len(veri) > 0 else HUCRE_SAYISI
                
                yasam_skoru = yasam_suresi / 60
                pop_skoru = son_populasyon / HUCRE_SAYISI
                if son_populasyon > 0:
                    kanser_orani = ort_kanser / son_populasyon
                else:
                    kanser_orani = 1.0
                kanser_skoru = max(0, min(1, 1 - kanser_orani))
                skor = yasam_skoru * 0.4 + pop_skoru * 0.3 + kanser_skoru * 0.3
                
                sonuclar.append({
                    'periyot': periyot, 'sure': sure, 'doz': doz,
                    'yasam_suresi': yasam_suresi, 'son_populasyon': son_populasyon,
                    'ort_kanser': ort_kanser, 'skor': skor
                })
                ilerleme_cubugu.update(1)
    
    ilerleme_cubugu.close()
    return pd.DataFrame(sonuclar)

# ===================== GÖRSELLEŞTİRME =====================
def strateji_karsilastirmasi_goster(sonuclar_sozlugu):
    sekil, eksenler = plt.subplots(2, 3, figsize=(18, 10))
    sekil.suptitle('Telomeraz Stratejileri Karşılaştırması', fontsize=16, fontweight='bold')
    
    renkler = {
        'hic': '#e74c3c', 'surekli': '#2ecc71', 'erken_patlama': '#f39c12',
        'periyodik': '#3498db', 'optimal': '#9b59b6'
    }
    
    for strateji, veri in sonuclar_sozlugu.items():
        renk = renkler.get(strateji, 'gray')
        etiket = STRATEJI_ETIKETLERI.get(strateji, strateji)
        
        eksenler[0, 0].plot(veri['nesil'], veri['canli_hucre'] / 1000, label=etiket, color=renk, linewidth=2.5, alpha=0.9)
        eksenler[0, 1].plot(veri['nesil'], veri['ort_telomer'], label=etiket, color=renk, linewidth=2.5, alpha=0.9)
        eksenler[1, 0].plot(veri['nesil'], veri['kanser_hucre'] / 1000, label=etiket, color=renk, linewidth=2.5, alpha=0.9)
        eksenler[1, 1].plot(veri['nesil'], veri['ort_mutasyon'], label=etiket, color=renk, linewidth=2.5, alpha=0.9)
        eksenler[0, 2].plot(veri['nesil'], veri['senesans_hucre'] / 1000, label=etiket, color=renk, linewidth=2.5, alpha=0.9)
        eksenler[1, 2].plot(veri['nesil'], veri['ort_bolunme'], label=etiket, color=renk, linewidth=2.5, alpha=0.9)
    
    eksenler[0, 0].set_title('Canlı Hücre Sayısı', fontsize=12, fontweight='bold')
    eksenler[0, 0].set_xlabel('Nesil')
    eksenler[0, 0].set_ylabel('Hücre Sayısı (×1000)')
    eksenler[0, 0].legend(loc='best')
    eksenler[0, 0].grid(alpha=0.3)
    
    eksenler[0, 1].set_title('Ortalama Telomer Uzunluğu (Normal Hücreler)', fontsize=12, fontweight='bold')
    eksenler[0, 1].set_xlabel('Nesil')
    eksenler[0, 1].set_ylabel('Telomer Uzunluğu (bp)')
    eksenler[0, 1].axhline(T_KRITIK, color='red', linestyle='--', alpha=0.5, label='Kritik Esik')
    eksenler[0, 1].axvline(HAYFLICK_LIMITI, color='orange', linestyle=':', alpha=0.5, label='Hayflick Limit')
    eksenler[0, 1].legend(loc='best')
    eksenler[0, 1].grid(alpha=0.3)
    
    eksenler[1, 0].set_title('Toplam Kanserli Hücre Sayısı', fontsize=12, fontweight='bold')
    eksenler[1, 0].set_xlabel('Nesil')
    eksenler[1, 0].set_ylabel('Kanser Hücresi (×1000)')
    eksenler[1, 0].legend(loc='best')
    eksenler[1, 0].grid(alpha=0.3)
    
    eksenler[1, 1].set_title('Ortalama Mutasyon Sayısı', fontsize=12, fontweight='bold')
    eksenler[1, 1].set_xlabel('Nesil')
    eksenler[1, 1].set_ylabel('Mutasyon Sayısı')
    eksenler[1, 1].axhline(KANSER_ESIGI, color='red', linestyle='--', alpha=0.5, label='Kanser Esigi')
    eksenler[1, 1].legend(loc='best')
    eksenler[1, 1].grid(alpha=0.3)
    
    eksenler[0, 2].set_title('O Nesilde Senesansa Giren Hücreler', fontsize=12, fontweight='bold')
    eksenler[0, 2].set_xlabel('Nesil')
    eksenler[0, 2].set_ylabel('Senesans Hücre (×1000)')
    eksenler[0, 2].legend(loc='best')
    eksenler[0, 2].grid(alpha=0.3)
    
    eksenler[1, 2].set_title('Ortalama Bölünme Sayısı', fontsize=12, fontweight='bold')
    eksenler[1, 2].set_xlabel('Nesil')
    eksenler[1, 2].set_ylabel('Bölünme Sayısı')
    eksenler[1, 2].axhline(HAYFLICK_LIMITI, color='red', linestyle='--', alpha=0.5, label='Hayflick Limiti')
    eksenler[1, 2].legend(loc='best')
    eksenler[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    return sekil

# ===================== STREAMLIT ARAYÜZÜ =====================
def ana():
    st.set_page_config(page_title="Telomeraz Optimizasyonu", layout="wide")
    
    st.title("🧬 Ölümsüzlük Bedeli: Telomeraz Optimizasyonu")
    st.markdown("### TÜBİTAK 2204-A Projesi")
    st.markdown("---")
    
    with st.sidebar:
        st.header("📊 Proje Bilgileri v7.3")
        st.markdown("""
        **Güncel Model Parametreleri:**
        
        🧬 **Telomer:**
        - Başlangıç: 10,000 bp (±1,000 bp)
        - Kritik Eşik: 4,500 bp (±500 bp)
        - Kayıp/bölünme: ~75 bp (±25 bp)
        
        🔬 **Hücre Dinamikleri:**
        - Hayflick Limiti: 50 bölünme (±10)
        - Normal bölünme: 2x
        - Kanser bölünme: 3x
        
        🧪 **Kanser Modeli:**
        - **Kanser Eşiği: 6 mutasyon (±1)** ✅ Literatür
        - **Mutasyon Oranı: 0.2% (düşük!)**
        - Kısa telomer: 2x artış
        - **⚠️ Telomeraz: 1.5x mutasyon artışı**
        - Mutasyon olur ama kanser zor gelişir
        - **Kanser hücreleri senesense GİRMEZ**
        
        💊 **Telomeraz:**
        - Ekleme: 60 bp
        - Etkinlik: %80
        - **Mutasyon riski: CİDDİ ARTIŞ**
        
        📊 **Senesens Kuralları:**
        - **Telomeraz YOK**: Telomer<4500 VEYA Hayflick≥50
        - **Diğer stratejiler**: Sadece Telomer<4500
        - **Kanser**: Senesense GİRMEZ
        
        📈 **Simülasyon:**
        - Başlangıç: 5,000 hücre
        - Maksimum: 50,000 hücre
        - Max nesil: 80
        """)
        
        st.markdown("---")
        simul_bas = st.button("🚀 Simülasyonu Başlat", type="primary")
        optim_bas = st.button("🎯 Optimizasyon Yap")
    
    sekme1, sekme2, sekme3, sekme4 = st.tabs([
        "📈 Strateji Karşılaştırması",
        "🎯 Optimizasyon Sonuçları",
        "🔬 Model Detayları",
        "📚 Bilimsel Açıklama"
    ])
    
    with sekme1:
        if simul_bas:
            st.header("Telomeraz Stratejilerinin Karşılaştırılması")
            
            stratejiler = ['hic', 'erken_patlama', 'periyodik', 'surekli', 'optimal']
            sonuclar = {}
            
            ilerleme = st.progress(0)
            durum = st.empty()
            
            for i, strateji in enumerate(stratejiler):
                durum.text(f"'{STRATEJI_ETIKETLERI[strateji]}' stratejisi çalışıyor...")
                
                if strateji == 'periyodik':
                    veri = populasyon_simule_et_varyasyonlu(strateji, period=20, duration=5, dose=0.8)
                else:
                    veri = populasyon_simule_et_varyasyonlu(strateji)
                
                sonuclar[strateji] = veri
                ilerleme.progress((i + 1) / len(stratejiler))
            
            durum.text("Tamamlandı! ✅")
            
            sekil = strateji_karsilastirmasi_goster(sonuclar)
            st.pyplot(sekil)
            
            st.subheader("📊 Özet İstatistikler")
            
            ozet = []
            for strateji, veri in sonuclar.items():
                if not veri.empty:
                    son_pop = veri['canli_hucre'].iloc[-1]
                    son_kanser = veri['kanser_hucre'].iloc[-1]
                    kanser_orani = (son_kanser / son_pop * 100) if son_pop > 0 else 0
                    
                    ozet.append({
                        'Strateji': STRATEJI_ETIKETLERI.get(strateji, strateji),
                        'Yaşam Süresi (nesil)': len(veri),
                        'Son Popülasyon': f"{son_pop/1000:.1f}K",
                        'Son Kanser Sayısı': f"{son_kanser/1000:.1f}K",
                        'Kanser Oranı (%)': f"{kanser_orani:.1f}%",
                        'Ortalama Kanser': f"{veri['kanser_hucre'].mean()/1000:.1f}K",
                        'Son Telomer (bp)': f"{veri['ort_telomer'].iloc[-1]:.0f}"
                    })
            
            ozet_veri = pd.DataFrame(ozet)
            st.dataframe(ozet_veri, use_container_width=True)
    
    with sekme2:
        if optim_bas:
            st.header("🎯 Optimal Telomeraz Zamanlaması")
            
            with st.spinner("Optimizasyon yapılıyor... (1-2 dakika sürebilir)"):
                opt_sonuclar = telomeraz_zamanlamasi_optimize()
            
            en_iyi = opt_sonuclar.nlargest(1, 'skor').iloc[0]
            
            st.success("✅ Optimizasyon Tamamlandı!")
            
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("📅 Optimal Periyot", f"{int(en_iyi['periyot'])} nesil")
            s2.metric("⏱️ Optimal Süre", f"{int(en_iyi['sure'])} nesil")
            s3.metric("💊 Optimal Doz", f"%{int(en_iyi['doz']*100)}")
            s4.metric("⭐ Skor", f"{en_iyi['skor']:.3f}")
            
            st.info(f"""
            ### 🎯 Önerilen Strateji:
            Her **{int(en_iyi['periyot'])}** nesilde bir, **{int(en_iyi['sure'])}** nesil boyunca
            **%{int(en_iyi['doz']*100)}** telomeraz aktivitesi.
            """)
            
            st.subheader("🏆 En İyi 10 Kombinasyon")
            ilk10 = opt_sonuclar.nlargest(10, 'skor')
            st.dataframe(ilk10, use_container_width=True)
    
    with sekme3:
        st.header("🔬 Model Detayları ve Parametreler")
        st.markdown("""
        ### Senesens Kuralları
        
        **Telomeraz YOK (Normal Hücreler):**
        - Telomer < 4500 bp → Senesens
        - Bölünme ≥ 50 (Hayflick) → Senesens
        
        **Telomeraz VAR:**
        - Sadece Telomer < 4500 bp → Senesens
        - Hayflick limiti geçerli DEĞİL (telomeraz uzatabilir)
        
        **Kanser Hücreleri:**
        - Senesense GİRMEZ
        - Sınırsız bölünme kapasitesi
        """)
    
    with sekme4:
        st.header("📚 Bilimsel Arka Plan")
        st.markdown("""
        
        ### Kaynakça

        Knudson, A. G. (1971). Mutation and cancer: Statistical study of retinoblastoma. Proceedings of the National Academy of Sciences, 68(4), 820-823.

        Shay, J. W., & Wright, W. E. (2019). Telomeres and telomerase: Three decades of progress. Nature Reviews Genetics, 20(5), 299-309.

        Hayflick, L., & Moorhead, P. S. (1961). The serial cultivation of human diploid cell strains. Experimental Cell Research, 25(3), 585-621.

        Harley, C. B., Futcher, A. B., & Greider, C. W. (1990). Telomeres shorten during ageing of human fibroblasts. Nature, 345(6274), 458-460.

        Kim, N. W., Piatyszek, M. A., Prowse, K. R., Harley, C. B., West, M. D., Ho, P. L., . . . Shay, J. W. (1994). Specific association of human telomerase activity with immortal cells and cancer. Science, 266(5193), 2011-2015.

        Bodnar, A. G., Ouellette, M., Frolkis, M., Holt, S. E., Chiu, C. P., Morin, G. B., . . . Wright, W. E. (1998). Extension of life-span by introduction of telomerase into normal human cells. Science, 279(5349), 349-352.

        Shay, J. W., & Wright, W. E. (2019). Telomeres and telomerase: Three decades of progress. Nature Reviews Genetics, 20(5), 299-309.

        ​Hanahan, D., & Weinberg, R. A. (2011). Hallmarks of cancer: the next generation. Cell, 144(5), 646-674.

        ​D'Orsogna, M. R., Breen, B., & Chou, T. (2017). Agent-based modeling of cancer cell populations. Physica A: Statistical Mechanics and its Applications, 486, 735-745.

        Artandi, S. E., & DePinho, R. A. (2010). Telomeres and telomerase in cancer. Cold Spring Harbor Perspectives in Biology, 2(12), a008138.

        ​Deakin, J., & Al-Tameemi, Y. (2015). The role of cellular heterogeneity in cancer. Frontiers in Oncology, 5, 22.

        """)

if __name__ == "__main__":
    ana()
