# -*- coding: utf-8 -*-
"""
Akademik Link Anket Analizi - Türkçe Açıklamalar ile
Turkish Political Survey Analysis with Turkish Explanations
Compatible version with improved chart explanations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import re
from collections import defaultdict

# Turkish font support for matplotlib
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma', 'sans-serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

print("=== AKADEMİK LİNK ANKET ANALİZİ - TÜRKÇE VERSİYON ===")
print("Turkish Political Survey Analysis - Turkish Version")
print("=" * 60)

# Load and prepare data
try:
    # Try different possible file paths
    possible_paths = [
        '/Users/ysk/Desktop/Projects/AkademiklinkAnket/Akademiklink gündem 2 (Yanıtlar).xlsx',
        '/Users/ysk/Desktop/Projects/Akedemik_Link_Anket/akademiklink_comparative_analysis.xlsx',
        '/Users/ysk/Desktop/Projects/Akedemik_Link_Anket/akademiklink_final_analysis.xlsx'
    ]
    
    df = None
    for path in possible_paths:
        try:
            df = pd.read_excel(path)
            print(f"✅ Veri dosyası başarıyla yüklendi: {path}")
            print(f"📊 Veri seti boyutu: {df.shape}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        raise FileNotFoundError("Hiçbir Excel dosyası bulunamadı")
        
except Exception as e:
    print(f"❌ Dosya yükleme hatası: {e}")
    print("Örnek veri seti oluşturuluyor...")
    
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 500
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'age': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n_samples),
        'gender': np.random.choice(['Kadın', 'Erkek'], n_samples),
        'city': np.random.choice(['İstanbul', 'Ankara', 'İzmir', 'Diğer'], n_samples),
        'income': np.random.choice(['≤5,000 TL', '5,001-15,000 TL', '15,001-30,000 TL', '30,001+ TL'], n_samples),
        'education': np.random.choice(['İlkokul', 'Lise', 'Üniversite', 'Yüksek Lisans'], n_samples),
        'last_vote': np.random.choice(['Erdoğan', 'Kılıçdaroğlu', 'Diğer', 'Oy vermedi'], n_samples),
        'current_preference': np.random.choice(['Erdoğan', 'İmamoğlu', 'Yavaş', 'Kararsız', 'Diğer'], n_samples),
        'main_problem': np.random.choice(['Ekonomi', 'İşsizlik', 'Enflasyon', 'Eğitim', 'Sağlık', 'Adalet'], n_samples),
        'social_media_usage': np.random.choice(['Her gün', 'Haftada birkaç kez', 'Nadiren'], n_samples),
    })
    
    # Add Likert scale questions (1-7)
    likert_questions = ['istanbul_election_cancel', 'friendship_akp', 'friendship_chp', 
                       'early_election', 'end_presidential_system', 'akp_chp_coalition']
    for col in likert_questions:
        df[col] = np.random.randint(1, 8, n_samples)

print(f"\n📋 Analiz edilen anket sayısı: {len(df)}")
print(f"📅 Veri toplama dönemi: {df['timestamp'].min()} - {df['timestamp'].max()}" if 'timestamp' in df.columns else "")

# PART 1: DEMOGRAFİK ANALİZ - DEMOGRAPHIC ANALYSIS
print("\n" + "="*60)
print("1. DEMOGRAFİK ANALİZ VE KATILIMCI PROFİLİ")
print("   DEMOGRAPHIC ANALYSIS AND PARTICIPANT PROFILE")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Anket Katılımcılarının Demografik Profili\nDemographic Profile of Survey Participants', 
             fontsize=16, fontweight='bold', y=0.98)

# 1.1 Yaş Dağılımı - Age Distribution
if 'age' in df.columns:
    age_counts = df['age'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(age_counts)))
    wedges, texts, autotexts = axes[0,0].pie(age_counts.values, labels=age_counts.index, 
                                            autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0,0].set_title('Yaş Dağılımı\nAge Distribution', fontweight='bold')
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

# 1.2 Cinsiyet Dağılımı - Gender Distribution
if 'gender' in df.columns:
    gender_counts = df['gender'].value_counts()
    colors = ['#FF6B9D', '#4ECDC4']
    axes[0,1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
                  startangle=90, colors=colors)
    axes[0,1].set_title('Cinsiyet Dağılımı\nGender Distribution', fontweight='bold')

# 1.3 Eğitim Seviyesi - Education Level
if 'education' in df.columns:
    edu_counts = df['education'].value_counts()
    bars = axes[0,2].barh(edu_counts.index, edu_counts.values, color='skyblue')
    axes[0,2].set_title('Eğitim Seviyesi Dağılımı\nEducation Level Distribution', fontweight='bold')
    axes[0,2].set_xlabel('Katılımcı Sayısı (Number of Participants)')
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, edu_counts.values)):
        axes[0,2].text(count + max(edu_counts.values)*0.01, i, str(count), 
                      va='center', fontsize=9)

# 1.4 Gelir Dağılımı - Income Distribution
if 'income' in df.columns:
    income_counts = df['income'].value_counts()
    axes[1,0].bar(range(len(income_counts)), income_counts.values, color='lightgreen')
    axes[1,0].set_xticks(range(len(income_counts)))
    axes[1,0].set_xticklabels(income_counts.index, rotation=45, ha='right')
    axes[1,0].set_title('Aylık Gelir Dağılımı\nMonthly Income Distribution', fontweight='bold')
    axes[1,0].set_ylabel('Katılımcı Sayısı (Number of Participants)')

# 1.5 Şehir Dağılımı - City Distribution
if 'city' in df.columns:
    city_counts = df['city'].value_counts()
    axes[1,1].bar(city_counts.index, city_counts.values, color='orange')
    axes[1,1].set_title('Şehir Dağılımı\nCity Distribution', fontweight='bold')
    axes[1,1].set_ylabel('Katılımcı Sayısı (Number of Participants)')
    axes[1,1].tick_params(axis='x', rotation=45)

# 1.6 Sosyal Medya Kullanımı - Social Media Usage
if 'social_media_usage' in df.columns:
    social_counts = df['social_media_usage'].value_counts()
    axes[1,2].pie(social_counts.values, labels=social_counts.index, autopct='%1.1f%%')
    axes[1,2].set_title('Sosyal Medya Kullanım Sıklığı\nSocial Media Usage Frequency', fontweight='bold')

plt.tight_layout()
plt.show()

# PART 2: SİYASİ TERCİHLER ANALİZİ - POLITICAL PREFERENCES ANALYSIS
print("\n" + "="*60)
print("2. SİYASİ TERCİHLER VE OYLAR ANALİZİ")
print("   POLITICAL PREFERENCES AND VOTING ANALYSIS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Siyasi Tercihler ve Oy Verme Davranışları\nPolitical Preferences and Voting Behavior', 
             fontsize=16, fontweight='bold')

# 2.1 Son Seçimdeki Oy - Last Vote
if 'last_vote' in df.columns:
    last_vote_counts = df['last_vote'].value_counts()
    colors = plt.cm.tab10(np.linspace(0, 1, len(last_vote_counts)))
    bars = axes[0,0].bar(range(len(last_vote_counts)), last_vote_counts.values, color=colors)
    axes[0,0].set_xticks(range(len(last_vote_counts)))
    axes[0,0].set_xticklabels(last_vote_counts.index, rotation=45, ha='right')
    axes[0,0].set_title('2023 Cumhurbaşkanlığı Seçiminde Verilen Oy\n2023 Presidential Election Vote', 
                       fontweight='bold')
    axes[0,0].set_ylabel('Oy Sayısı (Number of Votes)')
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, last_vote_counts.values)):
        axes[0,0].text(i, count + max(last_vote_counts.values)*0.01, str(count), 
                      ha='center', va='bottom', fontsize=9)

# 2.2 Güncel Tercih - Current Preference
if 'current_preference' in df.columns:
    current_counts = df['current_preference'].value_counts()
    axes[0,1].pie(current_counts.values, labels=current_counts.index, autopct='%1.1f%%')
    axes[0,1].set_title('Gelecek Seçimde Tercih Edilen Aday\nPreferred Candidate for Next Election', 
                       fontweight='bold')

# 2.3 Tercih Değişimi Analizi - Preference Change Analysis
if 'last_vote' in df.columns and 'current_preference' in df.columns:
    # Create crosstab for vote switching
    vote_matrix = pd.crosstab(df['last_vote'], df['current_preference'], margins=True)
    
    # Remove margins for heatmap
    vote_matrix_clean = vote_matrix.iloc[:-1, :-1]
    
    sns.heatmap(vote_matrix_clean, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
    axes[1,0].set_title('Oy Değişimi Matrisi\n(Satır: Önceki Oy, Sütun: Şimdiki Tercih)\nVote Change Matrix\n(Row: Previous Vote, Column: Current Preference)', 
                       fontweight='bold')
    axes[1,0].set_xlabel('Şimdiki Tercih (Current Preference)')
    axes[1,0].set_ylabel('Önceki Oy (Previous Vote)')

# 2.4 Ana Sorunlar - Main Problems
if 'main_problem' in df.columns:
    problem_counts = df['main_problem'].value_counts()
    bars = axes[1,1].barh(problem_counts.index, problem_counts.values, color='lightcoral')
    axes[1,1].set_title('Türkiye\'nin En Önemli Sorunu\nTurkey\'s Most Important Problem', 
                       fontweight='bold')
    axes[1,1].set_xlabel('Yanıt Sayısı (Number of Responses)')
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, problem_counts.values)):
        axes[1,1].text(count + max(problem_counts.values)*0.01, i, str(count), 
                      va='center', fontsize=9)

plt.tight_layout()
plt.show()

# PART 3: SİYASİ TUTUMLAR - POLITICAL ATTITUDES
print("\n" + "="*60)
print("3. SİYASİ TUTUMLAR VE GÖRÜŞLER ANALİZİ")
print("   POLITICAL ATTITUDES AND OPINIONS ANALYSIS")
print("="*60)

# Check for Likert scale questions
likert_cols = [col for col in df.columns if col in ['istanbul_election_cancel', 'friendship_akp', 
               'friendship_chp', 'early_election', 'end_presidential_system', 'akp_chp_coalition']]

if likert_cols:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Siyasi Tutumlar ve Görüşler (1=Kesinlikle Katılmıyorum, 7=Kesinlikle Katılıyorum)\nPolitical Attitudes and Opinions (1=Strongly Disagree, 7=Strongly Agree)', 
                 fontsize=14, fontweight='bold')
    
    # Turkish labels for questions
    question_labels = {
        'istanbul_election_cancel': 'İstanbul Seçiminin\nİptal Edilmesi',
        'friendship_akp': 'AKP\'li Arkadaşlık\nToleransı',
        'friendship_chp': 'CHP\'li Arkadaşlık\nToleransı',
        'early_election': 'Erken Seçim\nTalebi',
        'end_presidential_system': 'Başkanlık Sisteminin\nSona Ermesi',
        'akp_chp_coalition': 'AKP-CHP\nKoalisyonu'
    }
    
    for i, col in enumerate(likert_cols[:6]):  # Maximum 6 questions
        row = i // 3
        col_idx = i % 3
        
        if col in df.columns:
            # Calculate mean scores
            mean_score = df[col].mean()
            
            # Create histogram
            axes[row, col_idx].hist(df[col].dropna(), bins=7, range=(0.5, 7.5), 
                                  color='lightblue', alpha=0.7, edgecolor='black')
            axes[row, col_idx].axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                                     label=f'Ortalama: {mean_score:.1f}')
            axes[row, col_idx].set_title(question_labels.get(col, col), fontweight='bold')
            axes[row, col_idx].set_xlabel('Katılım Derecesi (Agreement Level)')
            axes[row, col_idx].set_ylabel('Katılımcı Sayısı (Number of Participants)')
            axes[row, col_idx].legend()
            axes[row, col_idx].grid(True, alpha=0.3)
            
            # Add scale labels
            axes[row, col_idx].set_xticks(range(1, 8))
    
    # Remove empty subplots
    for i in range(len(likert_cols), 6):
        row = i // 3
        col_idx = i % 3
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# PART 4: DEMOGRAFİK ÇAPRAZ ANALİZ - DEMOGRAPHIC CROSS-ANALYSIS
print("\n" + "="*60)
print("4. DEMOGRAFİK ÇAPRAZ ANALİZ")
print("   DEMOGRAPHIC CROSS-ANALYSIS")
print("="*60)

if 'current_preference' in df.columns and 'age' in df.columns:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Demografik Gruplara Göre Siyasi Tercihler\nPolitical Preferences by Demographic Groups', 
                 fontsize=16, fontweight='bold')
    
    # 4.1 Yaşa Göre Tercihler - Preferences by Age
    pref_age = pd.crosstab(df['age'], df['current_preference'], normalize='index') * 100
    sns.heatmap(pref_age, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=axes[0,0])
    axes[0,0].set_title('Yaş Grubuna Göre Siyasi Tercihler (%)\nPolitical Preferences by Age Group (%)', 
                       fontweight='bold')
    axes[0,0].set_xlabel('Siyasi Tercih (Political Preference)')
    axes[0,0].set_ylabel('Yaş Grubu (Age Group)')
    
    # 4.2 Eğitime Göre Tercihler - Preferences by Education
    if 'education' in df.columns:
        pref_edu = pd.crosstab(df['education'], df['current_preference'], normalize='index') * 100
        sns.heatmap(pref_edu, annot=True, fmt='.1f', cmap='viridis', ax=axes[0,1])
        axes[0,1].set_title('Eğitim Seviyesine Göre Siyasi Tercihler (%)\nPolitical Preferences by Education Level (%)', 
                           fontweight='bold')
        axes[0,1].set_xlabel('Siyasi Tercih (Political Preference)')
        axes[0,1].set_ylabel('Eğitim Seviyesi (Education Level)')
    
    # 4.3 Gelire Göre Tercihler - Preferences by Income
    if 'income' in df.columns:
        pref_income = pd.crosstab(df['income'], df['current_preference'], normalize='index') * 100
        sns.heatmap(pref_income, annot=True, fmt='.1f', cmap='plasma', ax=axes[1,0])
        axes[1,0].set_title('Gelir Seviyesine Göre Siyasi Tercihler (%)\nPolitical Preferences by Income Level (%)', 
                           fontweight='bold')
        axes[1,0].set_xlabel('Siyasi Tercih (Political Preference)')
        axes[1,0].set_ylabel('Gelir Seviyesi (Income Level)')
    
    # 4.4 Cinsiyete Göre Tercihler - Preferences by Gender
    if 'gender' in df.columns:
        pref_gender = pd.crosstab(df['gender'], df['current_preference'], normalize='index') * 100
        pref_gender.plot(kind='bar', ax=axes[1,1], colormap='Set2')
        axes[1,1].set_title('Cinsiyete Göre Siyasi Tercihler (%)\nPolitical Preferences by Gender (%)', 
                           fontweight='bold')
        axes[1,1].set_xlabel('Cinsiyet (Gender)')
        axes[1,1].set_ylabel('Yüzde (Percentage)')
        axes[1,1].legend(title='Siyasi Tercih', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1,1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.show()

# PART 5: ÖZET İSTATİSTİKLER - SUMMARY STATISTICS
print("\n" + "="*60)
print("5. ÖZET İSTATİSTİKLER VE SONUÇLAR")
print("   SUMMARY STATISTICS AND CONCLUSIONS")
print("="*60)

# Create summary statistics
print("📊 ANKET KATILIMCILARI ÖZETİ / SURVEY PARTICIPANTS SUMMARY")
print("-" * 50)

if 'age' in df.columns:
    print(f"👥 Yaş Dağılımı / Age Distribution:")
    age_summary = df['age'].value_counts()
    for age_group, count in age_summary.items():
        percentage = (count / len(df)) * 100
        print(f"   • {age_group}: {count} kişi ({percentage:.1f}%)")

if 'education' in df.columns:
    print(f"\n🎓 Eğitim Seviyesi / Education Level:")
    edu_summary = df['education'].value_counts()
    for edu_level, count in edu_summary.items():
        percentage = (count / len(df)) * 100
        print(f"   • {edu_level}: {count} kişi ({percentage:.1f}%)")

if 'current_preference' in df.columns:
    print(f"\n🗳️  Güncel Siyasi Tercihler / Current Political Preferences:")
    pref_summary = df['current_preference'].value_counts()
    for preference, count in pref_summary.items():
        percentage = (count / len(df)) * 100
        print(f"   • {preference}: {count} kişi ({percentage:.1f}%)")

if 'main_problem' in df.columns:
    print(f"\n⚠️  Ana Sorunlar / Main Problems:")
    problem_summary = df['main_problem'].value_counts()
    for problem, count in problem_summary.items():
        percentage = (count / len(df)) * 100
        print(f"   • {problem}: {count} yanıt ({percentage:.1f}%)")

# Likert scale summaries
if likert_cols:
    print(f"\n📈 SİYASİ TUTUM ORTALAMALARİ / POLITICAL ATTITUDE AVERAGES")
    print("(1=Kesinlikle Katılmıyorum, 7=Kesinlikle Katılıyorum)")
    print("(1=Strongly Disagree, 7=Strongly Agree)")
    print("-" * 50)
    
    attitude_meanings = {
        'istanbul_election_cancel': 'İstanbul seçiminin iptal edilmesi doğruydu',
        'friendship_akp': 'AKP\'li biriyle arkadaş olabilirim',
        'friendship_chp': 'CHP\'li biriyle arkadaş olabilirim',
        'early_election': 'Erken seçim yapılmalı',
        'end_presidential_system': 'Başkanlık sistemi sona ermeli',
        'akp_chp_coalition': 'AKP-CHP koalisyonu mümkün'
    }
    
    for col in likert_cols:
        if col in df.columns:
            mean_score = df[col].mean()
            std_score = df[col].std()
            meaning = attitude_meanings.get(col, col)
            
            # Determine agreement level
            if mean_score >= 5.5:
                level = "Yüksek Katılım"
            elif mean_score >= 4.5:
                level = "Orta Katılım"
            elif mean_score >= 3.5:
                level = "Nötr"
            elif mean_score >= 2.5:
                level = "Düşük Katılım"
            else:
                level = "Çok Düşük Katılım"
            
            print(f"   • {meaning}")
            print(f"     Ortalama: {mean_score:.1f}/7 ({level})")
            print(f"     Standart Sapma: {std_score:.1f}")
            print()

# Final summary visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Anket Sonuçları Genel Özeti\nSurvey Results Summary', fontsize=16, fontweight='bold')

# Summary chart 1: Top preferences
if 'current_preference' in df.columns:
    top_prefs = df['current_preference'].value_counts().head(5)
    bars = ax1.bar(range(len(top_prefs)), top_prefs.values, color='lightblue')
    ax1.set_xticks(range(len(top_prefs)))
    ax1.set_xticklabels(top_prefs.index, rotation=45, ha='right')
    ax1.set_title('En Popüler 5 Siyasi Tercih\nTop 5 Political Preferences', fontweight='bold')
    ax1.set_ylabel('Oy Sayısı (Number of Votes)')
    
    for i, (bar, count) in enumerate(zip(bars, top_prefs.values)):
        ax1.text(i, count + max(top_prefs.values)*0.01, str(count), 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Summary chart 2: Age distribution
if 'age' in df.columns:
    age_dist = df['age'].value_counts()
    ax2.pie(age_dist.values, labels=age_dist.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Yaş Dağılımı Özeti\nAge Distribution Summary', fontweight='bold')

# Summary chart 3: Main problems
if 'main_problem' in df.columns:
    top_problems = df['main_problem'].value_counts().head(5)
    bars = ax3.barh(range(len(top_problems)), top_problems.values, color='lightcoral')
    ax3.set_yticks(range(len(top_problems)))
    ax3.set_yticklabels(top_problems.index)
    ax3.set_title('En Önemli 5 Sorun\nTop 5 Important Problems', fontweight='bold')
    ax3.set_xlabel('Yanıt Sayısı (Number of Responses)')
    
    for i, (bar, count) in enumerate(zip(bars, top_problems.values)):
        ax3.text(count + max(top_problems.values)*0.01, i, str(count), 
                va='center', fontsize=10, fontweight='bold')

# Summary chart 4: Likert scale averages
if likert_cols:
    likert_means = [df[col].mean() for col in likert_cols if col in df.columns]
    likert_labels = [question_labels.get(col, col) for col in likert_cols if col in df.columns]
    
    colors = ['red' if mean < 3.5 else 'orange' if mean < 4.5 else 'lightgreen' 
              for mean in likert_means]
    
    bars = ax4.bar(range(len(likert_means)), likert_means, color=colors)
    ax4.set_xticks(range(len(likert_means)))
    ax4.set_xticklabels(likert_labels, rotation=45, ha='right')
    ax4.set_title('Siyasi Tutum Ortalamaları\nPolitical Attitude Averages', fontweight='bold')
    ax4.set_ylabel('Ortalama Puan (Average Score)')
    ax4.set_ylim(1, 7)
    ax4.axhline(y=4, color='black', linestyle='--', alpha=0.5, label='Nötr Çizgi')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    for i, (bar, mean) in enumerate(zip(bars, likert_means)):
        ax4.text(i, mean + 0.1, f'{mean:.1f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("✅ ANALİZ TAMAMLANDI / ANALYSIS COMPLETED")
print("="*60)
print(f"📋 Toplam analiz edilen anket: {len(df)}")
print(f"📊 Oluşturulan grafik sayısı: 15+")
print(f"🎯 Analiz edilen kategori sayısı: {len([col for col in ['age', 'gender', 'education', 'income', 'current_preference', 'main_problem'] if col in df.columns])}")
print("\n💡 ÖNEMLI NOTLAR / IMPORTANT NOTES:")
print("• Tüm grafikler Türkçe açıklamalar ile hazırlanmıştır")
print("• All charts are prepared with Turkish explanations")
print("• Veriler yüzdelik dilimler halinde sunulmuştur")
print("• Data is presented in percentage segments")
print("• Çapraz analizler demografik gruplar arası karşılaştırma sağlar")
print("• Cross-analyses provide comparisons between demographic groups")

print(f"\n🎉 Analiz başarıyla tamamlandı! / Analysis completed successfully!")
