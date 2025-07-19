# %%
# Required packages are installed in the virtual environment

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz
import re
import nltk
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import defaultdict
import json
try:
    from wordcloud import WordCloud
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    from textblob import TextBlob
except ImportError as e:
    print(f"Some optional libraries not installed: {e}")
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %%
try:
    df = pd.read_excel('/Users/ysk/Desktop/Projects/AkademiklinkAnket/Akademiklink gündem 2 (Yanıtlar).xlsx')
    print("Excel file loaded successfully!")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("Excel file not found. Please make sure the file exists in the current directory.")
    print("Looking for files with .xlsx or .xls extension...")
except (PermissionError, pd.errors.EmptyDataError) as e:
    print(f"Error loading Excel file. Please check the file path and format: {e}")

# %%
# Clean column names
df.columns = [
    'timestamp', 'age', 'gender', 'city', 'income', 'education', 'marital_status',
    'last_vote', 'current_preference', 'politics_important_marriage', 'attention_check',
    'social_media_usage', 'blocked_friends', 'unfollowed_influencers', 'main_problem',
    'istanbul_election_cancel', 'friendship_akp', 'friendship_chp', 'imamoglu_prison',
    'ozdag_prison', 'demirtas_prison', 'vote_opposite_party', 'support_lawbreaking',
    'early_election', 'end_presidential_system', 'akp_chp_coalition', 'boycotts_effective',
    'new_constitution', 'solution_process', 'akp_description', 'chp_description',
    'imamoglu_statement_read'
]

print("=== COMPREHENSIVE TURKISH POLITICAL SURVEY ANALYSIS ===")
print(f"Dataset shape: {df.shape}")
print(f"Survey responses: {len(df)}")

# %%
# 1. DATA CLEANING AND PREPROCESSING
print("\n1. DATA CLEANING AND PREPROCESSING")
print("=" * 50)

# Handle missing values and clean data
df_clean = df.copy()

# Clean age groups
age_mapping = {
    '18 altı': 'Under 18',
    '18-24': '18-24',
    '25-34': '25-34', 
    '35-44 yaş': '35-44',
    '45-54': '45-54',
    '55-64': '55-64',
    '65+': '65+'
}
df_clean['age'] = df_clean['age'].map(age_mapping)

# Clean income groups
income_mapping = {
    '5.000 TL ve altı': '≤5,000 TL',
    '20.001 – 30.000 TL': '20,001-30,000 TL',
    '30.001 TL ve 50.000 TL': '30,001-50,000 TL',
    '50.001 TL ve 80.000 TL': '50,001-80,000 TL',
    '80.001 TL ve üzeri': '≥80,001 TL'
}
df_clean['income'] = df_clean['income'].map(income_mapping)

# Clean education levels
education_mapping = {
    'İlkokul ': 'Primary School',
    'ortaokul ': 'Middle School ',
    'Lise (Hala öğrenci)': 'High School (Student)',
    'Lise': 'High School',
    'Üniversite (Hala öğrenci)': 'University (Student)',
    'Üniversite': 'University',
    'Yüksek Lisans/ Doktora (Hala öğrenci)': 'Graduate (Student)',
    'Yüksek Lisans/ Doktora': 'Graduate'
}
df_clean['education'] = df_clean['education'].map(education_mapping)

# Clean voting preferences
vote_mapping = {
    'Recep Tayyip Erdoğan': 'Erdoğan (AKP)',
    'Kemal Kılıçdaroğlu': 'Kılıçdaroğlu (CHP)',
    'Sinan Oğan': 'Oğan (ATA)',
    'Muharrem İnce': 'İnce (Memleket)',
    'Oy kullanmadım': 'Did not vote'
}
df_clean['last_vote'] = df_clean['last_vote'].map(vote_mapping)

# %%
# Clean current preferences
current_mapping = {
    # Recep Tayyip Erdoğan Variations
    'Erdoğan': 'Erdoğan',
    'Recep Tayyip Erdoğan': 'Erdoğan',
    'RTE': 'Erdoğan',
    'Reis': 'Erdoğan',
    'Tayyip Erdoğan': 'Erdoğan',
    'Erdoğan.': 'Erdoğan',
    'Recep tayyip': 'Erdoğan',
    'Erdoğan veya onun destekleyeceği kişi': 'Erdoğan',
    'Erdoğan veya onun cizgisinde birine': 'Erdoğan',
    'Recep Tayyip Erdoğan/ Hakan Fidan': 'Erdoğan or Fidan',

    # Ekrem İmamoğlu Variations
    'Ekrem İmamoğlu': 'İmamoğlu',
    'Ekrem Imamoglu': 'İmamoğlu',
    'Ekrem imamoğlu': 'İmamoğlu',
    'İmamoğlu': 'İmamoğlu',
    'imamoğlu': 'İmamoğlu',
    'Eko': 'İmamoğlu',
    'Ekrem': 'İmamoğlu',
    'Ekrem İmamoğlu (Recep Tayyip karşısında kim varsa)': 'İmamoğlu',
    'Ekrem İmamson:)': 'İmamoğlu',
    'Ekrem İMAMOĞLU': 'İmamoğlu',
    'imam oğlu': 'İmamoğlu',
    'Ekrem başkan': 'İmamoğlu',
    'Ekrem imaro': 'İmamoğlu',
    'imamov': 'İmamoğlu',

    # Mansur Yavaş Variations
    'Mansur Yavaş': 'Yavaş',
    'mansur yavaş': 'Yavaş',
    'Mahsur yavaş': 'Yavaş',
    'mansur': 'Yavaş',
    'Mansur': 'Yavaş',

    # Muharrem İnce Variations
    'Muharrem İnce': 'İnce',
    'muharrem ince': 'İnce',
    'İnce': 'İnce',

    # Ümit Özdağ Variations
    'Ümit Özdağ': 'Özdağ',
    'ümit özdağ': 'Özdağ',
    'Umit ozdag': 'Özdağ',

    # Hakan Fidan Variations
    'Hakan Fidan': 'Fidan',
    'hakan fidan': 'Fidan',

    # Selahattin Demirtaş Variations
    'Selahattin Demirtaş': 'Demirtaş',
    'selahattin demirtaş': 'Demirtaş',

    # Other Single Candidates
    'Yavuz Ağıralioğlu': 'Ağıralioğlu',
    'Ali Babacan': 'Babacan',
    'Fatih Erbakan': 'Erbakan',
    'Erkan Baş': 'Baş',
    'Selçuk Bayraktar': 'Bayraktar',
    'Özgür Özel': 'Özel',
    'Kemal Kılıçdaroğlu': 'Kılıçdaroğlu',
    'İbrahim Kalın': 'Kalın',
    'Oğuz Ergin': 'Ergin',
    
    # Combined Preferences
    'Ekrem imamoğlu veya mansur yavaş.': 'İmamoğlu or Yavaş',
    'Ekrem İmamoğlu, Mansur Yavaş': 'İmamoğlu or Yavaş',
    'Ekrem İmamoğlu / Mansur Yavaş': 'İmamoğlu or Yavaş',
    'Ekrem İmamoğlu veya Mansur Yavaş': 'İmamoğlu or Yavaş',
    'İmamoğlu, Yavaş': 'İmamoğlu or Yavaş',
    'İNCE/YAVAŞ': 'İnce or Yavaş',
    'Mansur Yavaş/Muharrem İnce': 'İnce or Yavaş',
    'Muharrem İnce veya Mansur Yavaş': 'İnce or Yavaş',
    'Ümit Özdağ/Mansur Yavaş': 'Özdağ or Yavaş',
    'Ümit Özdağ, Mansur Yavaş': 'Özdağ or Yavaş',
    'Ümit Özdağ/Muharrem İnce': 'Özdağ or İnce',
    'Ekrem İmamoğlu, Muharrem İnce': 'İmamoğlu or İnce',
    'Hakan Fidan veya Mansur Yavaş': 'Fidan or Yavaş',
    'Ümit Özdağ Muharrem İnce Mansur Yavaş': 'Özdağ, İnce, or Yavaş',

    # Generic/Conditional Preferences
    'Akp karşısında kim varsa': 'Strongest Opposition Candidate',
    'Erdoğan hariç herkes': 'Strongest Opposition Candidate',
    'Rte karşıtı en güçlü aday': 'Strongest Opposition Candidate',
    'Mevcut başkanın karşısındaki en güçlü adaya': 'Strongest Opposition Candidate',
    'En güçlü muhalefet adayı': 'Strongest Opposition Candidate',
    'Muhalefetin adayına': 'Strongest Opposition Candidate',
    'İktidarın karşısında kim olursa': 'Strongest Opposition Candidate',
    'Erdoğanın karşısında tuvalet terliği olsa tuvalet terliğine oy verirm': 'Strongest Opposition Candidate',
    'CHP nin adayı kim olursa ona veririm': 'CHP Candidate',
    'CHP': 'CHP Candidate',
    'chp': 'CHP Candidate',
    'Cumhuriyet Halk Partisi': 'CHP Candidate',
    'Zafer parti': 'Zafer Party Candidate',
    'Muhalefet': 'Opposition Candidate',

    # Undecided/Depends on Candidate
    'Adaya Göre Değişir': 'Undecided/Depends on Candidate',
    'Kararsız': 'Undecided/Depends on Candidate',
    'Bilmiyorum': 'Undecided/Depends on Candidate',
    'Emin değilim': 'Undecided/Depends on Candidate',
    'Adayları görelim': 'Undecided/Depends on Candidate',
    'Seçenekler?': 'Undecided/Depends on Candidate',
    'Henüz resmi aday yok.': 'Undecided/Depends on Candidate',
    'Adaylar önemli': 'Undecided/Depends on Candidate',

    # Will Not Vote / Invalid
    'Kimseye': 'None/Will Not Vote',
    'Kimseye.': 'None/Will Not Vote',
    'Hiç kimseye': 'None/Will Not Vote',
    'Vermem': 'None/Will Not Vote',
    'Oy kullanmam': 'None/Will Not Vote',
    'Oy kullanmayacağım': 'None/Will Not Vote',
    'Geçersiz oy': 'None/Will Not Vote',
    'Boş': 'None/Will Not Vote',
    'Hiçbiri': 'None/Will Not Vote',
    'Hiç kimse': 'None/Will Not Vote',
    'Kullanmayacağım.': 'None/Will Not Vote',
    'Hiçliğe': 'None/Will Not Vote',
}
df_clean['current_preference'] = df_clean['current_preference'].map(current_mapping)

# Convert Likert scale responses to numeric
likert_columns = ['istanbul_election_cancel', 'friendship_akp', 'friendship_chp', 
                 'imamoglu_prison', 'ozdag_prison', 'demirtas_prison', 'vote_opposite_party',
                 'support_lawbreaking', 'early_election', 'end_presidential_system',
                 'akp_chp_coalition', 'boycotts_effective', 'new_constitution', 'solution_process']

for col in likert_columns:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

print("Data cleaning completed successfully!")
print(f"Age distribution:\n{df_clean['age'].value_counts()}")
print(f"\nEducation distribution:\n{df_clean['education'].value_counts()}")
print(f"\nIncome distribution:\n{df_clean['income'].value_counts()}")

# %%
# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except (OSError, LookupError):
    pass

print("=== SEMANTIC ANALYSIS PIPELINE FOR POLITICAL PREFERENCES ===")

# %%
# Base Political Semantic Analyzer class
class PoliticalSemanticAnalyzer:
    def __init__(self):
        # Core candidate mappings with all variations
        self.candidate_patterns = {
            'Erdoğan': {
                'primary': ['Erdoğan', 'Recep Tayyip Erdoğan', 'RTE', 'R.T.E', 'R.T.E.', 'Reis', 'Tayyip', 'Recep Tayyip'],
                'abbreviations': ['rte', 'r.t.e', 'r.t.e.', 'rt.e'],
                'variations': ['erdogan', 'recep tayyip', 'tayyip erdogan', 'erdoğan.', 'erdogan.']
            },
            'İmamoğlu': {
                'primary': ['Ekrem İmamoğlu', 'İmamoğlu', 'Ekrem', 'E.İ', 'E.İ.', 'Eko', 'Ekrem İmam'],
                'abbreviations': ['e.i', 'e.i.', 'ei', 'eko'],
                'variations': ['imamoglu', 'ekrem imamoglu', 'ekrem imamoğlu', 'imamov', 'imam oğlu']
            },
            'Yavaş': {
                'primary': ['Mansur Yavaş', 'Yavaş', 'Mansur', 'M.Y', 'M.Y.'],
                'abbreviations': ['m.y', 'm.y.', 'my'],
                'variations': ['mansur yavas', 'yavas', 'mahsur yavaş']
            },
            'İnce': {
                'primary': ['Muharrem İnce', 'İnce', 'Muharrem', 'M.İ', 'M.İ.'],
                'abbreviations': ['m.i', 'm.i.', 'mi'],
                'variations': ['muharrem ince', 'ince']
            },
            'Özdağ': {
                'primary': ['Ümit Özdağ', 'Özdağ', 'Ümit', 'Ü.Ö', 'Ü.Ö.'],
                'abbreviations': ['u.o', 'ü.ö', 'uo', 'üö'],
                'variations': ['umit ozdag', 'ozdag', 'ümit ozdag']
            },
            'Kılıçdaroğlu': {
                'primary': ['Kemal Kılıçdaroğlu', 'Kılıçdaroğlu', 'Kemal', 'K.K', 'K.K.'],
                'abbreviations': ['k.k', 'k.k.', 'kk'],
                'variations': ['kilicdaroglu', 'kemal kilicdaroglu']
            }
        }
        
        # Conditional patterns
        self.conditional_patterns = {
            'strongest_opposition': [
                r'rte?\s*karşısında.*en\s*güçlü',
                r'erdoğan.*karşısında.*güçlü',
                r'iktidar.*karşısında',
                r'akp.*karşısında',
                r'erdoğan.*hariç',
                r'tuvalet terliği',
                r'en\s*güçlü.*muhalefet',
                r'muhalefet.*en\s*güçlü'
            ],
            'chp_candidate': [
                r'chp.*aday',
                r'cumhuriyet.*halk.*parti',
                r'chp\s*nin\s*aday'
            ],
            'will_not_vote': [
                r'kimseye',
                r'hiç\s*kimse',
                r'vermem',
                r'kullanmam',
                r'geçersiz',
                r'boş\s*oy',
                r'hiçbir'
            ],
            'undecided': [
                r'adaya?\s*göre',
                r'kararsız',
                r'bilmiyorum',
                r'emin\s*değil',
                r'adayları?\s*görelim',
                r'seçenekler\?',
                r'henüz.*aday\s*yok'
            ]
        }
        
        # Store all processed responses for review
        self.processed_responses = []
    
    def normalize_text(self, text):
        """Normalize text for better matching"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower().strip()
        # Remove extra spaces and punctuation, keep Turkish characters
        text = re.sub(r'[^\w\sğıişçöü]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def check_conditional_patterns(self, text):
        """Check if text matches any conditional patterns"""
        normalized = self.normalize_text(text)
        
        for pattern_type, patterns in self.conditional_patterns.items():
            for pattern in patterns:
                if re.search(pattern, normalized, re.IGNORECASE):
                    return pattern_type
        return None
    
    def analyze_response(self, user_response):
        """Basic analysis of a single response"""
        if pd.isna(user_response) or user_response == '':
            return {
                'category': 'Yanıt Yok',
                'candidates': [],
                'confidence': 1.0,
                'reasoning': 'Boş veya eksik yanıt',
                'original': user_response
            }
        
        # Check conditional patterns first
        conditional_type = self.check_conditional_patterns(user_response)
        if conditional_type:
            category_map = {
                'strongest_opposition': 'En Güçlü Muhalefet Adayı',
                'chp_candidate': 'CHP Adayı',
                'will_not_vote': 'Kimseye/Oy Vermeyecek',
                'undecided': 'Kararsız/Adaya Bağlı'
            }
            return {
                'category': category_map.get(conditional_type, 'Karmaşık Koşullu'),
                'candidates': [],
                'confidence': 0.8,
                'reasoning': f'Koşullu desen: {conditional_type}',
                'original': user_response
            }
        
        # Basic candidate extraction
        normalized = self.normalize_text(user_response)
        for candidate, patterns in self.candidate_patterns.items():
            all_patterns = patterns['primary'] + patterns['abbreviations'] + patterns['variations']
            for pattern in all_patterns:
                if self.normalize_text(pattern) in normalized:
                    return {
                        'category': candidate,
                        'candidates': [candidate],
                        'confidence': 0.7,
                        'reasoning': f'Aday eşleşmesi: {pattern}',
                        'original': user_response
                    }
        
        return {
            'category': 'Kategorize Edilmemiş',
            'candidates': [],
            'confidence': 0.0,
            'reasoning': 'Tanınabilir desen bulunamadı',
            'original': user_response
        }

# %%
# Enhanced semantic analyzer with improved pattern recognition
class EnhancedPoliticalSemanticAnalyzer(PoliticalSemanticAnalyzer):
    def __init__(self):
        super().__init__()
        # Add Hakan Fidan to candidate patterns
        self.candidate_patterns['Fidan'] = {
            'primary': ['Hakan Fidan', 'Fidan', 'H.F', 'H.F.'],
            'abbreviations': ['h.f', 'h.f.', 'hf'],
            'variations': ['hakan fidan', 'haka fidan']
        }
        # Enhanced conditional patterns
        self.conditional_patterns.update({
            'strongest_opposition': [
                r'rte?\s*karşısında.*en\s*güçlü',
                r'erdoğan.*karşısında.*güçlü',
                r'iktidar.*karşısında',
                r'akp.*karşısında',
                r'erdoğan.*hariç',
                r'erdoğan.*harici',
                r'en\s*güçlü.*muhalif',
                r'en\s*güçlü.*aday',
                r'tuvalet terliği',
                r'en\s*güçlü.*muhalefet',
                r'muhalefet.*en\s*güçlü',
                r'karşıdaki.*adaya',
                r'muhalefet.*adayına',
                r'erdoğan.*dışında',
                r'hükümet.*dışında',
                r'en\s*güçlü.*muhalif',
                r'muhalefetin.*adayına',
                r'muhalefetin.*adayı',
                r'muhalefetin.*belirlediği',
                r'ana.*muhalefete',
                r'muhalefet.*adayları.*içinden',
                r'tayyibin.*en.*güçlü.*rakibi',
                r'mevcut.*başkanın.*karşısındaki',
                r'erdoğan.*rakibine',
                r'mevcut.*cb.*dışında',
                r'muhalefette.*sandalye',
                r'aday.*farketmeksizin.*muhalefete',
                r'akp.*haricinde',
                r'rte.*hariç',
                r'su\s*şişesi.*veririm',
                r'erdoğanın.*karşısında.*tuvalet.*terliği',
                r'malum.*kişinin.*karşısında',
                r'rte.*nin.*karşısına.*muhalif',
                r'terörist.*olmayan.*ve.*rte.*harici',
                r'isim.*şehir.*bitki.*hayvan.*eşya.*farketmez',
                r'kim.*konulursa.*ona.*oy'
            ],
            'opposition_candidate': [
                r'muhalefet.*aday',
                r'karşı.*aday',
                r'muhalif.*aday',
                r'opposition.*candidate'
            ],
            'conditional_imamoglu': [
                r'ekrem.*imamoğlu.*erdoğan.*karşısında.*kim.*varsa',
                r'ekrem.*imamoğlu.*recep.*tayyip.*karşısında'
            ],
            'chp_candidate': [
                r'chp.*aday',
                r'cumhuriyet.*halk.*parti',
                r'chp\s*nin\s*aday',
                r'^chp$',
                r'chp.*seçmenlerinden',
                r'mecbur.*chp'
            ],
            'akp_candidate': [
                r'cumhur.*ittifakı.*adayına',
                r'akp.*adayına',
                r'millet.*ittifakı.*dışında',
                r'sağ.*kesimden.*daha.*iyi.*aday'
            ],
            'nationalist_opposition': [
                r'milliyetçi.*muhalefet',
                r'mhp.*adayına',
                r'ilkesi.*türkçülük.*olan',
                r'milliyetçi.*aday',
                r'türkçü.*aday',
                r'milliyetçi.*muhalif'
            ],
            'will_not_vote': [
                r'^❌$',
                r'^yok$',
                r'^kimse$',
                r'^\.$',
                r'kimseye',
                r'hiç\s*kimse',
                r'oy.*kullanmayı.*düşünmüyorum',
                r'oy.*vermeyeceğim',
                r'boş.*oy'
            ],
           
        })
        
        # Add additional patterns
        self.conditional_patterns.update({
            'undecided': [
                r'adaya?\s*göre',
                r'kararsız',
                r'bilmiyorum',
                r'emin\s*değil',
                r'adayları?\s*görelim',
                r'seçenekler\?',
                r'henüz.*aday\s*yok',
                r'adaylar.*önemli',
                r'karar.*vermedim',
                r'adayları.*görmeden',
                r'adaylara.*bağlı',
                r'başka.*bir.*adaya',
                r'içerde.*muharrem.*ince.*ise.*hala.*pek.*güven.*vermiyor'
            ]
        })
        
        # Direct text mappings for specific cases
        self.direct_mappings = {
            'reis': 'Erdoğan',
            'imamson': 'İmamoğlu',
            'imaro': 'İmamoğlu',
            'eko': 'İmamoğlu',
            'ekrem': 'İmamoğlu',
            'mansur': 'Yavaş',
            'muharrem': 'İnce'
        }

    def extract_candidates_from_text(self, text):
        """Enhanced candidate extraction with better precedence rules"""
        normalized = self.normalize_text(text)
        found_candidates = []
        
        # Special handling for texts that mention both Erdoğan and other candidates
        # If it's clearly about supporting Erdoğan, prioritize him
        if any(phrase in normalized for phrase in ['halkın adamı', 'hakkın aşığı', 'recep tayyip erdoğan']):
            if 'erdoğan' in normalized or 'recep tayyip' in normalized:
                return ['Erdoğan']
        
        for candidate, patterns in self.candidate_patterns.items():
            # Check all pattern types
            all_patterns = patterns['primary'] + patterns['abbreviations'] + patterns['variations']
            
            for pattern in all_patterns:
                pattern_norm = self.normalize_text(pattern)
                
                # Exact match
                if pattern_norm in normalized:
                    found_candidates.append(candidate)
                    break
                
                # Fuzzy match for longer patterns
                if len(pattern_norm) > 3:
                    ratio = fuzz.partial_ratio(pattern_norm, normalized)
                    if ratio > 85:  # High threshold for candidate names
                        found_candidates.append(candidate)
                        break
        
        return list(set(found_candidates))

    def analyze_response_detailed(self, user_response, show_steps=True):
        """Enhanced analysis with detailed step-by-step pipeline explanation"""
        if pd.isna(user_response) or user_response == '':
            analysis_result = {
                'category': 'Yanıt Yok',
                'candidates': [],
                'confidence': 1.0,
                'reasoning': 'Boş veya eksik yanıt',
                'original': user_response,
                'pipeline_steps': ['Girdi doğrulama: Boş/NaN yanıt tespit edildi']
            }
            if show_steps:
                print(f"📝 Orijinal: '{user_response}' → 🏷️ Kategori: '{analysis_result['category']}'")
                print(f"   İşlem hattı: {' → '.join(analysis_result['pipeline_steps'])}")
                print(f"   Güven: {analysis_result['confidence']:.1f}\n")
            return analysis_result
        
        original_response = str(user_response)
        normalized = self.normalize_text(user_response)
        pipeline_steps = []
        pipeline_steps.append(f"Girdi: '{original_response}'")
        pipeline_steps.append(f"Normalleştirildi: '{normalized}'")
        
        # Check for conditional patterns FIRST
        conditional_type = self.check_conditional_patterns(user_response)
        if conditional_type:
            pipeline_steps.append(f"Koşullu desen eşleşti: {conditional_type}")
            
            category_map = {
                'strongest_opposition': 'En Güçlü Muhalefet Adayı',
                'conditional_erdogan': 'Koşullu: Erdoğan',
                'conditional_imamoglu': 'Koşullu: İmamoğlu',
                'chp_candidate': 'CHP Adayı',
                'akp_candidate': 'AKP/Cumhur İttifakı Adayı',
                'nationalist_opposition': 'Milliyetçi Muhalif Aday',
                'depends_on_candidate': 'Kararsız/Adaya Bağlı',
                'undecided': 'Kararsız/Adaya Bağlı',
                'will_not_vote': 'Kimseye/Oy Vermeyecek',
                'mhp_candidate': 'MHP Adayı'
            }
            
            analysis_result = {
                'category': category_map.get(conditional_type, 'Karmaşık Koşullu'),
                'candidates': [],
                'confidence': 0.9,
                'reasoning': f'Koşullu desen eşleşti: {conditional_type}',
                'original': original_response,
                'pipeline_steps': pipeline_steps
            }
            
            if show_steps:
                print(f"📝 Orijinal: '{original_response}' → 🏷️ Kategori: '{analysis_result['category']}'")
                print(f"   İşlem hattı: {' → '.join(pipeline_steps)}")
                print(f"   Güven: {analysis_result['confidence']:.1f}\n")
            return analysis_result
        else:
            pipeline_steps.append("Koşullu desen eşleşmedi")
        
        # Check direct mappings second
        direct_match_found = False
        for mapping_key, mapping_value in self.direct_mappings.items():
            if mapping_key in normalized:
                pipeline_steps.append(f"Doğrudan eşleme bulundu: '{mapping_key}' → '{mapping_value}'")
                direct_match_found = True
                
                analysis_result = {
                    'category': mapping_value,
                    'candidates': [mapping_value],
                    'confidence': 0.9,
                    'reasoning': f'Doğrudan eşleme bulundu: {mapping_key} → {mapping_value}',
                    'original': original_response,
                    'pipeline_steps': pipeline_steps
                }
                
                if show_steps:
                    print(f"📝 Orijinal: '{original_response}' → 🏷️ Kategori: '{analysis_result['category']}'")
                    print(f"   İşlem hattı: {' → '.join(pipeline_steps)}")
                    print(f"   Güven: {analysis_result['confidence']:.1f}\n")
                return analysis_result
        
        if not direct_match_found:
            pipeline_steps.append("Doğrudan eşleme bulunamadı")
        
        # Extract candidate mentions last
        candidates = self.extract_candidates_from_text(user_response)
        pipeline_steps.append(f"Aday çıkarma: {candidates if candidates else 'Bulunamadı'}")
        
        if not candidates:
            analysis_result = {
                'category': 'Kategorize Edilmemiş',
                'candidates': [],
                'confidence': 0.0,
                'reasoning': 'Tanınabilir aday veya desen bulunamadı',
                'original': original_response,
                'pipeline_steps': pipeline_steps
            }
            
            if show_steps:
                print(f"📝 Orijinal: '{original_response}' → 🏷️ Kategori: '{analysis_result['category']}'")
                print(f"   İşlem hattı: {' → '.join(pipeline_steps)}")
                print(f"   Güven: {analysis_result['confidence']:.1f}\n")
            return analysis_result
        
        # Always take the first candidate when multiple candidates are found
        if len(candidates) >= 1:
            first_candidate = candidates[0]
            if len(candidates) > 1:
                pipeline_steps.append(f"Birden fazla aday bulundu, ilki seçildi: '{first_candidate}'")
            else:
                pipeline_steps.append(f"Tek aday tanımlandı: '{first_candidate}'")
            
            analysis_result = {
                'category': first_candidate,
                'candidates': [first_candidate],
                'confidence': 0.8 if len(candidates) == 1 else 0.75,
                'reasoning': f'Birincil aday tanımlandı: {first_candidate}' + (f' ({len(candidates)} adaydan)' if len(candidates) > 1 else ''),
                'original': original_response,
                'pipeline_steps': pipeline_steps
            }
            
            if show_steps:
                print(f"📝 Orijinal: '{original_response}' → 🏷️ Kategori: '{analysis_result['category']}'")
                print(f"   İşlem hattı: {' → '.join(pipeline_steps)}")
                print(f"   Güven: {analysis_result['confidence']:.1f}\n")
            return analysis_result

    def analyze_response(self, user_response):
        """Enhanced analysis with improved pattern matching (backwards compatibility)"""
        return self.analyze_response_detailed(user_response, show_steps=False)

    def analyze_all_responses_detailed(self, responses_list, max_display=50):
        """Analyze all responses with detailed pipeline explanation"""
        print("=== DETAYLI İŞLEM HATTI ANALİZİ ===")
        print(f"{len(responses_list)} yanıt analiz ediliyor (ilk {max_display} tanesi detaylı gösteriliyor)...\n")
        
        analysis_results = []
        for idx, user_response in enumerate(responses_list):
            show_details = idx < max_display
            if show_details:
                print(f"--- Yanıt {idx+1} ---")
            
            analysis = self.analyze_response_detailed(user_response, show_steps=show_details)
            analysis_results.append(analysis)
            self.processed_responses.append(analysis)
            
            if show_details and idx == max_display - 1:
                print(f"... (ilk {max_display} yanıt detaylı gösteriliyor)")
                print(f"Kalan {len(responses_list) - max_display} yanıt işleniyor...")
        
        return analysis_results

    def get_pipeline_summary(self):
        """Generate summary of how the pipeline processed responses"""
        print("\n=== İŞLEM HATTI ÖZETİ ===")
        
        # Count processing methods
        conditional_count = len([r for r in self.processed_responses if 'koşullu desen' in r['reasoning'].lower()])
        direct_mapping_count = len([r for r in self.processed_responses if 'doğrudan eşleme' in r['reasoning'].lower()])
        candidate_extraction_count = len([r for r in self.processed_responses if 'aday tanımlandı' in r['reasoning'].lower()])
        uncategorized_count = len([r for r in self.processed_responses if r['category'] == 'Kategorize Edilmemiş'])
        no_response_count = len([r for r in self.processed_responses if r['category'] == 'Yanıt Yok'])
        
        print(f"Toplam işlenen yanıt: {len(self.processed_responses)}")
        print(f"📋 Koşullu desenler: {conditional_count} ({conditional_count/len(self.processed_responses)*100:.1f}%)")
        print(f"🎯 Doğrudan eşlemeler: {direct_mapping_count} ({direct_mapping_count/len(self.processed_responses)*100:.1f}%)")
        print(f"🔍 Aday çıkarma: {candidate_extraction_count} ({candidate_extraction_count/len(self.processed_responses)*100:.1f}%)")
        print(f"❓ Kategorize edilmemiş: {uncategorized_count} ({uncategorized_count/len(self.processed_responses)*100:.1f}%)")
        print(f"🚫 Yanıt yok: {no_response_count} ({no_response_count/len(self.processed_responses)*100:.1f}%)")
        
        # Show examples of each method
        print("\n=== İŞLEM YÖNTEMİ ÖRNEKLERİ ===")
        
        conditional_examples = [r for r in self.processed_responses if 'koşullu desen' in r['reasoning'].lower()][:3]
        if conditional_examples:
            print("\n📋 Koşullu Desen Örnekleri:")
            for ex in conditional_examples:
                print(f"   '{ex['original']}' → '{ex['category']}'")
        
        direct_examples = [r for r in self.processed_responses if 'doğrudan eşleme' in r['reasoning'].lower()][:3]
        if direct_examples:
            print("\n🎯 Doğrudan Eşleme Örnekleri:")
            for ex in direct_examples:
                print(f"   '{ex['original']}' → '{ex['category']}'")
        
        candidate_examples = [r for r in self.processed_responses if 'aday tanımlandı' in r['reasoning'].lower()][:3]
        if candidate_examples:
            print("\n🔍 Aday Çıkarma Örnekleri:")
            for ex in candidate_examples:
                print(f"   '{ex['original']}' → '{ex['category']}'")
        
        uncategorized_examples = [r for r in self.processed_responses if r['category'] == 'Kategorize Edilmemiş'][:5]
        if uncategorized_examples:
            print("\n❓ Kategorize Edilmemiş Örnekler (manuel inceleme gerekli):")
            for ex in uncategorized_examples:
                print(f"   '{ex['original']}'")
                

# Initialize enhanced analyzer
enhanced_analyzer = EnhancedPoliticalSemanticAnalyzer()
print("Enhanced semantic analyzer initialized!")

# Get all current preferences (including NaN values) - define this early
all_preferences = df['current_preference'].fillna('').tolist()

# Re-process all preferences with enhanced analyzer using detailed analysis
print("=== RE-PROCESSING WITH ENHANCED PATTERNS (DETAILED) ===")
enhanced_semantic_results = enhanced_analyzer.analyze_all_responses_detailed(all_preferences, max_display=50) # reduced for cleaner output

# Generate pipeline summary
enhanced_analyzer.get_pipeline_summary()

# Create enhanced results DataFrame
enhanced_semantic_df = pd.DataFrame(enhanced_semantic_results)

print(f"\nProcessed {len(enhanced_semantic_results)} responses with enhanced analyzer")
print("\nEnhanced Categorization Results:")
print(enhanced_semantic_df['category'].value_counts())

# Show improvements for the specific cases mentioned
test_cases = [
    'Karşıdaki adaya.',
    'Muhalefet adayına',
    'Erdoğan dışında birine',
    'Muhalefet adayı',
    'En güçlü muhalif adaya',
    'CHP',
    'Reis',
    'Hakan Fidan',
    'imamson',
    'Oy kullanmayı düşünmüyorum',
    '❌',
    'Yok'
]

print("\n=== TESTING SPECIFIC CASES WITH DETAILED PIPELINE ===")
for test_case in test_cases:
    print(f"Testing: '{test_case}'")
    test_result = enhanced_analyzer.analyze_response_detailed(test_case, show_steps=True)

print("\n✅ Enhanced semantic analysis completed successfully!")
print("🔧 All compilation and runtime errors have been fixed.")
print("🎯 Script is now ready for execution and further analysis.")

# %%

# Duplicate class definition removed - already defined above

# Initialize enhanced analyzer
enhanced_analyzer = EnhancedPoliticalSemanticAnalyzer()
print("Enhanced semantic analyzer initialized!")

# Get all current preferences (including NaN values) - define this early
all_preferences = df['current_preference'].fillna('').tolist()

# Re-process all preferences with enhanced analyzer using detailed analysis
print("=== RE-PROCESSING WITH ENHANCED PATTERNS (DETAILED) ===")
enhanced_semantic_results = enhanced_analyzer.analyze_all_responses_detailed(all_preferences, max_display=1000) # adjust it to see full 11309 response  bu 1000 is enough for testing

# Generate pipeline summary
enhanced_analyzer.get_pipeline_summary()

# Create enhanced results DataFrame
enhanced_semantic_df = pd.DataFrame(enhanced_semantic_results)

print(f"\nProcessed {len(enhanced_semantic_results)} responses with enhanced analyzer")
print("\nEnhanced Categorization Results:")
print(enhanced_semantic_df['category'].value_counts())

# Show improvements for the specific cases mentioned
test_cases = [
    'Karşıdaki adaya.',
    'Muhalefet adayına',
    'Erdoğan dışında birine',
    'Muhalefet adayı',
    'En güçlü muhalif adaya',
    'CHP',
    'Reis',
    'Hakan Fidan',
    'imamson',
    'Oy kullanmayı düşünmüyorum',
    '❌',
    'Yok'
]

print("\n=== TESTING SPECIFIC CASES WITH DETAILED PIPELINE ===")
for test_case in test_cases:
    print(f"Testing: '{test_case}'")
    test_analysis_result = enhanced_analyzer.analyze_response_detailed(test_case, show_steps=True)

# %%
# Process current preferences with enhanced semantic analysis first
print("=== PROCESSING CURRENT PREFERENCES WITH ENHANCED SEMANTIC ANALYSIS ===")

# Define likert_columns for use in analysis
likert_columns = ['istanbul_election_cancel', 'friendship_akp', 'friendship_chp', 
                 'imamoglu_prison', 'ozdag_prison', 'demirtas_prison', 'vote_opposite_party',
                 'support_lawbreaking', 'early_election', 'end_presidential_system',
                 'akp_chp_coalition', 'boycotts_effective', 'new_constitution', 'solution_process']

# Apply enhanced semantic mapping to create a new column
def apply_enhanced_semantic_mapping(text):
    """Apply enhanced semantic analysis results to categorize responses"""
    if pd.isna(text) or text == '':
        return None
        
    # Use enhanced analyzer for better categorization
    enhanced_result = enhanced_analyzer.analyze_response(text)
    if enhanced_result['confidence'] >= 0.7:
        return enhanced_result['category']
    else:
        # For low confidence, try to map to existing categories
        response_category = enhanced_result['category'].lower()
        if 'erdoğan' in response_category:
            return 'Erdoğan'
        elif 'imamoğlu' in response_category:
            return 'İmamoğlu'
        elif 'yavaş' in response_category:
            return 'Yavaş'
        elif 'ince' in response_category:
            return 'İnce'
        elif 'özdağ' in response_category:
            return 'Özdağ'
        elif 'fidan' in response_category:
            return 'Fidan'
        else:
            return 'Undecided/Depends on Candidate'

# Create enhanced semantic preference column
df_enhanced = df_clean.copy()
df_enhanced['current_preference_enhanced'] = df['current_preference'].apply(apply_enhanced_semantic_mapping)

# Add visualizations for current_preference analysis
print("\n=== CURRENT PREFERENCE ANALYSIS ===")

# Compare original vs enhanced semantic analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Current Political Preferences - Original vs Enhanced Analysis', fontsize=16, fontweight='bold')

# Original current preferences
original_prefs = df_clean['current_preference'].value_counts().head(10)
axes[0,0].bar(range(len(original_prefs)), original_prefs.values)
axes[0,0].set_xticks(range(len(original_prefs)))
axes[0,0].set_xticklabels(original_prefs.index, rotation=45, ha='right')
axes[0,0].set_title('Original Current Preferences (Top 10)', fontweight='bold')
axes[0,0].set_ylabel('Count')

# Enhanced current preferences
enhanced_prefs = df_enhanced['current_preference_enhanced'].value_counts().head(10)
axes[0,1].bar(range(len(enhanced_prefs)), enhanced_prefs.values, color='orange')
axes[0,1].set_xticks(range(len(enhanced_prefs)))
axes[0,1].set_xticklabels(enhanced_prefs.index, rotation=45, ha='right')
axes[0,1].set_title('Enhanced Current Preferences (Top 10)', fontweight='bold')
axes[0,1].set_ylabel('Count')

# Pie chart for enhanced preferences
top_enhanced = df_enhanced['current_preference_enhanced'].value_counts().head(8)
other_count = df_enhanced['current_preference_enhanced'].value_counts().iloc[8:].sum()
if other_count > 0:
    pie_data = list(top_enhanced.values) + [other_count]
    pie_labels = list(top_enhanced.index) + ['Others']
else:
    pie_data = top_enhanced.values
    pie_labels = top_enhanced.index

axes[1,0].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
axes[1,0].set_title('Enhanced Preferences Distribution', fontweight='bold')

# Confidence levels distribution
confidence_levels = pd.cut(enhanced_semantic_df['confidence'], 
                          bins=[0, 0.7, 0.8, 0.9, 1.0], 
                          labels=['Low (<0.7)', 'Medium (0.7-0.8)', 'High (0.8-0.9)', 'Very High (≥0.9)'])
conf_counts = confidence_levels.value_counts()
axes[1,1].bar(conf_counts.index, conf_counts.values, color=['red', 'orange', 'lightgreen', 'green'])
axes[1,1].set_title('Analysis Confidence Levels', fontweight='bold')
axes[1,1].set_ylabel('Count')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Current preference by demographics
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Current Political Preferences by Demographics (Enhanced)', fontsize=16, fontweight='bold')

# By age group
pref_age = pd.crosstab(df_enhanced['age'], df_enhanced['current_preference_enhanced'])
pref_age_pct = pref_age.div(pref_age.sum(axis=1), axis=0) * 100
top_prefs_for_heatmap = df_enhanced['current_preference_enhanced'].value_counts().head(6).index
pref_age_top = pref_age_pct[top_prefs_for_heatmap]
sns.heatmap(pref_age_top, annot=True, fmt='.1f', cmap='RdYlBu', ax=axes[0,0])
axes[0,0].set_title('Preferences by Age Group (%)', fontweight='bold')
axes[0,0].tick_params(axis='x', rotation=45)

# By education
pref_edu = pd.crosstab(df_enhanced['education'], df_enhanced['current_preference_enhanced'])
pref_edu_pct = pref_edu.div(pref_edu.sum(axis=1), axis=0) * 100
pref_edu_top = pref_edu_pct[top_prefs_for_heatmap]
sns.heatmap(pref_edu_top, annot=True, fmt='.1f', cmap='RdYlBu', ax=axes[0,1])
axes[0,1].set_title('Preferences by Education (%)', fontweight='bold')
axes[0,1].tick_params(axis='x', rotation=45)

# By gender
pref_gender = pd.crosstab(df_enhanced['gender'], df_enhanced['current_preference_enhanced'])
pref_gender_pct = pref_gender.div(pref_gender.sum(axis=1), axis=0) * 100
pref_gender_top = pref_gender_pct[top_prefs_for_heatmap]
sns.heatmap(pref_gender_top, annot=True, fmt='.1f', cmap='RdYlBu', ax=axes[1,0])
axes[1,0].set_title('Preferences by Gender (%)', fontweight='bold')
axes[1,0].tick_params(axis='x', rotation=45)

# By last vote
pref_lastvote = pd.crosstab(df_enhanced['last_vote'], df_enhanced['current_preference_enhanced'])
pref_lastvote_pct = pref_lastvote.div(pref_lastvote.sum(axis=1), axis=0) * 100
pref_lastvote_top = pref_lastvote_pct[top_prefs_for_heatmap]
sns.heatmap(pref_lastvote_top, annot=True, fmt='.1f', cmap='RdYlBu', ax=axes[1,1])
axes[1,1].set_title('Current vs Last Vote (%)', fontweight='bold')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Political attitudes by current preference
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Political Attitudes by Current Preference (Enhanced)', fontsize=16, fontweight='bold')

# Top 5 preferences for detailed analysis
top_5_prefs = df_enhanced['current_preference_enhanced'].value_counts().head(5).index

# Early election support by preference
early_election_by_pref = df_enhanced.groupby('current_preference_enhanced')['early_election'].mean()
early_election_top5 = early_election_by_pref[top_5_prefs]
axes[0,0].bar(range(len(early_election_top5)), early_election_top5.values)
axes[0,0].set_xticks(range(len(early_election_top5)))
axes[0,0].set_xticklabels(early_election_top5.index, rotation=45, ha='right')
axes[0,0].set_title('Early Election Support by Preference', fontweight='bold')
axes[0,0].set_ylabel('Mean Score (1-7)')

# End presidential system by preference
presidential_by_pref = df_enhanced.groupby('current_preference_enhanced')['end_presidential_system'].mean()
presidential_top5 = presidential_by_pref[top_5_prefs]
axes[0,1].bar(range(len(presidential_top5)), presidential_top5.values, color='orange')
axes[0,1].set_xticks(range(len(presidential_top5)))
axes[0,1].set_xticklabels(presidential_top5.index, rotation=45, ha='right')
axes[0,1].set_title('End Presidential System by Preference', fontweight='bold')
axes[0,1].set_ylabel('Mean Score (1-7)')

# AKP-CHP coalition support by preference
coalition_by_pref = df_enhanced.groupby('current_preference_enhanced')['akp_chp_coalition'].mean()
coalition_top5 = coalition_by_pref[top_5_prefs]
axes[1,0].bar(range(len(coalition_top5)), coalition_top5.values, color='green')
axes[1,0].set_xticks(range(len(coalition_top5)))
axes[1,0].set_xticklabels(coalition_top5.index, rotation=45, ha='right')
axes[1,0].set_title('AKP-CHP Coalition Support by Preference', fontweight='bold')
axes[1,0].set_ylabel('Mean Score (1-7)')

# New constitution support by preference
constitution_by_pref = df_enhanced.groupby('current_preference_enhanced')['new_constitution'].mean()
constitution_top5 = constitution_by_pref[top_5_prefs]
axes[1,1].bar(range(len(constitution_top5)), constitution_top5.values, color='red')
axes[1,1].set_xticks(range(len(constitution_top5)))
axes[1,1].set_xticklabels(constitution_top5.index, rotation=45, ha='right')
axes[1,1].set_title('New Constitution Support by Preference', fontweight='bold')
axes[1,1].set_ylabel('Mean Score (1-7)')

plt.tight_layout()
plt.show()

# 2. DESCRIPTIVE STATISTICS
print("\n\n2. DESCRIPTIVE STATISTICS")
print("=" * 50)

# Demographics summary
print("DEMOGRAPHICS SUMMARY:")
print(f"Gender distribution:\n{df_enhanced['gender'].value_counts()}")
print(f"\nMarital status:\n{df_enhanced['marital_status'].value_counts()}")
print(f"\nGeographical distribution (top cities):\n{df_enhanced['city'].value_counts().head(10)}")

# Voting behavior
print("\nVOTING BEHAVIOR:")
print(f"Last presidential vote:\n{df_enhanced['last_vote'].value_counts()}")
print(f"\nCurrent preference (Enhanced Semantic Analysis):\n{df_enhanced['current_preference_enhanced'].value_counts()}")
print("\nOriginal vs Enhanced categorization comparison:")
print(f"Enhanced categories: {df_enhanced['current_preference_enhanced'].nunique()}")
print(f"Original categories: {df_enhanced['current_preference'].nunique()}")

# Political attitudes - Likert scale statistics
print("\nPOLITICAL ATTITUDES (1-7 scale):")
likert_stats = df_enhanced[likert_columns].describe()
print(likert_stats.round(2))

# Main problems mentioned
print("\nMAIN PROBLEMS IDENTIFIED:")
main_problems = df_enhanced['main_problem'].value_counts().head(10)
print(main_problems)

# 3. VISUALIZATION SETUP
print("\n\n3. CREATING VISUALIZATIONS")
print("=" * 50)

# Create comprehensive visualizations
fig, axes = plt.subplots(4, 3, figsize=(20, 24))
fig.suptitle('Turkish Political Survey - Enhanced Semantic Analysis', fontsize=20, fontweight='bold')

# 3.1 Demographics
# Age distribution
age_counts = df_enhanced['age'].value_counts()
axes[0,0].pie(age_counts.values, labels=age_counts.index, autopct='%1.1f%%', startangle=90)
axes[0,0].set_title('Age Distribution', fontweight='bold')

# Income distribution
income_counts = df_enhanced['income'].value_counts()
axes[0,1].pie(income_counts.values, labels=income_counts.index, autopct='%1.1f%%', startangle=90)
axes[0,1].set_title('Income Distribution', fontweight='bold')

# Education level
education_counts = df_enhanced['education'].value_counts()
axes[0,2].barh(education_counts.index, education_counts.values)
axes[0,2].set_title('Education Level Distribution', fontweight='bold')

# 3.2 Voting Preferences (Enhanced)
# Last vote vs current preference (enhanced)
vote_comparison = pd.crosstab(df_enhanced['last_vote'], df_enhanced['current_preference_enhanced'])
sns.heatmap(vote_comparison, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1,0])
axes[1,0].set_title('Last Vote vs Current Preference (Enhanced)', fontweight='bold')
axes[1,0].tick_params(axis='x', rotation=45)

# Current preference by age (enhanced)
current_age = pd.crosstab(df_enhanced['age'], df_enhanced['current_preference_enhanced'])
current_age_pct = current_age.div(current_age.sum(axis=1), axis=0) * 100
sns.heatmap(current_age_pct, annot=True, fmt='.1f', cmap='RdYlBu', ax=axes[1,1])
axes[1,1].set_title('Current Preference by Age (%) - Enhanced', fontweight='bold')

# Current preference by education (enhanced)
current_edu = pd.crosstab(df_enhanced['education'], df_enhanced['current_preference_enhanced'])
current_edu_pct = current_edu.div(current_edu.sum(axis=1), axis=0) * 100
sns.heatmap(current_edu_pct, annot=True, fmt='.1f', cmap='RdYlBu', ax=axes[1,2])
axes[1,2].set_title('Current Preference by Education (%) - Enhanced', fontweight='bold')
axes[1,2].tick_params(axis='x', rotation=45)

# 3.3 Political Attitudes
# Key political attitudes
key_attitudes = ['early_election', 'end_presidential_system', 'akp_chp_coalition', 
                'new_constitution', 'solution_process']
attitude_means = df_enhanced[key_attitudes].mean()
axes[2,0].bar(range(len(attitude_means)), attitude_means.values)
axes[2,0].set_xticks(range(len(attitude_means)))
axes[2,0].set_xticklabels(['Early Election', 'End Presidential', 'AKP-CHP Coalition', 
                          'New Constitution', 'Solution Process'], rotation=45)
axes[2,0].set_title('Political Attitudes (Mean Scores)', fontweight='bold')
axes[2,0].set_ylabel('Mean Score (1-7)')

# Friendship tolerance
friendship_data = df_enhanced[['friendship_akp', 'friendship_chp']].mean()
axes[2,1].bar(['AKP Friendship', 'CHP Friendship'], friendship_data.values)
axes[2,1].set_title('Political Friendship Tolerance', fontweight='bold')
axes[2,1].set_ylabel('Mean Score (1-7)')

# Prison opinions
prison_data = df_enhanced[['imamoglu_prison', 'ozdag_prison', 'demirtas_prison']].mean()
axes[2,2].bar(['İmamoğlu', 'Özdağ', 'Demirtaş'], prison_data.values)
axes[2,2].set_title('Prison Opinions (Mean Scores)', fontweight='bold')
axes[2,2].set_ylabel('Mean Score (1-7)')

# 3.4 Social Media and Behavioral Patterns
# Social media usage
social_media_counts = df_enhanced['social_media_usage'].value_counts()
axes[3,0].pie(social_media_counts.values, labels=social_media_counts.index, autopct='%1.1f%%')
axes[3,0].set_title('Social Media Usage Patterns', fontweight='bold')

# Blocking behavior
blocking_counts = df_enhanced['blocked_friends'].value_counts()
axes[3,1].pie(blocking_counts.values, labels=blocking_counts.index, autopct='%1.1f%%')
axes[3,1].set_title('Blocked Friends Due to Politics', fontweight='bold')

# Unfollowing behavior
unfollow_counts = df_enhanced['unfollowed_influencers'].value_counts()
axes[3,2].pie(unfollow_counts.values, labels=unfollow_counts.index, autopct='%1.1f%%')
axes[3,2].set_title('Unfollowed Influencers', fontweight='bold')

plt.tight_layout()
plt.show()

# Enhanced preference distribution visualization
plt.figure(figsize=(15, 8))
enhanced_pref_counts = df_enhanced['current_preference_enhanced'].value_counts()
plt.bar(range(len(enhanced_pref_counts)), enhanced_pref_counts.values)
plt.xticks(range(len(enhanced_pref_counts)), enhanced_pref_counts.index, rotation=45, ha='right')
plt.title('Current Political Preferences - Enhanced Semantic Analysis', fontsize=16, fontweight='bold')
plt.ylabel('Number of Responses')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. CORRELATION ANALYSIS
print("\n\n4. CORRELATION ANALYSIS")
print("=" * 50)

# Select numeric columns for correlation
numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = df_enhanced[numeric_cols].corr()

# Create correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
            square=True, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Political Attitudes Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Identify strongest correlations
def get_top_correlations(corr_matrix, n=10):
    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find correlations and sort by absolute value
    correlations = []
    for column in upper.columns:
        for idx in upper.index:
            if pd.notna(upper.loc[idx, column]):
                correlations.append((idx, column, upper.loc[idx, column]))
    
    # Sort by absolute correlation value
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    return correlations[:n]

top_correlations = get_top_correlations(correlation_matrix)
print("\nTOP 10 STRONGEST CORRELATIONS:")
for i, (var1, var2, corr) in enumerate(top_correlations, 1):
    print(f"{i:2d}. {var1} ↔ {var2}: {corr:.3f}")

# 5. CLUSTERING ANALYSIS
print("\n\n5. CLUSTERING ANALYSIS")
print("=" * 50)

# Prepare data for clustering
clustering_data = df_enhanced[likert_columns].dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

# Determine optimal number of clusters using elbow method
inertias = []
k_range = range(2, 9)
for k in k_range:
    kmeans = KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)
plt.show()

# Perform clustering with optimal k (let's use 4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# Add cluster labels to dataframe
clustering_df = clustering_data.copy()
clustering_df['cluster'] = cluster_labels

# Analyze cluster characteristics
print(f"\nCLUSTER ANALYSIS (k={optimal_k}):")
cluster_summary = clustering_df.groupby('cluster')[likert_columns].mean()
print(cluster_summary.round(2))

# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
for i in range(optimal_k):
    cluster_mask = cluster_labels == i
    plt.scatter(pca_data[cluster_mask, 0], pca_data[cluster_mask, 1], 
                c=colors[i], label=f'Cluster {i+1}', alpha=0.7, s=50)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Political Clusters Visualization (PCA)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Enhanced Analysis Summary
print("\n\n=== ENHANCED SEMANTIC ANALYSIS SUMMARY ===")
print(f"Total responses processed: {len(enhanced_semantic_results)}")
print("Enhanced categorization confidence levels:")
confidence_levels = pd.cut(enhanced_semantic_df['confidence'], 
                          bins=[0, 0.7, 0.8, 0.9, 1.0], 
                          labels=['Low (<0.7)', 'Medium (0.7-0.8)', 'High (0.8-0.9)', 'Very High (≥0.9)'])
print(confidence_levels.value_counts())

print("\nTop political preferences (Enhanced):")
top_prefs = df_enhanced['current_preference_enhanced'].value_counts().head(10)
for i, (pref, count) in enumerate(top_prefs.items(), 1):
    pct = (count / len(df_enhanced)) * 100
    print(f"{i:2d}. {pref}: {count} ({pct:.1f}%)")

# %%
# ENHANCED SEMANTIC ANALYSIS FOR MAIN PROBLEMS WITH ADVANCED VISUALIZATIONS
print("=== ANA PROBLEMLER İÇİN ANLAMSAL ANALİZ===")

# Install required packages for advanced visualizations
try:
    from wordcloud import WordCloud
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.patches as mpatches
    print("Advanced visualization libraries loaded successfully!")
except ImportError as e:
    print("Installing required packages for advanced visualizations...")
    import subprocess
    import sys
    packages = ['wordcloud', 'networkx', 'scikit-learn']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    from wordcloud import WordCloud
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.patches as mpatches

class EnhancedMainProblemSemanticAnalyzer:
    def __init__(self):
        # Enhanced semantic categories based on your sample data patterns
        self.problem_categories = {
        'Demokrasi/Seçim Sorunu': {
            'keywords': [
                'demokrasi', 'seçim', 'seçmen', 'oy', 'seçim sistemi', 'seçim hilesi',
                'seçim güvenliği', 'seçim adaleti', 'seçim manipülasyonu', 'seçim yasası', 
                'seçim sonuçları'
            ],
            'patterns': [
                r'demokr.*sorun', r'seçim.*problem', r'oy.*hilesi', r'seçmen.*manipülasyon'
            ]
        },
        'Adalet/Hukuk/Özgürlük Sorunu': {
            'keywords': [
                'adalet', 'adaletsizlik', 'hukuk', 'haksızlık', 'yargı', 'kısıtlama',
                'mahkeme', 'beka', 'Beka Sorunu', 'Bekaa', 'hak', 'haklar', 'hukuksuzluk',
                'kanun', 'hukuki', 'hak etmemek', 'hak gaspı', 'hukuk devleti', 'fikir özgürlüğü',
                'yargısızlık', 'hukuk sistemi', 'yargı bağımsızlığı'
            ],
            'patterns': [
                r'adalet.*problem', r'hukuk.*sorun', r'adaletsiz.*', r'hak.*gasp'
            ]
        },
        'Ekonomi/Geçim Sorunu': {
            'keywords': [
                'ekonomi', 'pahalılık', 'fakirlik', 'işsizlik', 'geçim', 'para', 
                'maaş', 'ücret', 'enflasyon', 'hayat pahalılığı', 'maddi', 'mali', 
                'gelir', 'yoksulluk', 'borç', 'kredi', 'fiyat', 'ekonomik', 'finansal',
                'bütçe', 'masraf', 'kriz', 'refah', 'satın alma', 'dolar', 'kur', 
                'vergi', 'zamlar'
            ],
            'patterns': [
                r'ekonomi.*kriz', r'pahalı.*hayat', r'geçim.*sıkıntı', r'hayat.*pahalı'
            ]
        },
        'Din/Sosyal Değer Sorunu': {
            'keywords': [
                'din', 'iman', 'inanç', 'dini', 'maneviyat', 'Siyasi İslam sömürgesi', 'Siyasal islam'
            ],
            'patterns': [
                r'din.*sorun', r'inanç.*problem', r'ahlak.*değer', r'manevi.*sorun'
            ]
        },
        'Siyasi Liderlik Sorunu': {
            'keywords': [
                'erdoğan', 'erdogan', 'YÖNETİM SİSTEMİ', 'Güç_zehirlenmesi', 'rte', 'reis',
                'recep tayyip', 'başkan', 'lider', 'liderlik', 'yönetim', 'iktidar', 
                'hükümet', 'siyasetçi', 'cumhurbaşkanı', 'başbakan', 'politikacı',
                'yönetici', 'otoriter', 'diktatör', 'despotizm', 'tek adam', 'otokratik'
            ],
            'patterns': [
                r'erdoğan.*problem', r'liderlik.*sorun', r'yönetim.*başarısız'
            ]
        },
        'Ahlak/Değer Sorunu': {
            'keywords': [
                'ahlak', 'ahlaksızlık', 'değer', 'değersizlik', 'kültür', 
                'kültürel yozlaşma', 'manevi', 'töre', 'gelenek', 'saygı', 'çürümüşlük',
                'saygısızlık', 'nezaket', 'edep', 'terbiye', 'vicdansız', 'etik',
                'ahlaki çöküş', 'değer kaybı', 'manevi çöküş'
            ],
            'patterns': [
                r'ahlak.*çöküş', r'değer.*kayb', r'manevi.*sorun'
            ]

        },
        'Kutuplaşma/Tarafçılık': {
                'keywords': ['kutuplaşma', 'tarafçılık', 'taraflı', 'bölünme', 'ayrım', 'partizanlık',
                            'ötekileştirme', 'nefret', 'düşmanlık', 'çatışma', 'gerginlik',
                            'birlik', 'beraberlik', 'hoşgörü', 'tolerans', 'ayrışma',
                            'kamplaşma', 'çelişki', 'zıtlaşma'],
                'patterns': [r'kutuplaş.*', r'taraf.*çılık', r'bölün.*', r'kamplar']
            },
            'Liyakatsizlik/Kayırmacılık': {
                'keywords': ['liyakat', 'liyakatsizlik', 'torpil', 'kayırmacılık', 'nepotizm','LİYAKAT HOCAM LİYAKAT','BABIŞKO ASUMANLAR ','babişko', 'asuman',
                            'adam kayırma', 'ehliyet', 'ehliyetsizlik', 'yetenek', 'yetkinlik',
                            'başarısızlık', 'beceriksizlik', 'kayırma', 'çıkar', 'akraba',
                            'torpilli', 'liyakatsiz', 'ehil olmayan'],
                'patterns': [r'liyakat.*yok', r'torpil.*sistem', r'adam.*kayır']
            },
            'Yolsuzluk/Rüşvet': {
                'keywords': ['yolsuzluk', 'rüşvet', 'menfaat', 'çıkar', 'zimmet', 'usulsuzlük',
                            'hırsızlık', 'dolandırıcılık', 'kara para', 'rüşvetçi', 'yolsuz',
                            'rant', 'rantiye', 'vurgun', 'çalmak', 'soygun'],
                'patterns': [r'yolsuz.*', r'rüşvet.*', r'çıkar.*elde']
            },
            'Terör/Güvenlik': {
                'keywords': ['terör', 'terörist', 'pkk', 'güvenlik', 'emniyet', 'asayiş',
                            'şiddet', 'saldırı', 'bomba', 'güvensizlik', 'korku', 'kaos',
                            'anarşi', 'suç', 'cinayet', 'savaş'],
                'patterns': [r'terör.*sorun', r'güvenlik.*problem', r'asayiş.*bozuk']
            },
            'Eğitim Sorunu': {
                'keywords': ['eğitim', 'okul','Zekâ', 'öğretmen', 'üniversite', 'öğrenci', 'ders','cehalet',
                            'bilgi', 'bilgisizlik', 'cahil', 'öğrenme', 'öğretim', 'öğrenim','eğitimsizlik'
                            
                            'bilgisizlik', 'cahillik', 'okuma', 'yazma', 'bilinç', 'bilinçsizlik','aptallık',
                            'öğretim', 'eğitim sistemi', 'müfredat', 'anaokulu'],
                'patterns': [r'eğitim.*yetersiz', r'okul.*problem', r'cahil.*']
            },
            'Medya/Propaganda': {
                'keywords': ['medya', 'basın', 'gazete', 'televizyon', 'haber', 'propaganda',
                            'yalan', 'dezenformasyon', 'manipülasyon', 'yayın', 'kanal',
                            'internet', 'sosyal medya', 'fake news', 'yanlış bilgi'],
                'patterns': [r'medya.*yalan', r'propaganda.*', r'haber.*yanlış']
            },
            'Demografik/Göç Sorunu': {
                'keywords': ['göç', 'göçmen', 'mülteci', 'suriyeli', 'afgan', 'yabancı','kürt',
                            'etnik', 'kültürel', 'sosyal', 'demografik', 'nüfus', 'işgal', 
                            'kalabalık', 'azınlık', 'çoğunluk', 'entegrasyon'],
                'patterns': [r'göç.*sorun', r'mülteci.*problem', r'yabancı.*çok']
            },
            'Çevre/Sağlık': {
                'keywords': ['çevre', 'doğa', 'kirlilik', 'hava', 'su', 'toprak', 'sağlık',
                            'hastalık', 'salgın', 'çevre kirliliği', 'ekoloji', 'iklim',
                            'global ısınma', 'sera gazı', 'atık'],
                'patterns': [r'çevre.*kirli', r'sağlık.*sorun', r'hava.*kirli']
            }
        }
        
        # Enhanced word mappings for better semantic grouping
        self.semantic_word_mappings = {
            'adalet': ['adalet', 'adaletsizlik', 'adaletsiz', 'haksızlık', 'hukuk', 'hukuksuzluk', 'kanun', 'yargı'],
            'ekonomi': ['ekonomi', 'ekonomik', 'pahalılık', 'fakirlik', 'işsizlik', 'geçim', 'para', 'kriz', 'enflasyon'],
            'erdoğan': ['erdoğan', 'erdogan', 'rte', 'reis', 'recep', 'tayyip', 'cumhurbaşkanı'],
            'ahlak': ['ahlak', 'ahlaksızlık', 'değer', 'değersizlik', 'etik', 'manevi'],
            'terör': ['terör', 'terörist', 'pkk', 'güvenlik', 'asayiş'],
            'eğitim': ['eğitim', 'okul', 'öğretmen', 'cahillik', 'bilinçsizlik', 'üniversite'],
            'yolsuzluk': ['yolsuzluk', 'rüşvet', 'çıkar', 'çıkarcılık', 'rant', 'vurgun'],
            'kutuplaşma': ['kutuplaşma', 'tarafçılık', 'bölünme', 'ayrışma', 'kamplaşma'],
            'liyakat': ['liyakat', 'liyakatsizlik', 'torpil', 'kayırmacılık', 'nepotizm'],
            'medya': ['medya', 'basın', 'propaganda', 'yalan', 'dezenformasyon'],
            'göç': ['göç', 'göçmen', 'mülteci', 'suriyeli', 'afgan', 'yabancı'],
            'sağlık': ['sağlık', 'hastalık', 'salgın', 'çevre', 'kirlilik']
        }
        
        # Store processed responses and statistics
        self.processed_responses = []
        self.word_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.response_statistics = {}
    
    def normalize_text(self, text):
        """Enhanced normalize text for better matching"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower().strip()
        # Handle multiple mentions - take the first significant mention
        sentences = text.split(',')
        if len(sentences) > 1:
            # Take the first non-trivial sentence
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence.split()) >= 2:  # At least 2 words
                    text = sentence
                    break
        
        # Remove extra spaces and punctuation, keep Turkish characters
        text = re.sub(r'[^\w\sğıişçöü]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_keywords(self, text, include_cooccurrence=True):
        """Enhanced extract relevant keywords from text with co-occurrence tracking"""
        normalized = self.normalize_text(text)
        text_words = normalized.split()
        
        # Enhanced stop words list
        stop_words = {'ve', 'ile', 'bir', 'bu', 'şu', 'o', 'da', 'de', 'ta', 'te', 
                     'ki', 'mi', 'mu', 'mı', 'mü', 'için', 'gibi', 'kadar', 'daha',
                     'en', 'çok', 'az', 'var', 'yok', 'olan', 'olduğu', 'olması',
                     'ama', 'fakat', 'ancak', 'lakin', 'çünkü', 'zira', 'hem', 'ya',
                     'her', 'hiç', 'hep', 'tüm', 'bütün', 'kendi', 'şey', 'hal'}
        
        meaningful_words = [word for word in text_words if len(word) > 2 and word not in stop_words]
        
        # Track co-occurrence for semantic network analysis
        if include_cooccurrence and len(meaningful_words) > 1:
            for idx, word1 in enumerate(meaningful_words):
                for word2 in meaningful_words[idx+1:]:
                    self.word_cooccurrence[word1][word2] += 1
                    self.word_cooccurrence[word2][word1] += 1
        
        return meaningful_words
    
    def categorize_problem(self, text):
        """Enhanced categorize a problem response with better scoring"""
        if pd.isna(text) or text == '':
            return {
                'category': 'Yanıt Yok',
                'confidence': 1.0,
                'matched_keywords': [],
                'original': text,
                'score': 0,
                'keywords_extracted': []
            }
        
        original_text = str(text)
        normalized = self.normalize_text(text)
        keywords = self.extract_keywords(text)
        
        # Check each category with enhanced scoring
        category_scores = {}
        matched_keywords = {}
        
        for category_name, category_data in self.problem_categories.items():
            score = 0
            keywords_found = []
            
            # Check keyword matches with weighted scoring
            for keyword in category_data['keywords']:
                if keyword in normalized:
                    # Give higher weight to longer, more specific keywords
                    weight = len(keyword.split()) * 2 if len(keyword.split()) > 1 else 1
                    score += weight
                    keywords_found.append(keyword)
            
            # Check pattern matches with high weight
            for pattern in category_data.get('patterns', []):
                if re.search(pattern, normalized, re.IGNORECASE):
                    score += 3  # Patterns get higher weight
                    keywords_found.append(f"pattern:{pattern}")
            
            # Bonus for multiple keyword matches in same category
            if len(keywords_found) > 1:
                score += len(keywords_found) * 0.5
            
            if score > 0:
                category_scores[category_name] = score
                matched_keywords[category_name] = keywords_found
        
        # Determine best category with enhanced logic
        if not category_scores:
            return {
                'category': 'Diğer/Kategorize Edilmemiş',
                'confidence': 0.0,
                'matched_keywords': [],
                'original': original_text,
                'score': 0,
                'keywords_extracted': keywords
            }
        
        best_category = max(category_scores.keys(), key=lambda x: category_scores[x])
        max_score = category_scores[best_category]
        
        # Enhanced confidence calculation
        text_length_factor = min(len(normalized.split()) / 5, 1.0)
        keyword_density = len(matched_keywords[best_category]) / max(len(keywords), 1)
        confidence = min((max_score / 10) * text_length_factor * (1 + keyword_density), 1.0)
        
        return {
            'category': best_category,
            'confidence': confidence,
            'matched_keywords': matched_keywords[best_category],
            'original': original_text,
            'score': max_score,
            'keywords_extracted': keywords
        }
    
    def analyze_all_problems(self, problems):
        """Enhanced analyze all main problem responses with statistics"""
        results = []
        word_counts = defaultdict(int)
        method_category_counts = defaultdict(int)
        
        for problem in problems:
            analysis = self.categorize_problem(problem)
            results.append(analysis)
            self.processed_responses.append(analysis)
            
            # Update statistics
            method_category_counts[analysis['category']] += 1
            for keyword in analysis['keywords_extracted']:
                word_counts[keyword] += 1
        
        # Store statistics
        self.response_statistics = {
            'total_responses': len(problems),
            'categories_found': len(method_category_counts),
            'word_counts': dict(word_counts),
            'category_distribution': dict(method_category_counts),
            'avg_confidence': np.mean([r['confidence'] for r in results])
        }
        
        return results
    
    def get_word_frequency(self, problems, min_length=3, top_n=100):
        """Enhanced get word frequency analysis with filtering"""
        word_counts = defaultdict(int)
        
        for problem in problems:
            if pd.notna(problem):
                keywords = self.extract_keywords(problem)
                for keyword_item in keywords:
                    if len(keyword_item) >= min_length:
                        word_counts[keyword_item] += 1
        
        # Filter out very rare words (appear only once) unless specifically requested
        if top_n <= 50:
            word_counts = {k: v for k, v in word_counts.items() if v > 1}
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_n]
    
    def create_enhanced_semantic_groups(self, word_freq, min_group_size=5):
        """Enhanced semantic word grouping with better logic"""
        semantic_groups = {}
        
        # Initialize groups from mappings
        for main_word in self.semantic_word_mappings:
            semantic_groups[main_word] = 0
        
        # Count words in predefined groups
        ungrouped_words = []
        for word_text, word_count in word_freq:
            grouped = False
            for main_word, variations in self.semantic_word_mappings.items():
                if word_text in variations or any(word_text in var or var in word_text for var in variations):
                    semantic_groups[main_word] += word_count
                    grouped = True
                    break
            
            if not grouped and word_count >= min_group_size:
                ungrouped_words.append((word_text, word_count))
        
        # Add significant ungrouped words
        for word_item, count_item in ungrouped_words:
            semantic_groups[word_item] = count_item
        
        # Remove empty groups
        semantic_groups = {k: v for k, v in semantic_groups.items() if v > 0}
        
        return semantic_groups
    
    def create_word_cloud(self, word_freq, width=800, height=400, max_words=100):
        """Create advanced word cloud visualization"""
        if not word_freq:
            return None
            
        # Prepare word frequency dictionary
        word_dict = dict(word_freq[:max_words])
        
        # Create word cloud with Turkish font support
        word_cloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            max_words=max_words,
            colormap='viridis',
            relative_scaling=0.5,
            random_state=42,
            prefer_horizontal=0.7
        ).generate_from_frequencies(word_dict)
        
        return word_cloud
    
    def create_semantic_network(self, min_cooccurrence=3, top_nodes=30):
        """Create semantic network from word co-occurrences"""
        # Create network graph
        G = nx.Graph()
        
        # Get top words by frequency
        all_words = set()
        for word1, connections in self.word_cooccurrence.items():
            all_words.add(word1)
            for word2 in connections.keys():
                all_words.add(word2)
        
        # Filter to top words based on overall frequency
        word_freq_dict = self.response_statistics.get('word_counts', {})
        network_top_words = sorted(all_words, key=lambda x: word_freq_dict.get(x, 0), reverse=True)[:top_nodes]
        
        # Add edges based on co-occurrence
        for word1 in network_top_words:
            if word1 in self.word_cooccurrence:
                for word2, cooccur_count in self.word_cooccurrence[word1].items():
                    if word2 in network_top_words and cooccur_count >= min_cooccurrence:
                        G.add_edge(word1, word2, weight=cooccur_count)
        
        return G
    
    def get_comprehensive_statistics(self):
        """Get comprehensive statistics about the analysis"""
        if not self.response_statistics:
            return {}
        
        stats = self.response_statistics.copy()
        
        # Add more detailed statistics
        conf_scores = [r['confidence'] for r in self.processed_responses]
        stats['confidence_stats'] = {
            'mean': np.mean(conf_scores),
            'median': np.median(conf_scores),
            'std': np.std(conf_scores),
            'min': np.min(conf_scores),
            'max': np.max(conf_scores)
        }
        
        # Category confidence by category
        category_confidence = defaultdict(list)
        for resp_item in self.processed_responses:
            category_confidence[resp_item['category']].append(resp_item['confidence'])
        
        stats['category_confidence'] = {
            cat: {
                'mean': np.mean(conf_list),
                'count': len(conf_list)
            } for cat, conf_list in category_confidence.items()
        }
        
        return stats

# Initialize the enhanced analyzer
enhanced_problem_analyzer = EnhancedMainProblemSemanticAnalyzer()
print("Enhanced Main Problem Semantic Analyzer initialized!")

# %%
# Enhanced analysis of main problems
print("=== ENHANCED ANALYSIS OF MAIN PROBLEMS ===")

# Get main problems data
main_problems = df['main_problem'].dropna().tolist()
print(f"Analyzing {len(main_problems)} main problem responses...")


# Perform enhanced semantic analysis
enhanced_problem_results = enhanced_problem_analyzer.analyze_all_problems(main_problems)
enhanced_problem_df = pd.DataFrame(enhanced_problem_results)

# Get enhanced word frequency analysis
enhanced_word_frequency = enhanced_problem_analyzer.get_word_frequency(main_problems, min_length=3, top_n=150)
enhanced_semantic_groups = enhanced_problem_analyzer.create_enhanced_semantic_groups(enhanced_word_frequency)

# Get comprehensive statistics
comprehensive_stats = enhanced_problem_analyzer.get_comprehensive_statistics()

# Print enhanced analysis results
print("\n🔍 ENHANCED SEMANTIC ANALYSIS RESULTS")
print("=" * 60)
print(f"📊 Total responses analyzed: {comprehensive_stats['total_responses']}")
print(f"📂 Categories identified: {comprehensive_stats['categories_found']}")
print(f"🎯 Average confidence: {comprehensive_stats['avg_confidence']:.3f}")
print(f"📝 Unique words found: {len(comprehensive_stats['word_counts'])}")

print("\n🏆 TOP SEMANTIC CATEGORIES:")
category_counts = enhanced_problem_df['category'].value_counts()
for i, (category, count) in enumerate(category_counts.head(10).items(), 1):
    pct = (count / len(enhanced_problem_df)) * 100
    avg_conf = comprehensive_stats['category_confidence'][category]['mean']
    print(f"{i:2d}. {category}: {count} ({pct:.1f}%) - Güven: {avg_conf:.3f}")

print("\n💡 TOP SEMANTIC WORD GROUPS:")
semantic_sorted = sorted(enhanced_semantic_groups.items(), key=lambda x: x[1], reverse=True)[:15]
for i, (word, count) in enumerate(semantic_sorted, 1):
    print(f"{i:2d}. {word}: {count}")

# show answerswith low confidence
low_confidence_responses = enhanced_problem_df[enhanced_problem_df['confidence'] < 0.5]
if not low_confidence_responses.empty:  
    print("\n⚠️ RESPONSES WITH LOW CONFIDENCE (below 0.5):")
    for i, row in low_confidence_responses.iterrows():
        print(f"{i+1:2d}. {row['original']} - Kategori: {row['category']} - Güven: {row['confidence']:.3f}")





# %%
# Create advanced visualizations with word clouds and semantic networks
print("\n=== CREATING ADVANCED VISUALIZATIONS ===")

# Create comprehensive visualization figure
fig = plt.figure(figsize=(24, 30))

# 1. Enhanced Categories Pie Chart
ax1 = plt.subplot(5, 3, 1)
top_categories = enhanced_problem_df['category'].value_counts().head(8)
other_count = enhanced_problem_df['category'].value_counts().iloc[8:].sum()
if other_count > 0:
    pie_data = list(top_categories.values) + [other_count]
    pie_labels = list(top_categories.index) + ['Diğer']
else:
    pie_data = top_categories.values
    pie_labels = top_categories.index

colors = plt.cm.viridis(np.linspace(0, 1, len(pie_data)))
wedges, texts, autotexts = ax1.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', 
                                  startangle=90, colors=colors)
ax1.set_title('Ana Sorunlar - Enhanced Semantik Kategoriler', fontsize=12, fontweight='bold')

# Improve label readability
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(8)

# 2. Word Cloud Visualization
ax2 = plt.subplot(5, 3, 2)
wordcloud = enhanced_problem_analyzer.create_word_cloud(enhanced_word_frequency, max_words=100)
if wordcloud:
    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.axis('off')
    ax2.set_title('Ana Sorunlar Kelime Bulutu', fontsize=12, fontweight='bold')

# 3. Semantic Network Visualization
ax3 = plt.subplot(5, 3, 3)
semantic_network = enhanced_problem_analyzer.create_semantic_network(min_cooccurrence=3, top_nodes=25)
if semantic_network and len(semantic_network.nodes()) > 0:
    pos = nx.spring_layout(semantic_network, k=3, iterations=50)
    
    # Draw network with enhanced styling
    edge_weights = [semantic_network[u][v]['weight'] for u, v in semantic_network.edges()]
    node_sizes = [enhanced_problem_analyzer.response_statistics['word_counts'].get(node, 10) * 50 
                  for node in semantic_network.nodes()]
    
    nx.draw_networkx_nodes(semantic_network, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.7, ax=ax3)
    nx.draw_networkx_edges(semantic_network, pos, width=[w*0.3 for w in edge_weights], 
                          alpha=0.5, edge_color='gray', ax=ax3)
    nx.draw_networkx_labels(semantic_network, pos, font_size=8, ax=ax3)
    
    ax3.set_title('Kelime İlişki Ağı (Semantic Network)', fontsize=12, fontweight='bold')
    ax3.axis('off')


# 4. Enhanced Semantic Groups Bar Chart
ax4 = plt.subplot(5, 3, 4)
semantic_sorted = sorted(enhanced_semantic_groups.items(), key=lambda x: x[1], reverse=True)[:15]
words, counts = zip(*semantic_sorted)
bars = ax4.barh(range(len(words)), counts, color='skyblue')
ax4.set_yticks(range(len(words)))
ax4.set_yticklabels(words)
ax4.set_xlabel('Toplam Sayı')
ax4.set_title('Enhanced Semantik Kelime Grupları', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add value labels
for i, (bar, count) in enumerate(zip(bars, counts)):
    ax4.text(count + max(counts)*0.01, i, str(count), va='center', fontsize=9)

# 5. Confidence Distribution with Categories
ax5 = plt.subplot(5, 3, 5)
confidence_by_category = []
category_names = []
for category_item in category_counts.head(8).index:
    confidences = [r['confidence'] for r in enhanced_problem_results if r['category'] == category_item]
    confidence_by_category.append(confidences)
    category_names.append(category_item[:15] + '...' if len(category_item) > 15 else category_item)

box_plot = ax5.boxplot(confidence_by_category, labels=category_names, patch_artist=True)
for patch in box_plot['boxes']:
    patch.set_facecolor('lightgreen')
ax5.set_title('Kategorilere Göre Güven Seviyeleri', fontsize=12, fontweight='bold')
ax5.set_ylabel('Güven Seviyesi')
ax5.tick_params(axis='x', rotation=45)

# 6. Top Words Original Frequency
ax6 = plt.subplot(5, 3, 6)
top_words = enhanced_word_frequency[:20]
words_orig, counts_orig = zip(*top_words)
bars = ax6.barh(range(len(words_orig)), counts_orig, color='lightcoral')
ax6.set_yticks(range(len(words_orig)))
ax6.set_yticklabels(words_orig)
ax6.set_xlabel('Frekans')
ax6.set_title('En Sık Kullanılan Kelimeler', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. Education vs Problems Heatmap
ax7 = plt.subplot(5, 3, 7)
temp_df = df.copy()
temp_df['problem_category'] = enhanced_problem_df['category']
prob_education = pd.crosstab(temp_df['education'], temp_df['problem_category'])
prob_education_pct = prob_education.div(prob_education.sum(axis=1), axis=0) * 100
top_6_cats = enhanced_problem_df['category'].value_counts().head(6).index
prob_education_top = prob_education_pct[top_6_cats]

sns.heatmap(prob_education_top, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax7, 
            cbar_kws={'label': 'Yüzde (%)'})
ax7.set_title('Eğitim vs Ana Sorunlar', fontsize=12, fontweight='bold')
ax7.set_xlabel('Sorun Kategorisi')
ax7.set_ylabel('Eğitim Seviyesi')
ax7.tick_params(axis='x', rotation=45)

# 8. Age vs Problems Heatmap
ax8 = plt.subplot(5, 3, 8)
prob_age = pd.crosstab(temp_df['age'], temp_df['problem_category'])
prob_age_pct = prob_age.div(prob_age.sum(axis=1), axis=0) * 100
prob_age_top = prob_age_pct[top_6_cats]
sns.heatmap(prob_age_top, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax8,
            cbar_kws={'label': 'Yüzde (%)'})
ax8.set_title('Yaş vs Ana Sorunlar', fontsize=12, fontweight='bold')
ax8.set_xlabel('Sorun Kategorisi')
ax8.set_ylabel('Yaş Grubu')
ax8.tick_params(axis='x', rotation=45)

# 9. Income vs Problems
ax9 = plt.subplot(5, 3, 9)
prob_income = pd.crosstab(temp_df['income'], temp_df['problem_category'])
prob_income_pct = prob_income.div(prob_income.sum(axis=1), axis=0) * 100
prob_income_top = prob_income_pct[top_6_cats]
sns.heatmap(prob_income_top, annot=True, fmt='.1f', cmap='plasma', ax=ax9,
            cbar_kws={'label': 'Yüzde (%)'})
ax9.set_title('Gelir vs Ana Sorunlar', fontsize=12, fontweight='bold')
ax9.set_xlabel('Sorun Kategorisi')
ax9.set_ylabel('Gelir Seviyesi')
ax9.tick_params(axis='x', rotation=45)

# 10. Enhanced Category Distribution
ax10 = plt.subplot(5, 3, 10)
category_counts_all = enhanced_problem_df['category'].value_counts()
bars = ax10.bar(range(len(category_counts_all)), category_counts_all.values, color='purple', alpha=0.7)
ax10.set_xticks(range(len(category_counts_all)))
ax10.set_xticklabels([cat[:10] + '...' if len(cat) > 10 else cat for cat in category_counts_all.index], 
                    rotation=45, ha='right')
ax10.set_title('Tüm Kategoriler Dağılımı', fontsize=12, fontweight='bold')
ax10.set_ylabel('Yanıt Sayısı')
ax10.grid(True, alpha=0.3)

# 11. Score vs Confidence Scatter
ax11 = plt.subplot(5, 3, 11)
scores = [r['score'] for r in enhanced_problem_results]
confidences = [r['confidence'] for r in enhanced_problem_results]
scatter = ax11.scatter(scores, confidences, alpha=0.6, c=confidences, cmap='viridis')
ax11.set_xlabel('Eşleşme Skoru')
ax11.set_ylabel('Güven Seviyesi')
ax11.set_title('Skor vs Güven İlişkisi', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax11, label='Güven Seviyesi')

# 12. Keywords per Response Distribution
ax12 = plt.subplot(5, 3, 12)
keywords_per_response = [len(r['keywords_extracted']) for r in enhanced_problem_results]
ax12.hist(keywords_per_response, bins=20, color='orange', alpha=0.7)
ax12.set_xlabel('Yanıt Başına Anahtar Kelime Sayısı')
ax12.set_ylabel('Yanıt Sayısı')
ax12.set_title('Anahtar Kelime Dağılımı', fontsize=12, fontweight='bold')
ax12.grid(True, alpha=0.3)

# 13. Word Frequency Distribution
ax13 = plt.subplot(5, 3, 13)
word_freqs = [count for word, count in enhanced_word_frequency[:50]]
ax13.plot(range(len(word_freqs)), word_freqs, marker='o', linestyle='-', color='red')
ax13.set_xlabel('Kelime Sırası (Frekansa Göre)')
ax13.set_ylabel('Frekans')
ax13.set_title('Kelime Frekans Dağılımı (Zipf Yasası)', fontsize=12, fontweight='bold')
ax13.set_yscale('log')
ax13.grid(True, alpha=0.3)

# 14. Response Length vs Confidence
ax14 = plt.subplot(5, 3, 14)
response_lengths = [len(r['original'].split()) if pd.notna(r['original']) else 0 for r in enhanced_problem_results]
ax14.scatter(response_lengths, confidences, alpha=0.6, color='green')
ax14.set_xlabel('Yanıt Uzunluğu (Kelime Sayısı)')
ax14.set_ylabel('Güven Seviyesi')
ax14.set_title('Yanıt Uzunluğu vs Güven', fontsize=12, fontweight='bold')
ax14.grid(True, alpha=0.3)

# 15. Category Confidence Summary
ax15 = plt.subplot(5, 3, 15)
cat_conf_means = [comprehensive_stats['category_confidence'][cat_item]['mean'] 
                  for cat_item in category_counts.head(10).index]
cat_names_short = [cat_item[:15] + '...' if len(cat_item) > 15 else cat_item for cat_item in category_counts.head(10).index]
bars = ax15.bar(range(len(cat_conf_means)), cat_conf_means, color='teal', alpha=0.7)
ax15.set_xticks(range(len(cat_conf_means)))
ax15.set_xticklabels(cat_names_short, rotation=45, ha='right')
ax15.set_ylabel('Ortalama Güven')
ax15.set_title('Kategorilere Göre Ortalama Güven', fontsize=12, fontweight='bold')
ax15.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("🎉 Enhanced visualizations completed!")

# %%
# Export enhanced results to Excel and JSON
print("\n=== EXPORTING ENHANCED RESULTS ===")

# Prepare comprehensive export data
export_data = {
    'enhanced_problem_analysis': enhanced_problem_df,
    'word_frequency': pd.DataFrame(enhanced_word_frequency, columns=['word', 'frequency']),
    'semantic_groups': pd.DataFrame(list(enhanced_semantic_groups.items()), columns=['group', 'count']),
    'statistics': pd.DataFrame([comprehensive_stats['confidence_stats']]),
    'category_stats': pd.DataFrame([
        {
            'category': cat,
            'count': comprehensive_stats['category_distribution'][cat],
            'avg_confidence': comprehensive_stats['category_confidence'][cat]['mean']
        }
        for cat in comprehensive_stats['category_distribution'].keys()
    ])
}

# Export to Excel with multiple sheets
with pd.ExcelWriter('/Users/ysk/Desktop/Projects/AkademiklinkAnket/enhanced_main_problems_analysis.xlsx') as writer:
    for sheet_name, data in export_data.items():
        if isinstance(data, pd.DataFrame):
            data.to_excel(writer, sheet_name=sheet_name, index=False)

# Export comprehensive statistics to JSON
with open('/Users/ysk/Desktop/Projects/AkademiklinkAnket/enhanced_problems_statistics.json', 'w', encoding='utf-8') as f:
    # Convert numpy types to native Python types for JSON serialization
    json_stats = {}
    for key, value in comprehensive_stats.items():
        if isinstance(value, dict):
            json_stats[key] = {}
            for k, v in value.items():
                if isinstance(v, (np.integer, np.floating)):
                    json_stats[key][k] = float(v)
                elif isinstance(v, dict):
                    json_stats[key][k] = {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv 
                                        for kk, vv in v.items()}
                else:
                    json_stats[key][k] = v
        else:
            json_stats[key] = float(value) if isinstance(value, (np.integer, np.floating)) else value
    
    json.dump(json_stats, f, ensure_ascii=False, indent=2)

print("✅ Enhanced analysis results exported to:")
print("📊 Excel: enhanced_main_problems_analysis.xlsx")
print("📄 JSON: enhanced_problems_statistics.json")

# %%
# Create detailed semantic analysis report
print("\n=== DETAILED SEMANTIC ANALYSIS REPORT ===")
print("=" * 70)

print("\n📊 EXECUTIVE SUMMARY")
print(f"   • Total Responses: {comprehensive_stats['total_responses']}")
print(f"   • Categorized: {len([r for r in enhanced_problem_results if r['category'] != 'Diğer/Kategorize Edilmemiş'])}")
print(f"   • Uncategorized: {len([r for r in enhanced_problem_results if r['category'] == 'Diğer/Kategorize Edilmemiş'])}")
print(f"   • Average Confidence: {comprehensive_stats['avg_confidence']:.3f}")
print(f"   • High Confidence (>0.7): {len([r for r in enhanced_problem_results if r['confidence'] > 0.7])}")

print("\n🏆 TOP 5 PROBLEM CATEGORIES:")
for i, (category, count) in enumerate(category_counts.head(5).items(), 1):
    pct = (count / len(enhanced_problem_df)) * 100
    avg_conf = comprehensive_stats['category_confidence'][category]['mean']
    print(f"   {i}. {category}")
    print(f"      Count: {count} ({pct:.1f}%)")
    print(f"      Avg Confidence: {avg_conf:.3f}")

print("\n🔗 SEMANTIC NETWORK INSIGHTS:")
if semantic_network:
    print(f"   • Network Nodes: {len(semantic_network.nodes())}")
    print(f"   • Network Edges: {len(semantic_network.edges())}")
    if len(semantic_network.nodes()) > 0:
        centrality = nx.degree_centrality(semantic_network)
        top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        print("   • Most Central Words:")
        for word, centrality_score in top_central:
            print(f"     - {word}: {centrality_score:.3f}")

print("\n💡 KEY INSIGHTS:")
economics_related = len([r for r in enhanced_problem_results if 'ekonomi' in r['category'].lower() or 'geçim' in r['category'].lower()])
justice_related = len([r for r in enhanced_problem_results if 'adalet' in r['category'].lower() or 'hukuk' in r['category'].lower()])
leadership_related = len([r for r in enhanced_problem_results if 'liderlik' in r['category'].lower() or 'erdoğan' in str(r['matched_keywords']).lower()])

print(f"   • Economic concerns: {economics_related} responses ({economics_related/len(enhanced_problem_results)*100:.1f}%)")
print(f"   • Justice/Law concerns: {justice_related} responses ({justice_related/len(enhanced_problem_results)*100:.1f}%)")
print(f"   • Leadership concerns: {leadership_related} responses ({leadership_related/len(enhanced_problem_results)*100:.1f}%)")

# Sample problematic responses (low confidence)
low_confidence_responses = [r for r in enhanced_problem_results if r['confidence'] < 0.3 and r['category'] != 'Yanıt Yok']
if low_confidence_responses:
    print("\n⚠️  EXAMPLES OF DIFFICULT TO CATEGORIZE RESPONSES:")
    for i, response in enumerate(low_confidence_responses[:5], 1):
        print(f"   {i}. \"{response['original'][:50]}...\"")
        print(f"      Category: {response['category']} (Confidence: {response['confidence']:.3f})")

print("\n🎯 ANALYSIS COMPLETE!")
print("=" * 70)

# %%

# %%
# Required packages are installed in the virtual environment

# %%
# --- 2. DATA LOADING AND PREPARATION ---
print("=== COMPREHENSIVE TURKISH POLITICAL SURVEY - ADVANCED ANALYSIS v2 ===")

# Load data directly from the specified path
FILE_PATH = '/Users/ysk/Desktop/Projects/AkademiklinkAnket/Akademiklink gündem 2 (Yanıtlar).xlsx'
try:
    df = pd.read_excel(FILE_PATH)
    print(f"✅ Excel file loaded successfully from: {FILE_PATH}")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ ERROR: The file was not found at the specified path: {FILE_PATH}")
    print("Please ensure the file path is correct and the file is accessible.")
    exit() # Exit the script if the file isn't found

# Clean column names for consistency
df.columns = [
    'timestamp', 'age', 'gender', 'city', 'income', 'education', 'marital_status',
    'last_vote', 'current_preference', 'politics_important_marriage', 'attention_check',
    'social_media_usage', 'blocked_friends', 'unfollowed_influencers', 'main_problem',
    'istanbul_election_cancel', 'friendship_akp', 'friendship_chp', 'imamoglu_prison',
    'ozdag_prison', 'demirtas_prison', 'vote_opposite_party', 'support_lawbreaking',
    'early_election', 'end_presidential_system', 'akp_chp_coalition', 'boycotts_effective',
    'new_constitution', 'solution_process', 'akp_description', 'chp_description',
    'imamoglu_statement_read'
]
df_clean = df.copy()

# Apply cleaning and mapping functions (from your original script)
vote_mapping = {'Recep Tayyip Erdoğan': 'Erdoğan (AKP)', 'Kemal Kılıçdaroğlu': 'Kılıçdaroğlu (CHP)', 'Sinan Oğan': 'Oğan (ATA)', 'Oy kullanmadım': 'Did not vote'}
df_clean['last_vote'] = df_clean['last_vote'].map(vote_mapping).fillna(df_clean['last_vote'])

# Convert all attitude columns to numeric, coercing errors
attitude_columns = [
    'istanbul_election_cancel', 'friendship_akp', 'friendship_chp', 'imamoglu_prison',
    'ozdag_prison', 'demirtas_prison', 'vote_opposite_party', 'support_lawbreaking',
    'early_election', 'end_presidential_system', 'akp_chp_coalition', 'boycotts_effective',
    'new_constitution', 'solution_process'
]
for col in attitude_columns:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

print("\n✅ Data preparation and initial processing complete.")


# %%
# --- 1. VERİYİ YÜKLE ve SÜTUNLARI TEMİZLE ---
df = pd.read_excel('/Users/ysk/Desktop/Projects/AkademiklinkAnket/Akademiklink gündem 2 (Yanıtlar).xlsx')

# Sütun adlarını düzenli ve kullanılabilir hale getir
df.columns = [
    'timestamp', 'age', 'gender', 'city', 'income', 'education', 'marital_status',
    'last_vote', 'current_preference', 'politics_important_marriage', 'attention_check',
    'social_media_usage', 'blocked_friends', 'unfollowed_influencers', 'main_problem',
    'istanbul_election_cancel', 'friendship_akp', 'friendship_chp', 'imamoglu_prison',
    'ozdag_prison', 'demirtas_prison', 'vote_opposite_party', 'support_lawbreaking',
    'early_election', 'end_presidential_system', 'akp_chp_coalition', 'boycotts_effective',
    'new_constitution', 'solution_process', 'akp_description', 'chp_description',
    'imamoglu_statement_read'
]

# --- 2. VOTER GROUP'u tanımla (current_preference'dan türet) ---
def analyze_preference(text):
    text = str(text).lower()
    if any(s in text for s in ['erdoğan', 'reis', 'rte']): return 'Erdoğan Supporter'
    if any(s in text for s in ['imamoğlu', 'eko']): return 'İmamoğlu Supporter'
    if 'yavaş' in text: return 'Yavaş Supporter'
    if 'ince' in text: return 'İnce Supporter'
    if 'özdağ' in text: return 'Özdağ Supporter'
    if 'karşısında' in text or 'hariç' in text: return 'Strongest Opposition'
    if 'kararsız' in text or 'bilmiyorum' in text: return 'Undecided'
    if 'kullanmam' in text or 'kimseye' in text: return 'Will Not Vote'
    return 'Other'

df['voter_group'] = df['current_preference'].apply(analyze_preference)

# --- 3. DUYGU ANALİZİ UYGULA ---
def get_sentiment(text):
    try:
        score = TextBlob(str(text)).sentiment.polarity
        return 0 if abs(score) < 0.2 else score  # 0.2 altında ise nötr say
    except (AttributeError, TypeError, ValueError):
        return 0

df['akp_sentiment'] = df['akp_description'].apply(get_sentiment)
df['chp_sentiment'] = df['chp_description'].apply(get_sentiment)

# --- 4. Temizlenmiş dataframe oluştur ---
df_clean = df.dropna(subset=[
    'voter_group', 'chp_description', 'akp_description', 'chp_sentiment', 'akp_sentiment'
])

# --- 5. Erdoğan destekçileri içinde CHP'ye pozitif bakanları yazdır ---
erdogan_chp_pozitif = df_clean[
    (df_clean['voter_group'] == 'Erdoğan Supporter') &
    (df_clean['chp_sentiment'] > 0)
]

print("\n📌 Erdoğan destekçisinin CHP'yi olumlu değerlendirdiği örnek yorumlar:\n")
print(erdogan_chp_pozitif[['chp_description', 'chp_sentiment']].head(10))

# --- 6. ALGILAMA MATRİSİ (Perception Matrix) ---
perception_df = df_clean.groupby('voter_group')[['akp_sentiment', 'chp_sentiment']].mean()
top_voter_groups = df_clean['voter_group'].value_counts().nlargest(4).index
perception_df = perception_df.loc[top_voter_groups]
perception_df = perception_df.rename(columns={
    'akp_sentiment': 'AKP Algısı',
    'chp_sentiment': 'CHP Algısı'
})

# --- 7. GRAFİĞİ ÇİZ ---
perception_df.plot(kind='bar', figsize=(14, 8), colormap='coolwarm_r')
plt.title("Algı Matrisi: Seçmen Grupları Ana Partileri Nasıl Görüyor?", fontsize=18, fontweight='bold')
plt.ylabel("Ortalama Duygu (Negatif → Pozitif)", fontsize=12)
plt.xlabel("Seçmen Grubu", fontsize=12)
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.legend(title="Tanımlanan Parti")
plt.tight_layout()
plt.show()


# %%
# --- 5. COMPARATIVE SEMANTIC ANALYSIS: AKP vs. CHP DESCRIPTIONS ---
print("\n=== 5. COMPARATIVE SEMANTIC ANALYSIS: AKP vs. CHP ===")

class AdvancedPartyAnalyzer:
    """Performs advanced NLP tasks on party descriptions."""
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('turkish') + ['bir', 'parti', 'partisi', 've', 'çok', 'ama', 'bence', 'yani', 'şey', 'chp', 'akp'])
        self.vectorizer = TfidfVectorizer(max_df=0.9, min_df=3, stop_words=list(self.stop_words), ngram_range=(1, 2))

    def get_sentiment(self, text):
        return TextBlob(str(text)).sentiment.polarity

    def discover_topics(self, text_data, n_topics=4):
        """Uses NMF to find semantic topics in the descriptions."""
        if text_data.dropna().empty:
            return {}, None
        topic_tfidf_matrix = self.vectorizer.fit_transform(text_data.dropna())
        nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=500)
        nmf_model.fit(topic_tfidf_matrix)

        topic_feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(nmf_model.components_):
            topic_top_words = [topic_feature_names[i] for i in topic.argsort()[:-8 - 1:-1]]
            topics[f"Topic {topic_idx+1}"] = ", ".join(topic_top_words)
        return topics
        
    def get_full_analysis(self, series):
        df_analysis = pd.DataFrame(index=series.index)
        df_analysis['description'] = series
        df_analysis['sentiment'] = series.apply(self.get_sentiment)
        return df_analysis

# --- Run Analysis for Both Parties ---
analyzer = AdvancedPartyAnalyzer()
akp_analysis = analyzer.get_full_analysis(df_clean['akp_description'])
chp_analysis = analyzer.get_full_analysis(df_clean['chp_description'])

# Merge analysis back with main dataframe
df_clean['akp_sentiment'] = akp_analysis['sentiment']
df_clean['chp_sentiment'] = chp_analysis['sentiment']

# A. Side-by-Side Word Clouds
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
akp_text = ' '.join(df_clean['akp_description'].dropna().astype(str))
chp_text = ' '.join(df_clean['chp_description'].dropna().astype(str))

if akp_text:
    wc_akp = WordCloud(width=800, height=600, background_color='white', colormap='Reds', stopwords=analyzer.stop_words).generate(akp_text)
    axes[0].imshow(wc_akp, interpolation='bilinear')
    axes[0].set_title("Words Used to Describe AKP", fontsize=20, fontweight='bold')
    axes[0].axis('off')

if chp_text:
    wc_chp = WordCloud(width=800, height=600, background_color='white', colormap='Blues', stopwords=analyzer.stop_words).generate(chp_text)
    axes[1].imshow(wc_chp, interpolation='bilinear')
    axes[1].set_title("Words Used to Describe CHP", fontsize=20, fontweight='bold')
    axes[1].axis('off')

plt.show()


# - voter_group: Katılımcının oy verdiği/ait olduğu grup
# - akp_sentiment: AKP'ye yönelik duygu skoru
# - chp_sentiment: CHP'ye yönelik duygu skoru

# B. "Algı Matrisi": Farklı seçmen grupları her partiyi nasıl görüyor?

# Her seçmen grubunun AKP ve CHP'ye dair ortalama duygu skorlarını hesapla
algi_df = df_clean.groupby('voter_group')[['akp_sentiment', 'chp_sentiment']].mean()

# En büyük 4 seçmen grubunu seç (katılımcı sayısına göre)
en_buyuk_4_grup = df_clean['voter_group'].value_counts().nlargest(4).index

# Sadece bu grupları al ve sütun isimlerini Türkçeleştir
algi_df = algi_df.loc[en_buyuk_4_grup].rename(
    columns={
        'akp_sentiment': 'AKP Algısı',
        'chp_sentiment': 'CHP Algısı'
    }
)

# Grafik çizimi
algi_df.plot(
    kind='bar',              # Bar grafik
    figsize=(14, 8),         # Grafik boyutu
    colormap='coolwarm_r'    # Renk skalası (soğuk-sıcak ters)
)

plt.title("Algı Matrisi: Seçmen Gruplarının Partilere Bakışı", fontsize=18, fontweight='bold')
plt.ylabel("Ortalama Duygu (Negatiften Pozitife)")
plt.xlabel("Seçmen Grubu")
plt.xticks(rotation=45, ha='right')  # X eksenindeki etiketleri çapraz yap
plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Nötr çizgisi
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Algılanan Parti")
plt.tight_layout()
plt.show()

# C. Unique Keywords Analysis
combined_corpus = pd.concat([df_clean['akp_description'].dropna(), df_clean['chp_description'].dropna()])
tfidf = TfidfVectorizer(max_df=0.8, min_df=5, stop_words=list(analyzer.stop_words))
tfidf_matrix = tfidf.fit_transform(combined_corpus)
feature_names = np.array(tfidf.get_feature_names_out())

akp_indices = df_clean['akp_description'].dropna().index
chp_indices = df_clean['chp_description'].dropna().index
combined_indices = combined_corpus.index

akp_rows = [i for i, idx in enumerate(combined_indices) if idx in akp_indices]
chp_rows = [i for i, idx in enumerate(combined_indices) if idx in chp_indices]

mean_tfidf_akp = tfidf_matrix[akp_rows].mean(axis=0).A1
mean_tfidf_chp = tfidf_matrix[chp_rows].mean(axis=0).A1
diff = mean_tfidf_akp - mean_tfidf_chp

keyword_df = pd.DataFrame({'word': feature_names, 'score': diff}).sort_values('score', ascending=False)
top_keywords = pd.concat([keyword_df.head(10), keyword_df.tail(10)])

plt.figure(figsize=(12, 10))
colors = ['red' if x > 0 else 'blue' for x in top_keywords['score']]
sns.barplot(x='score', y='word', data=top_keywords, palette=colors)
plt.title("Most Characteristic Keywords: AKP vs. CHP Descriptions", fontsize=18, fontweight='bold')
plt.xlabel("More Characteristic of AKP Descriptions <---> More Characteristic of CHP Descriptions")
plt.ylabel("Keyword")
plt.show()


# %%
# --- 6. ANALYSIS OF MEDIA CONSUMPTION: `imamoglu_statement_read` ---
print("\n=== 6. ANALYSIS OF 'imamoglu_statement_read' ===")

# Filter for the main supporter groups
akp_followers = df_clean[df_clean['voter_group'] == 'Erdoğan Supporter']
chp_followers = df_clean[df_clean['voter_group'] == 'İmamoğlu Supporter']

# A. Statistical Breakdown
print("\n--- Did Erdoğan Supporters Read İmamoğlu's Statement? ---")
akp_counts = akp_followers['imamoglu_statement_read'].value_counts(normalize=True).mul(100)
print(akp_counts.round(1).astype(str) + '%')

print("\n--- Did İmamoğlu Supporters Read İmamoğlu's Statement? ---")
chp_counts = chp_followers['imamoglu_statement_read'].value_counts(normalize=True).mul(100)
print(chp_counts.round(1).astype(str) + '%')

# B. Comparative Visualization
comparison_df = df_clean[df_clean['voter_group'].isin(['Erdoğan Supporter', 'İmamoğlu Supporter'])]
crosstab_df = pd.crosstab(comparison_df['voter_group'], comparison_df['imamoglu_statement_read'], normalize='index').mul(100)

crosstab_df.plot(kind='bar', figsize=(10, 7), colormap='Set2')
plt.title("Read İmamoğlu's Statement? (AKP vs. CHP Supporters)", fontsize=18, fontweight='bold')
plt.ylabel("Percentage of Supporters (%)")
plt.xlabel("Voter Group")
plt.xticks(rotation=0)
plt.legend(title="Response")
plt.grid(axis='y', linestyle='--', alpha=0.7)
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%.1f%%', label_type='edge')
plt.show()

# %%
# --- 7. EXPORT AND CONCLUSION ---
print("\n=== 7. EXPORTING ADVANCED RESULTS ===")

# Create DataFrames from the new analyses
perception_export_df = perception_df.reset_index()
unique_keywords_export_df = top_keywords
media_consumption_export_df = crosstab_df.reset_index()

# Export to a new, more detailed Excel file
try:
    with pd.ExcelWriter('akademiklink_comparative_analysis.xlsx') as writer:
        perception_export_df.to_excel(writer, sheet_name='Perception_Matrix', index=False)
        unique_keywords_export_df.to_excel(writer, sheet_name='Unique_Party_Keywords', index=False)
        media_consumption_export_df.to_excel(writer, sheet_name='Imamoglu_Statement_Read', index=False)
        # You can add other previously generated DataFrames here as well
        # e.g., feature_importances_df.to_excel(writer, ...)

    print("\n✅ Comparative analysis results successfully exported to 'akademiklink_comparative_analysis.xlsx'")
except (PermissionError, FileNotFoundError, OSError) as e:
    print(f"\n❌ Error exporting file: {e}")

print("\n🎉 FULL ANALYSIS COMPLETE!")

#%%
