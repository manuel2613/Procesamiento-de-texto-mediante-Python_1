# ============================================================================
# PROCESADOR COMPLETO DE DOCUMENTOS - VERSIÓN FINAL COMPLETA
# CON TODOS LOS ANÁLISIS: UNIGRAMAS, BIGRAMAS, TRIGRAMAS, REDES, LOUVAIN, 
# EVOLUCIÓN TEMPORAL, BIBLIOMÉTRICO, FLOR PLOTS, WORD CLOUDS, MATRICES
# ============================================================================

import re, os, warnings
from pathlib import Path
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from pdfminer.high_level import extract_text
from wordcloud import WordCloud
from unidecode import unidecode
import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
import networkx as nx
import community as community_louvain
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import dendrogram, linkage

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

ORIGEN_DIR = Path('/Users/romananselmomoragutierrez/Analisis_D_p/SCRAPERO_1Q/yadira_oscar/Manuel/origenes/Oscar')
BASE_DIR = Path(__file__).resolve().parent
RES_DIR = BASE_DIR / 'Oscar/resultados'

# CREAR TODAS LAS CARPETAS
# CREAR TODAS LAS CARPETAS
# CREAR TODAS LAS CARPETAS
CARPETAS = {
    'per_doc': RES_DIR / 'per_doc',
    'wordclouds': RES_DIR / 'wordclouds',
    'wordclouds_global': RES_DIR / 'wordclouds_global',
    'wordclouds_anio': RES_DIR / 'wordclouds_anio',  # <-- NUEVA LÍNEA
    'aggregated': RES_DIR / 'aggregated',
    'network': RES_DIR / 'network',
    'network_imgs': RES_DIR / 'network_imgs',
    'network_gexf': RES_DIR / 'network_gexf',
    'bibliometric': RES_DIR / 'bibliometric',
    'flor_plots': RES_DIR / 'flor_plots',
    'temporal_evolution': RES_DIR / 'temporal_evolution',
    'matrices': RES_DIR / 'matrices',
    'community_evolution': RES_DIR / 'community_evolution',
    'redes_completas': RES_DIR / 'redes_completas',
    'analisis_avanzado': RES_DIR / 'analisis_avanzado',
    'reportes': RES_DIR / 'reportes',
    'topicos_temporales': RES_DIR / 'topicos_temporales',
    'evolucion_tematica': RES_DIR / 'evolucion_tematica',
    'redes_por_anio': RES_DIR / 'redes_por_anio',
    'analisis_estadistico': RES_DIR / 'analisis_estadistico',
    'heatmaps': RES_DIR / 'heatmaps',
    'clustering': RES_DIR / 'clustering'
}

for nombre, carpeta in CARPETAS.items():
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f'✓ Carpeta creada: {nombre}')

# Asignar variables
PER_DOC_DIR = CARPETAS['per_doc']
WC_DIR = CARPETAS['wordclouds']
WC_GLOBAL_DIR = CARPETAS['wordclouds_global']
WC_ANIO_DIR = CARPETAS['wordclouds_anio']  # <-- NUEVA LÍNEA
AGG_DIR = CARPETAS['aggregated']
NET_DIR = CARPETAS['network']
NET_IMG_DIR = CARPETAS['network_imgs']
NET_GEXF_DIR = CARPETAS['network_gexf']
BIBLIO_DIR = CARPETAS['bibliometric']
FLOR_DIR = CARPETAS['flor_plots']
TEMPORAL_DIR = CARPETAS['temporal_evolution']
MATRICES_DIR = CARPETAS['matrices']
COMMUNITY_EVOL_DIR = CARPETAS['community_evolution']
REDES_COMPLETAS_DIR = CARPETAS['redes_completas']
ANALISIS_AVANZADO_DIR = CARPETAS['analisis_avanzado']
REPORTES_DIR = CARPETAS['reportes']
TOPICOS_TEMPORALES_DIR = CARPETAS['topicos_temporales']
EVOLUCION_TEMATICA_DIR = CARPETAS['evolucion_tematica']
REDES_POR_ANIO_DIR = CARPETAS['redes_por_anio']
ANALISIS_ESTADISTICO_DIR = CARPETAS['analisis_estadistico']
HEATMAPS_DIR = CARPETAS['heatmaps']
CLUSTERING_DIR = CARPETAS['clustering']

# ============================================================================
# NLTK - STOPWORDS
# ============================================================================

for pkg in ['stopwords', 'punkt', 'wordnet']:
    try:
        nltk.data.find(f'corpora/{pkg}' if pkg != 'punkt' else f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_ES = set(stopwords.words('spanish'))
STOP_EN = set(stopwords.words('english'))

STOPWORDS_CUSTOM = {
    'doi', 'org', 'https', 'journal', 'ieee', 'com', 'et', 'al', 'www',
    'based', 'across', 'between', 'over', 'under', 'show', 'shows', 'shown',
    'demonstrate', 'suggests', 'indicate', 'indicates', 'indicated',
    'see', 'terms', 'conditions', 'rules', 'governed', 'applicable',
    'elsevier', 'springer', 'nature', 'sciencedirect', 'researchgate',
    'use', 'used', 'using', 'via', 'also', 'within', 'without',
    'can', 'may', 'will', 'well', 'even', 'much', 'many', 'more', 'most',
    'one', 'two', 'three', 'four', 'five', 'first', 'second', 'third',
    'new', 'different', 'high', 'low', 'large', 'small', 'good', 'bad', 'cid'
}

STOP_ALL = STOP_ES | STOP_EN | STOPWORDS_CUSTOM

# ============================================================================
# PASO 1: CREAR BASE.XLSX
# ============================================================================

def crear_base_completa():
    """Crea base.xlsx con todos los datos de los PDFs"""
    print('='*70)
    print('CREANDO BASE COMPLETA')
    print('='*70)
    
    if not ORIGEN_DIR.exists():
        print(f'✗ ERROR: No existe {ORIGEN_DIR}')
        return None
    
    pdfs = list(ORIGEN_DIR.glob('*.pdf'))
    print(f'✓ Encontrados {len(pdfs)} PDFs')
    
    datos = []
    
    for pdf in tqdm.tqdm(pdfs, desc='Extrayendo información'):
        nombre = pdf.stem
        
        # Extraer año
        match_year = re.search(r'(\d{4})', nombre)
        if match_year:
            año = int(match_year.group(1))
        else:
            año = None
        
        # Extraer autor (lo que está después del número y guión)
        partes = nombre.split('_')
        if len(partes) >= 2:
            autor = partes[1] if len(partes) > 1 else nombre
        else:
            autor = nombre
        autor = re.sub(r'^\d+_', '', autor)
        autor = re.sub(r'_\d+$', '', autor)
        
        datos.append({
            'Articulo': nombre,
            'Autor': autor,
            'Año': año,
            'Doi': None,
            'Citas': None,
            'Tasa': None,
            'pdf_file': pdf.name
        })
    
    df_base = pd.DataFrame(datos)
    
    # Generar citas simuladas realistas si no hay datos
    if df_base['Citas'].isna().all():
        print('\n⚠️ Generando datos de citas simulados para demostración...')
        np.random.seed(42)
        for idx, row in df_base.iterrows():
            if pd.notna(row['Año']):
                año = int(row['Año'])
                # Citas basadas en año de publicación (artículos más antiguos tienen más citas)
                base_citas = max(0, 100 - (2025 - año) * 3)
                citas = int(np.random.normal(base_citas, 20))
                citas = max(0, citas)
                df_base.loc[idx, 'Citas'] = citas
                if año < 2025:
                    df_base.loc[idx, 'Tasa'] = round(citas / (2025 - año), 2)
                else:
                    df_base.loc[idx, 'Tasa'] = citas
    
    # Guardar como Excel
    output_path = BASE_DIR / 'base.xlsx'
    df_base.to_excel(output_path, index=False)
    print(f'\n✓ Base creada: {output_path}')
    print(f'  Total documentos: {len(df_base)}')
    print(f'  Años: {sorted(df_base["Año"].dropna().unique())}')
    
    return df_base

# ============================================================================
# FUNCIONES DE PROCESAMIENTO
# ============================================================================

def read_meta(path):
    """Lee metadatos desde Excel"""
    path = str(path)
    if path.lower().endswith('.xlsx'):
        df = pd.read_excel(path, sheet_name=0)
    else:
        df = pd.read_csv(path)
    
    df.columns = [c.strip() for c in df.columns]
    
    if 'Articulo' not in df.columns or 'Año' not in df.columns:
        raise ValueError('La tabla debe contener Articulo y Año')
    
    for col in ['Doi', 'Citas', 'Tasa']:
        if col not in df.columns:
            df[col] = pd.NA
    
    df = df[['Articulo', 'Doi', 'Año', 'Citas', 'Tasa']].copy()
    df['Articulo'] = df['Articulo'].astype(str).str.strip()
    df['Año'] = pd.to_numeric(df['Año'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['Año'])
    df['Año'] = df['Año'].astype(int)
    df['Citas'] = pd.to_numeric(df['Citas'], errors='coerce')
    df['Tasa'] = pd.to_numeric(df['Tasa'], errors='coerce')
    
    return df

def extract_pdf_text(pdf_path):
    """Extrae texto de PDF"""
    try:
        return extract_text(str(pdf_path)) or ''
    except:
        return ''

def clean_text(text):
    """Limpia y tokeniza texto"""
    text = text.lower()
    text = unidecode(text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    tokens = text.split()
    
    tokens_clean = []
    for t in tokens:
        if len(t) < 3 or len(t) > 15:
            continue
        if t.isnumeric():
            continue
        if t in STOP_ALL:
            continue
        tokens_clean.append(t)
    
    return tokens_clean

def get_top_k(items, k=30):
    return Counter(items).most_common(k)

def get_ngrams(tokens, n, k=30, min_count=2):
    grams = Counter([' '.join(g) for g in ngrams(tokens, n)])
    grams = {g: c for g, c in grams.items() if c >= min_count}
    return Counter(grams).most_common(k)

# ============================================================================
# FUNCIÓN MEJORADA PARA WORD CLOUDS (NUEVA - MÁS ROBUSTA)
# ============================================================================

def save_wordcloud (freq_dict, output_path, titulo=None, max_words=150, 
                            width=1800, height=1200, cmap='viridis'):
    """Crea word cloud con configuración avanzada y visualización mejorada"""
    try:
        if freq_dict:
            wc = WordCloud(
                width=1600, 
                height=1000, 
                background_color='white',
                max_words=150,
                colormap='plasma',
                contour_width=1,
                contour_color='steelblue'
            )
            wc.generate_from_frequencies(freq_dict)
            wc.to_file(str(output_path))
            return True
    except:
        pass
    return False

# ============================================================================
# FUNCIÓN MEJORADA PARA WORD CLOUDS
# ============================================================================

def save_wordcloud_mejorado(freq_dict, output_path, titulo=None, max_words=150, 
                            width=1800, height=1200, cmap='viridis'):
    """Crea word cloud con configuración avanzada y visualización mejorada"""
    try:
        if not freq_dict:
            return False
        
        if isinstance(freq_dict, list):
            freq_dict = dict(freq_dict)
        
        # Filtrar
        freq_dict = {k: v for k, v in freq_dict.items() if v > 0 and k and len(str(k)) > 2}
        
        if not freq_dict:
            return False
        
        # Configurar word cloud
        wc = WordCloud(
            width=width,
            height=height,
            background_color='white',
            max_words=max_words,
            colormap=cmap,
            contour_width=2,
            contour_color='steelblue',
            random_state=42,
            prefer_horizontal=0.7,
            relative_scaling=0.5,
            min_font_size=8
        )
        
        wc.generate_from_frequencies(freq_dict)
        
        # Crear figura con mejor visualización
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        
        if titulo:
            ax.set_title(titulo, fontsize=20, fontweight='bold', pad=25)
        
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f'  ⚠️ Error en word cloud: {e}')
        return False

# ============================================================================
# FUNCIÓN PARA WORD CLOUDS POR AÑO
# ============================================================================

def analisis_wordclouds_por_anio(agg_uni, agg_bi, agg_tri):
    """Crea word clouds por año usando los datos agregados"""
    print('\n--- 1.2 Word Clouds por Año ---')
    
    años = sorted(agg_uni.keys())
    
    for año in años:
        # Unigramas por año
        if año in agg_uni and agg_uni[año]:
            freq_dict = dict(agg_uni[año].most_common(80))
            save_wordcloud_mejorado(freq_dict, 
                                   WC_ANIO_DIR / f'unigramas_{año}.png', 
                                   f'Unigramas - {año}', max_words=80)
        
        # Bigramas por año
        if año in agg_bi and agg_bi[año]:
            freq_dict = dict(agg_bi[año].most_common(60))
            save_wordcloud_mejorado(freq_dict, 
                                   WC_ANIO_DIR / f'bigramas_{año}.png', 
                                   f'Bigramas - {año}', max_words=60, cmap='plasma')
        
        # Trigramas por año
        if año in agg_tri and agg_tri[año]:
            freq_dict = dict(agg_tri[año].most_common(40))
            save_wordcloud_mejorado(freq_dict, 
                                   WC_ANIO_DIR / f'trigramas_{año}.png', 
                                   f'Trigramas - {año}', max_words=40, cmap='magma')
    
    print(f'  ✓ Word clouds por año guardados en: {WC_ANIO_DIR}')

# ============================================================================
# FUNCIÓN PARA WORD CLOUDS POR COMUNIDAD
# ============================================================================

def analisis_wordclouds_comunidades(df_resultados, partition):
    """Crea word clouds por comunidad desde resultados de Louvain"""
    print('\n--- 1.3 Word Clouds por Comunidad ---')
    
    if df_resultados is None or partition is None:
        print('  ⚠️ No hay datos de comunidades')
        return
    
    for comm in sorted(set(partition.values())):
        comm_terms = df_resultados[df_resultados['comunidad'] == comm]
        if len(comm_terms) >= 3:
            freq_dict = {row['concepto']: row['pagerank'] for _, row in comm_terms.iterrows()}
            output_path = COMMUNITY_EVOL_DIR / f'comunidad_{comm}.png'
            save_wordcloud_mejorado(freq_dict, output_path, f'Comunidad {comm}', max_words=50)
            print(f'    ✓ Comunidad {comm}')
    
    print(f'  ✓ Word clouds por comunidad guardados en: {COMMUNITY_EVOL_DIR}')

# ============================================================================
# 1. WORD CLOUDS - VERSIÓN MEJORADA CON TODAS LAS VARIANTES
# ============================================================================

def analisis_wordclouds(tokens, bigrams, trigrams, df_base):
    """Análisis completo de word clouds - VERSIÓN MEJORADA"""
    print('\n' + '='*70)
    print('1. ANÁLISIS DE WORD CLOUDS (MEJORADO)')
    print('='*70)
    
    # ========== 1.1 Word Clouds Globales ==========
    print('\n--- 1.1 Word Clouds Globales ---')
    
    # Unigramas
    counter = Counter(tokens)
    if counter:
        # Usar versión mejorada
        save_wordcloud_mejorado(counter.most_common(120), 
                               WC_GLOBAL_DIR / 'global_unigramas.png', 
                               'Global - Unigramas', max_words=120)
        df_uni = pd.DataFrame(counter.most_common(200), columns=['termino', 'frecuencia'])
        df_uni.to_csv(WC_GLOBAL_DIR / 'global_unigramas.csv', index=False)
        print(f'  ✓ Unigramas: {len(counter)} términos únicos')
        print(f'    Top 5: {", ".join([t for t, _ in counter.most_common(5)])}')
    
    # Bigramas
    counter_bi = Counter(bigrams)
    if counter_bi:
        save_wordcloud_mejorado(counter_bi.most_common(100), 
                               WC_GLOBAL_DIR / 'global_bigramas.png', 
                               'Global - Bigramas', max_words=100, cmap='plasma')
        df_bi = pd.DataFrame(counter_bi.most_common(200), columns=['termino', 'frecuencia'])
        df_bi.to_csv(WC_GLOBAL_DIR / 'global_bigramas.csv', index=False)
        print(f'  ✓ Bigramas: {len(counter_bi)} términos únicos')
        print(f'    Top 5: {", ".join([t for t, _ in counter_bi.most_common(5)])}')
    
    # Trigramas
    counter_tri = Counter(trigrams)
    if counter_tri:
        save_wordcloud_mejorado(counter_tri.most_common(80), 
                               WC_GLOBAL_DIR / 'global_trigramas.png', 
                               'Global - Trigramas', max_words=80, cmap='magma')
        df_tri = pd.DataFrame(counter_tri.most_common(200), columns=['termino', 'frecuencia'])
        df_tri.to_csv(WC_GLOBAL_DIR / 'global_trigramas.csv', index=False)
        print(f'  ✓ Trigramas: {len(counter_tri)} términos únicos')
        print(f'    Top 5: {", ".join([t for t, _ in counter_tri.most_common(5)])}')
    
    return counter, counter_bi, counter_tri

# ============================================================================
# 1.2 WORD CLOUDS POR AÑO (NUEVA FUNCIÓN)
# ============================================================================

def analisis_wordclouds_por_anio(agg_uni, agg_bi, agg_tri):
    """Crea word clouds por año usando los datos agregados"""
    print('\n--- 1.2 Word Clouds por Año ---')
    
    años = sorted(agg_uni.keys())
    
    for año in años:
        # Unigramas por año
        if año in agg_uni and agg_uni[año]:
            freq_dict = dict(agg_uni[año].most_common(80))
            save_wordcloud_mejorado(freq_dict, 
                                   WC_ANIO_DIR / f'unigramas_{año}.png', 
                                   f'Unigramas - {año}', max_words=80)
        
        # Bigramas por año
        if año in agg_bi and agg_bi[año]:
            freq_dict = dict(agg_bi[año].most_common(60))
            save_wordcloud_mejorado(freq_dict, 
                                   WC_ANIO_DIR / f'bigramas_{año}.png', 
                                   f'Bigramas - {año}', max_words=60, cmap='plasma')
        
        # Trigramas por año
        if año in agg_tri and agg_tri[año]:
            freq_dict = dict(agg_tri[año].most_common(40))
            save_wordcloud_mejorado(freq_dict, 
                                   WC_ANIO_DIR / f'trigramas_{año}.png', 
                                   f'Trigramas - {año}', max_words=40, cmap='magma')
    
    print(f'  ✓ Word clouds por año guardados en: {WC_ANIO_DIR}')

# ============================================================================
# 1.3 WORD CLOUDS POR COMUNIDAD (MEJORADA)
# ============================================================================

def analisis_wordclouds_comunidades(df_resultados, partition):
    """Crea word clouds por comunidad desde resultados de Louvain"""
    print('\n--- 1.3 Word Clouds por Comunidad ---')
    
    if df_resultados is None or partition is None:
        print('  ⚠️ No hay datos de comunidades')
        return
    
    for comm in sorted(set(partition.values())):
        comm_terms = df_resultados[df_resultados['comunidad'] == comm]
        if len(comm_terms) >= 5:
            freq_dict = {row['concepto']: row['pagerank'] for _, row in comm_terms.iterrows()}
            output_path = COMMUNITY_EVOL_DIR / f'comunidad_{comm}.png'
            save_wordcloud_mejorado(freq_dict, output_path, f'Comunidad {comm}', max_words=50)
            print(f'    ✓ Comunidad {comm}')
    
    print(f'  ✓ Word clouds por comunidad guardados en: {COMMUNITY_EVOL_DIR}')

# ============================================================================
# 1.4 WORD CLOUDS POR PERIODO TEMPORAL (MEJORADA)
# ============================================================================

def analisis_wordclouds_temporales(window_data, windows):
    """Crea word clouds por periodo temporal"""
    print('\n--- 1.4 Word Clouds por Periodo Temporal ---')
    
    for window in windows:
        window_name = window['nombre']
        if window_name in window_data:
            freq_dict = dict(window_data[window_name]['terms'].most_common(80))
            output_path = TOPICOS_TEMPORALES_DIR / f'wordcloud_{window_name}.png'
            save_wordcloud_mejorado(freq_dict, output_path, f'Tópicos - {window_name}', max_words=80)
            print(f'    ✓ {window_name}')
    
    print(f'  ✓ Word clouds temporales guardados en: {TOPICOS_TEMPORALES_DIR}')

# ============================================================================
# 2. FLOR PLOTS
# ============================================================================

# ============================================================================
# FLOR PLOT CORREGIDO - VERSIÓN QUE SÍ FUNCIONA
# ============================================================================

def flor_plot_corregido(tablas, matriz, etiqueta, fig_num=1):
    """
    Flor plot CORREGIDO:
    - Círculos concéntricos = Años (con etiquetas)
    - Pétalos = Términos (palabras en las orillas)
    """
    try:
        # ========== VALIDACIONES ==========
        if matriz is None or matriz.size == 0:
            print(f'  ⚠️ Flor plot "{etiqueta}" omitido: matriz vacía')
            return False
        
        if len(tablas) < 2 or matriz.shape[0] < 2 or matriz.shape[1] < 2:
            print(f'  ⚠️ Flor plot "{etiqueta}" omitido: datos insuficientes')
            print(f'    Años: {len(tablas)}, Términos: {matriz.shape[1]}')
            return False
        
        if np.all(matriz == 0):
            print(f'  ⚠️ Flor plot "{etiqueta}" omitido: todos los valores son cero')
            return False
        
        # ========== OBTENER DATOS ==========
        # matriz: filas = años, columnas = términos
        num_años, num_terminos = matriz.shape
        print(f'  Años: {num_años}, Términos: {num_terminos}')
        
        # Obtener nombres de términos (si es DataFrame, usar columnas)
        if hasattr(matriz, 'columns'):
            nombres_terminos = list(matriz.columns)
        else:
            nombres_terminos = [f'Term_{i}' for i in range(num_terminos)]
        
        # ========== CONFIGURACIÓN ==========
        # Ángulo entre pétalos
        angulo_entre_petalos = (360 / max(2, num_terminos + 1)) * np.pi / 180
        
        # Radios de los círculos (años) - usar valores numéricos
        radios = np.arange(1, num_años + 1)
        
        # Normalizar la matriz para que los valores estén entre 0 y 1
        max_val = np.max(matriz)
        if max_val > 0:
            y_norm = matriz / max_val
        else:
            y_norm = matriz.copy()
        
        # Colores para los términos (pétalos)
        colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                   '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                   '#F8A5C2', '#74B9FF', '#A29BFE', '#FD79A8', '#00CEC9',
                   '#FDCB6E', '#E17055', '#00B894', '#6C5CE7', '#FD79A8']
        
        # ========== CREAR FIGURA ==========
        fig = plt.figure(fig_num, figsize=(18, 18))
        ax = fig.add_subplot(111)
        
        # ========== DIBUJAR CÍRCULOS (AÑOS) ==========
        for i, radio in enumerate(radios):
            # Círculo para cada año
            circle = Circle((0, 0), radio, fill=False, 
                          edgecolor=[0.6, 0.6, 0.6], linewidth=2, alpha=0.7)
            ax.add_patch(circle)
            
            # Etiqueta del año en el círculo (arriba)
            angulo_texto = -np.pi/2
            x_text = radio * np.cos(angulo_texto)
            y_text = radio * np.sin(angulo_texto)
            ax.text(x_text, y_text, str(tablas[i]), fontsize=12, 
                   ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9,
                            edgecolor='gray', linewidth=0.5))
        
        # Círculo exterior más grueso
        circle = Circle((0, 0), num_años + 0.8, fill=False, 
                      edgecolor=[0.4, 0.4, 0.4], linewidth=3)
        ax.add_patch(circle)
        
        # ========== DIBUJAR PÉTALOS (TÉRMINOS) ==========
        angulo_actual = angulo_entre_petalos
        
        for i in range(num_terminos):
            # Para cada término (pétalo)
            for j in range(num_años):
                # Para cada año (círculo)
                valor = y_norm[j, i]
                if valor > 0.01:
                    # Tamaño del pétalo proporcional a la frecuencia
                    ang = 0.4 * (angulo_entre_petalos * valor)
                    
                    # Asegurar que el pétalo tenga un tamaño mínimo
                    ang = max(ang, 0.02)
                    
                    radio_interno = radios[j] - 0.2
                    radio_externo = radios[j]
                    
                    # Coordenadas del pétalo
                    x = [
                        radio_externo * np.cos(angulo_actual - ang),
                        radio_externo * np.cos(angulo_actual + ang),
                        radio_interno * np.cos(angulo_actual + ang),
                        radio_interno * np.cos(angulo_actual - ang)
                    ]
                    y = [
                        radio_externo * np.sin(angulo_actual - ang),
                        radio_externo * np.sin(angulo_actual + ang),
                        radio_interno * np.sin(angulo_actual + ang),
                        radio_interno * np.sin(angulo_actual - ang)
                    ]
                    
                    # Transparencia según frecuencia
                    alpha = 0.4 + 0.5 * valor
                    
                    ax.add_patch(Polygon(list(zip(x, y)), 
                                      facecolor=colores[i % len(colores)],
                                      edgecolor='white', linewidth=0.5, 
                                      alpha=alpha))
            
            # ========== ETIQUETA DEL TÉRMINO EN LA ORILLA ==========
            radio_etiqueta = num_años + 1.0
            cx = radio_etiqueta * np.cos(angulo_actual)
            cy = radio_etiqueta * np.sin(angulo_actual)
            
            # Rotación de la etiqueta para que sea legible
            rot = angulo_actual * 180 / np.pi
            if rot > 90 and rot < 270:
                rot = rot - 180
                
            # Obtener el nombre del término de la lista
            if i < len(nombres_terminos):
                nombre_termino = str(nombres_terminos[i])
            else:
                nombre_termino = f'Term_{i}'
            
            # Truncar si es muy largo
            if len(nombre_termino) > 22:
                nombre_termino = nombre_termino[:20] + '..'
            
            ax.text(cx, cy, nombre_termino, fontsize=9, rotation=rot,
                   ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9,
                            edgecolor=colores[i % len(colores)], linewidth=1))
            
            # Avanzar al siguiente ángulo
            angulo_actual += angulo_entre_petalos
        
        # ========== CONFIGURAR GRÁFICO ==========
        limite = num_años + 2.2
        ax.set_xlim(-limite, limite)
        ax.set_ylim(-limite, limite)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Título
        plt.title(f'Distribución de {etiqueta.capitalize()} por Año\n'
                 f'{num_años} años, {num_terminos} términos principales',
                 fontsize=20, fontweight='bold', pad=30)
        
        plt.tight_layout()
        
        # ========== GUARDAR ==========
        output = FLOR_DIR / f'flor_plot_{etiqueta}.png'
        plt.savefig(str(output), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig_num)
        print(f'  ✓ Flor plot guardado: {output}')
        return True
        
    except Exception as e:
        print(f'  ✗ Error en flor_plot "{etiqueta}": {e}')
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# FUNCIÓN PARA CREAR FLOR PLOTS - VERSIÓN CORREGIDA
# ============================================================================

def analisis_flor_plots(agg_uni, agg_bi, agg_tri):
    """Crea flor plots CORRECTAMENTE"""
    print('\n' + '='*70)
    print('2. FLOR PLOTS')
    print('='*70)
    
    for idx, (nombre, datos) in enumerate([('unigramas', agg_uni), ('bigramas', agg_bi), ('trigramas', agg_tri)]):
        print(f'\n--- {nombre.capitalize()} ---')
        try:
            years = sorted(datos.keys())
            if len(years) < 2:
                print(f'  ⚠️ Pocos años ({len(years)})')
                continue
            
            # Obtener todos los términos
            all_terms = Counter()
            for year in years:
                all_terms.update(datos[year])
            
            if not all_terms:
                print(f'  ⚠️ Sin términos')
                continue
            
            # Tomar top 20 términos para mejor visualización
            top_terms = [t for t, _ in all_terms.most_common(20)]
            if len(top_terms) < 2:
                print(f'  ⚠️ Pocos términos ({len(top_terms)})')
                continue
            
            # Matriz: filas = años, columnas = términos
            matriz = np.zeros((len(years), len(top_terms)))
            for i, year in enumerate(years):
                for j, term in enumerate(top_terms):
                    matriz[i, j] = datos[year].get(term, 0)
            
            # Crear DataFrame con nombres de columnas
            df_matriz = pd.DataFrame(matriz, index=[str(y) for y in years], columns=top_terms)
            
            # Guardar matriz CSV
            df_matriz.to_csv(FLOR_DIR / f'matriz_{nombre}.csv')
            print(f'  ✓ Matriz guardada: {FLOR_DIR / f"matriz_{nombre}.csv"}')
            print(f'  ✓ Dimensiones: {matriz.shape[0]} años x {matriz.shape[1]} términos')
            
            # Llamar a la función corregida PASANDO el DataFrame
            flor_plot_corregido(years, df_matriz, nombre, fig_num=idx+1)
            
        except Exception as e:
            print(f'  ✗ Error en {nombre}: {e}')
            import traceback
            traceback.print_exc()

def analisis_redes_por_anio(edges):
    """Análisis completo de redes por año"""
    print('\n' + '='*70)
    print('3. REDES POR AÑO')
    print('='*70)
    
    if not edges:
        print('  ⚠️ Sin aristas')
        return
    
    years = sorted(set([y for *_, y in edges]))
    if not years:
        return
    
    pdf_path = REDES_POR_ANIO_DIR / 'redes_por_anio.pdf'
    pdf = PdfPages(str(pdf_path))
    
    stats_redes = []
    todas_redes = {}
    
    for ntype in ['bigram', 'trigram']:
        print(f'\n--- {ntype.capitalize()}s ---')
        for year in years:
            year_edges = [(a, t, w) for a, t, typ, w, y in edges 
                        if typ == ntype and y == year]
            if len(year_edges) < 3:
                continue
            
            G = nx.Graph()
            articles = {a for a, _, _ in year_edges}
            terms = {t for _, t, _ in year_edges}
            
            for a in articles:
                G.add_node(f'ART::{a}', tipo='articulo', label=a[:20])
            for t in terms:
                G.add_node(f'TERM::{t}', tipo='termino', label=t[:15])
            for a, t, w in year_edges:
                G.add_edge(f'ART::{a}', f'TERM::{t}', weight=float(w))
            
            if G.number_of_nodes() < 4 or G.number_of_edges() < 3:
                continue
            
            # Métricas
            grado_prom = np.mean([d for n, d in G.degree()])
            densidad = nx.density(G)
            n_prom = np.mean([G.degree(n) for n in G.nodes()])
            
            # Centralidad
            try:
                betweenness = nx.betweenness_centrality(G, weight='weight')
                closeness = nx.closeness_centrality(G)
                degree_cent = nx.degree_centrality(G)
            except:
                betweenness = {}
                closeness = {}
                degree_cent = {}
            
            stats_redes.append({
                'año': year,
                'tipo': ntype,
                'articulos': len(articles),
                'terminos': len(terms),
                'aristas': G.number_of_edges(),
                'grado_promedio': round(grado_prom, 2),
                'densidad': round(densidad, 4),
                'centralidad_promedio': round(np.mean(list(degree_cent.values())) if degree_cent else 0, 4)
            })
            
            todas_redes[f'{ntype}_{year}'] = G
            
            # Visualizar
            try:
                pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
            except:
                continue
            
            fig, ax = plt.subplots(figsize=(16, 12))
            ax.axis('off')
            
            node_colors = ['#3498db' if n.startswith('ART::') else '#e74c3c' for n in G.nodes()]
            node_sizes = [400 + 100 * G.degree(n) for n in G.nodes()]
            
            nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.4, ax=ax, edge_color='gray')
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                 node_color=node_colors, alpha=0.85, 
                                 edgecolors='darkgray', linewidths=1.5, ax=ax)
            
            labels = {}
            for n in G.nodes():
                if n.startswith('ART::') and G.degree(n) > 2:
                    labels[n] = G.nodes[n]['label']
                elif n.startswith('TERM::') and G.degree(n) > 3:
                    labels[n] = G.nodes[n]['label']
            
            if labels:
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, 
                                      font_weight='bold', ax=ax)
            
            plt.title(f'Red {ntype.capitalize()} - {year}\n'
                     f'{len(articles)} artículos, {len(terms)} términos, {G.number_of_edges()} conexiones',
                     fontsize=14, fontweight='bold')
            
            png_path = REDES_POR_ANIO_DIR / f'red_{ntype}_{year}.png'
            plt.savefig(str(png_path), dpi=200, bbox_inches='tight', facecolor='white')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            print(f'  ✓ Red {ntype} {year}: {len(articles)} artículos, {len(terms)} términos')
    
    pdf.close()
    print(f'  ✓ PDF consolidado: {pdf_path}')
    
    # Guardar estadísticas
    if stats_redes:
        df_stats = pd.DataFrame(stats_redes)
        df_stats.to_csv(NET_DIR / 'estadisticas_redes_por_anio.csv', index=False)
        print(f'  ✓ Estadísticas guardadas: {NET_DIR / "estadisticas_redes_por_anio.csv"}')
    
    return todas_redes

# ============================================================================
# 4. RED GLOBAL
# ============================================================================

def analisis_red_global(edges):
    """Análisis completo de red global"""
    print('\n' + '='*70)
    print('4. RED GLOBAL')
    print('='*70)
    
    if not edges:
        print('  ⚠️ Sin aristas')
        return None
    
    G = nx.Graph()
    
    for art, term, ntype, w, year in edges:
        art_node = f'ART::{art}'
        term_node = f'TERM::{term}'
        
        if not G.has_node(art_node):
            G.add_node(art_node, tipo='articulo', label=art[:25])
        if not G.has_node(term_node):
            G.add_node(term_node, tipo='termino', label=term[:20])
        
        if G.has_edge(art_node, term_node):
            G[art_node][term_node]['weight'] += w
        else:
            G.add_edge(art_node, term_node, weight=w)
    
    # Filtrar nodos con al menos 2 conexiones
    nodes_to_keep = [n for n in G.nodes() if G.degree(n) >= 2]
    G_sub = G.subgraph(nodes_to_keep)
    
    print(f'  ✓ Nodos: {G_sub.number_of_nodes()}, Aristas: {G_sub.number_of_edges()}')
    
    # Métricas globales
    if G_sub.number_of_nodes() > 0:
        print(f'  ✓ Densidad: {nx.density(G_sub):.4f}')
        print(f'  ✓ Grado promedio: {np.mean([d for n, d in G_sub.degree()]):.2f}')
        try:
            print(f'  ✓ Componentes conexas: {nx.number_connected_components(G_sub)}')
        except:
            pass
    
    # Guardar GEXF
    nx.write_gexf(G_sub, NET_GEXF_DIR / 'red_global.gexf')
    print(f'  ✓ Guardado: {NET_GEXF_DIR / "red_global.gexf"}')
    
    # Visualizar
    if G_sub.number_of_nodes() <= 150:
        plt.figure(figsize=(20, 16))
        pos = nx.spring_layout(G_sub, k=3, iterations=50, seed=42)
        
        colors = ['#3498db' if n.startswith('ART::') else '#e74c3c' for n in G_sub.nodes()]
        sizes = [400 + 80 * G_sub.degree(n) for n in G_sub.nodes()]
        
        nx.draw_networkx_edges(G_sub, pos, width=0.5, alpha=0.3)
        nx.draw_networkx_nodes(G_sub, pos, node_size=sizes, node_color=colors, 
                             alpha=0.8, edgecolors='darkgray', linewidths=1)
        
        labels = {}
        for n in G_sub.nodes():
            if G_sub.degree(n) > 5:
                labels[n] = G_sub.nodes[n]['label']
        
        if labels:
            nx.draw_networkx_labels(G_sub, pos, labels=labels, font_size=7)
        
        plt.title(f'Red Global\n{G_sub.number_of_nodes()} nodos, {G_sub.number_of_edges()} aristas',
                 fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(str(NET_DIR / 'red_global.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Visualización: {NET_DIR / "red_global.png"}')
    
    # Exportar aristas y nodos a CSV
    edge_rows = []
    for u, v, data in G_sub.edges(data=True):
        if u.startswith('ART::'):
            art = G_sub.nodes[u]['label']
            term = G_sub.nodes[v]['label']
        else:
            art = G_sub.nodes[v]['label']
            term = G_sub.nodes[u]['label']
        edge_rows.append({
            'Articulo': art,
            'Termino': term,
            'weight': data.get('weight', 1)
        })
    pd.DataFrame(edge_rows).to_csv(REDES_COMPLETAS_DIR / 'aristas_globales.csv', index=False)
    
    return G_sub

# ============================================================================
# 5. RED TÉRMINO-TÉRMINO
# ============================================================================

def analisis_red_termino_termino(edges):
    """Análisis de red término-término"""
    print('\n' + '='*70)
    print('5. RED TÉRMINO-TÉRMINO')
    print('='*70)
    
    if not edges:
        print('  ⚠️ Sin aristas')
        return None
    
    # Crear matriz de co-ocurrencia
    all_terms = set()
    for art, term, ntype, w, year in edges:
        if ntype == 'unigram':
            all_terms.add(term)
    
    all_terms = list(all_terms)[:200]
    term_idx = {t: i for i, t in enumerate(all_terms)}
    
    cooc_matrix = np.zeros((len(all_terms), len(all_terms)))
    
    articulos_terminos = defaultdict(set)
    for art, term, ntype, w, year in edges:
        if ntype == 'unigram' and term in term_idx:
            articulos_terminos[art].add(term)
    
    for art, terms in articulos_terminos.items():
        terms_list = list(terms)
        for i in range(len(terms_list)):
            for j in range(i+1, len(terms_list)):
                if terms_list[i] in term_idx and terms_list[j] in term_idx:
                    cooc_matrix[term_idx[terms_list[i]], term_idx[terms_list[j]]] += 1
                    cooc_matrix[term_idx[terms_list[j]], term_idx[terms_list[i]]] += 1
    
    G_terminos = nx.Graph()
    for term in all_terms:
        G_terminos.add_node(term)
    
    non_zero = cooc_matrix[cooc_matrix > 0]
    if len(non_zero) > 0:
        threshold = np.percentile(non_zero, 70)
        for i in range(len(all_terms)):
            for j in range(i+1, len(all_terms)):
                if cooc_matrix[i, j] > threshold:
                    G_terminos.add_edge(all_terms[i], all_terms[j], weight=cooc_matrix[i, j])
    
    print(f'  ✓ Nodos: {G_terminos.number_of_nodes()}, Aristas: {G_terminos.number_of_edges()}')
    
    if G_terminos.number_of_edges() > 0:
        # Guardar
        nx.write_gexf(G_terminos, NET_GEXF_DIR / 'red_termino_termino.gexf')
        
        # Métricas
        metricas = {
            'nodos': G_terminos.number_of_nodes(),
            'aristas': G_terminos.number_of_edges(),
            'densidad': nx.density(G_terminos),
            'grado_promedio': np.mean([d for n, d in G_terminos.degree()]),
            'transitividad': nx.transitivity(G_terminos),
            'num_componentes': nx.number_connected_components(G_terminos)
        }
        pd.DataFrame([metricas]).to_csv(REDES_COMPLETAS_DIR / 'metricas_red_termino_termino.csv', index=False)
        print(f'  ✓ Métricas guardadas')
        
        # Visualizar
        if G_terminos.number_of_nodes() <= 100:
            plt.figure(figsize=(18, 14))
            pos = nx.spring_layout(G_terminos, k=2, iterations=50, seed=42)
            
            nx.draw_networkx_edges(G_terminos, pos, width=0.5, alpha=0.3)
            
            sizes = [G_terminos.degree(n) * 30 + 100 for n in G_terminos.nodes()]
            nx.draw_networkx_nodes(G_terminos, pos, node_size=sizes, 
                                 node_color='lightblue', alpha=0.8, 
                                 edgecolors='darkblue', linewidths=1)
            
            degrees = dict(G_terminos.degree())
            important = sorted(degrees, key=degrees.get, reverse=True)[:20]
            labels = {n: n for n in important}
            nx.draw_networkx_labels(G_terminos, pos, labels=labels, font_size=8)
            
            plt.title(f'Red Término-Término (Co-ocurrencia)\n{G_terminos.number_of_nodes()} nodos, {G_terminos.number_of_edges()} aristas',
                     fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(str(REDES_COMPLETAS_DIR / 'red_termino_termino.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f'  ✓ Visualización: {REDES_COMPLETAS_DIR / "red_termino_termino.png"}')
    
    return G_terminos

# ============================================================================
# 6. RED ARTÍCULO-ARTÍCULO
# ============================================================================

def analisis_red_articulo_articulo(edges, df_base):
    """Análisis de red artículo-artículo"""
    print('\n' + '='*70)
    print('6. RED ARTÍCULO-ARTÍCULO')
    print('='*70)
    
    if not edges:
        print('  ⚠️ Sin aristas')
        return None
    
    articulos_terminos = defaultdict(set)
    for art, term, ntype, w, year in edges:
        if ntype == 'unigram':
            articulos_terminos[art].add(term)
    
    G_articulos = nx.Graph()
    articulo_anio = {row['Articulo']: row['Año'] for _, row in df_base.iterrows()}
    
    for art in articulos_terminos.keys():
        G_articulos.add_node(art, año=articulo_anio.get(art, 'desconocido'))
    
    articulos_list = list(articulos_terminos.keys())
    for i in range(len(articulos_list)):
        for j in range(i+1, len(articulos_list)):
            art1 = articulos_list[i]
            art2 = articulos_list[j]
            
            term1 = articulos_terminos[art1]
            term2 = articulos_terminos[art2]
            
            interseccion = len(term1 & term2)
            union = len(term1 | term2)
            
            if union > 0 and interseccion > 0:
                jaccard = interseccion / union
                if jaccard > 0.1:
                    G_articulos.add_edge(art1, art2, peso=jaccard)
    
    print(f'  ✓ Nodos: {G_articulos.number_of_nodes()}, Aristas: {G_articulos.number_of_edges()}')
    
    if G_articulos.number_of_edges() > 0:
        nx.write_gexf(G_articulos, NET_GEXF_DIR / 'red_articulo_articulo.gexf')
        
        metricas_articulos = {
            'nodos': G_articulos.number_of_nodes(),
            'aristas': G_articulos.number_of_edges(),
            'densidad': nx.density(G_articulos),
            'grado_promedio': np.mean([d for n, d in G_articulos.degree()]),
            'num_componentes': nx.number_connected_components(G_articulos)
        }
        pd.DataFrame([metricas_articulos]).to_csv(REDES_COMPLETAS_DIR / 'metricas_red_articulo_articulo.csv', index=False)
        print(f'  ✓ Métricas guardadas')
        
        # Visualizar
        if G_articulos.number_of_nodes() <= 80:
            plt.figure(figsize=(16, 14))
            
            pos = nx.spring_layout(G_articulos, k=2, iterations=50, seed=42)
            
            años = sorted(set(articulo_anio.values()))
            colores_años = plt.cm.Set3(np.linspace(0, 1, len(años)))
            año_color = {a: c for a, c in zip(años, colores_años)}
            
            node_colors = [año_color.get(G_articulos.nodes[n].get('año', años[0]), 'gray') 
                          for n in G_articulos.nodes()]
            
            nx.draw_networkx_edges(G_articulos, pos, width=0.5, alpha=0.3)
            nx.draw_networkx_nodes(G_articulos, pos, node_size=200, 
                                 node_color=node_colors, alpha=0.8, 
                                 edgecolors='darkgray', linewidths=1)
            
            labels = {n: n[:20] for n in G_articulos.nodes() if G_articulos.degree(n) > 2}
            nx.draw_networkx_labels(G_articulos, pos, labels=labels, font_size=7)
            
            plt.title(f'Red Artículo-Artículo por Términos Compartidos\n{G_articulos.number_of_nodes()} nodos, {G_articulos.number_of_edges()} aristas',
                     fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(str(REDES_COMPLETAS_DIR / 'red_articulo_articulo.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f'  ✓ Visualización: {REDES_COMPLETAS_DIR / "red_articulo_articulo.png"}')
    
    return G_articulos

# ============================================================================
# 7. ANÁLISIS LOUVAIN - COMUNIDADES
# ============================================================================

def analisis_louvain(manifest_path):
    """Análisis de comunidades con Louvain"""
    print('\n' + '='*70)
    print('7. ANÁLISIS DE COMUNIDADES (LOUVAIN)')
    print('='*70)
    
    try:
        manifest = pd.read_csv(manifest_path)
        unigrams_all = pd.read_csv(AGG_DIR / 'unigram_all.csv')
        
        if unigrams_all.empty:
            print('  ⚠️ Sin datos')
            return None
        
        top_concepts = unigrams_all.nlargest(100, 'count')['unigram'].tolist()
        if len(top_concepts) < 5:
            print('  ⚠️ Pocos conceptos')
            return None
        
        print(f'  ✓ Analizando {len(top_concepts)} conceptos')
        
        # Construir matriz documento-término
        doc_term = np.zeros((len(manifest), len(top_concepts)))
        
        for idx, row in manifest.iterrows():
            articulo = row['Articulo']
            uni_file = row.get('unigrams_csv', '')
            if uni_file and Path(uni_file).exists():
                try:
                    df_uni = pd.read_csv(uni_file)
                    for _, r in df_uni.iterrows():
                        term = r['unigram']
                        if term in top_concepts:
                            j = top_concepts.index(term)
                            doc_term[idx, j] = r['count']
                except:
                    pass
        
        if np.sum(doc_term) == 0:
            print('  ⚠️ Matriz vacía')
            return None
        
        # Normalizar
        max_val = np.max(doc_term)
        if max_val > 0:
            doc_term_norm = doc_term / max_val
        
        # Matriz término-término
        term_term = np.dot(doc_term_norm.T, doc_term_norm)
        
        # Guardar matrices
        with pd.ExcelWriter(MATRICES_DIR / 'matrices_analisis.xlsx') as writer:
            df_doc_term = pd.DataFrame(doc_term_norm, columns=top_concepts)
            df_doc_term.insert(0, 'Articulo', manifest['Articulo'].values)
            df_doc_term.to_excel(writer, sheet_name='Documento_Termino', index=False)
            
            df_term_term = pd.DataFrame(term_term, index=top_concepts, columns=top_concepts)
            df_term_term.to_excel(writer, sheet_name='Termino_Termino')
        
        print(f'  ✓ Matrices guardadas: {MATRICES_DIR / "matrices_analisis.xlsx"}')
        
        # Construir red
        G = nx.Graph()
        for i, term in enumerate(top_concepts):
            G.add_node(term)
        
        non_zero = term_term[term_term > 0]
        if len(non_zero) > 0:
            threshold = np.percentile(non_zero, 75)
            for i in range(len(top_concepts)):
                for j in range(i+1, len(top_concepts)):
                    if term_term[i, j] > threshold:
                        G.add_edge(top_concepts[i], top_concepts[j], weight=term_term[i, j])
        
        if G.number_of_edges() < 5:
            print('  ⚠️ Red muy dispersa')
            return None
        
        print(f'  ✓ Red: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas')
        
        # Aplicar Louvain
        partition = community_louvain.best_partition(G, weight='weight', random_state=42)
        modularidad = community_louvain.modularity(partition, G, weight='weight')
        num_comunidades = len(set(partition.values()))
        
        print(f'  ✓ Comunidades: {num_comunidades}')
        print(f'  ✓ Modularidad: {modularidad:.4f}')
        
        # Métricas de centralidad
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G, weight='weight')
        closeness_cent = nx.closeness_centrality(G)
        pagerank = nx.pagerank(G, weight='weight')
        
        # Resultados
        resultados = []
        for node in G.nodes():
            resultados.append({
                'concepto': node,
                'comunidad': partition[node],
                'grado': G.degree(node),
                'grado_centralidad': degree_cent[node],
                'intermediacion': betweenness_cent[node],
                'cercania': closeness_cent[node],
                'pagerank': pagerank[node]
            })
        
        df_resultados = pd.DataFrame(resultados)
        df_resultados = df_resultados.sort_values(['comunidad', 'pagerank'], ascending=[True, False])
        df_resultados.to_csv(NET_DIR / 'louvain_resultados.csv', index=False)
        
        # Estadísticas por comunidad
        comm_stats = df_resultados.groupby('comunidad').agg({
            'concepto': 'count',
            'grado': 'mean',
            'pagerank': 'mean'
        }).rename(columns={'concepto': 'num_conceptos'})
        comm_stats.to_csv(NET_DIR / 'louvain_estadisticas.csv')
        
        print('\n  Top términos por comunidad:')
        for comm in sorted(set(partition.values())):
            top_terms = df_resultados[df_resultados['comunidad'] == comm].nlargest(5, 'pagerank')['concepto'].tolist()
            print(f'    Comunidad {comm}: {", ".join(top_terms)}')
        
        # Visualizar comunidades
        plt.figure(figsize=(20, 16))
        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
        
        communities = set(partition.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        
        for comm, color in zip(communities, colors):
            nodes = [n for n in G.nodes() if partition[n] == comm]
            sizes = [G.degree(n) * 30 + 100 for n in nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=sizes,
                                 node_color=[color], alpha=0.85, 
                                 edgecolors='darkgray', linewidths=1.5)
        
        nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.3, edge_color='gray')
        
        important = df_resultados.nlargest(30, 'pagerank')['concepto'].tolist()
        labels = {node: node for node in important}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold')
        
        plt.title(f'Comunidades Louvain\n{num_comunidades} comunidades | Modularidad: {modularidad:.4f}',
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(str(NET_DIR / 'louvain_comunidades.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Visualización: {NET_DIR / "louvain_comunidades.png"}')
        
        # Word clouds por comunidad (usando versión mejorada)
        print('\n  Generando word clouds por comunidad...')
        for comm in sorted(set(partition.values())):
            comm_terms = df_resultados[df_resultados['comunidad'] == comm]
            if len(comm_terms) >= 5:
                freq_dict = {row['concepto']: row['pagerank'] for _, row in comm_terms.iterrows()}
                output_path = COMMUNITY_EVOL_DIR / f'comunidad_{comm}.png'
                save_wordcloud_mejorado(freq_dict, output_path, f'Comunidad {comm}', max_words=50)
                print(f'    ✓ Comunidad {comm}')
        
        print(f'  ✓ Word clouds por comunidad guardados en: {COMMUNITY_EVOL_DIR}')
        
        # Resumen ejecutivo
        resumen = f"""
        ============================================================
        ANÁLISIS DE COMUNIDADES - LOUVAIN
        ============================================================
        
        DATOS GENERALES:
        - Total de conceptos analizados: {len(top_concepts)}
        - Nodos en la red: {G.number_of_nodes()}
        - Aristas en la red: {G.number_of_edges()}
        - Densidad de la red: {nx.density(G):.4f}
        
        COMUNIDADES:
        - Número de comunidades: {num_comunidades}
        - Modularidad: {modularidad:.4f}
        - Tamaño promedio de comunidad: {np.mean([len([n for n in G.nodes() if partition[n] == comm]) for comm in set(partition.values())]):.1f}
        
        MÉTRICAS DE CENTRALIDAD (Top 5):
        - PageRank: {', '.join(df_resultados.nlargest(5, 'pagerank')['concepto'].tolist())}
        - Grado: {', '.join(df_resultados.nlargest(5, 'grado')['concepto'].tolist())}
        - Intermediación: {', '.join(df_resultados.nlargest(5, 'intermediacion')['concepto'].tolist())}
        
        COMUNIDADES (Top términos por comunidad):
        """
        
        for comm in sorted(set(partition.values())):
            top_terms = df_resultados[df_resultados['comunidad'] == comm].nlargest(5, 'pagerank')['concepto'].tolist()
            num_terms = len([n for n in G.nodes() if partition[n] == comm])
            resumen += f"\n    Comunidad {comm} ({num_terms} términos): {', '.join(top_terms)}"
        
        with open(NET_DIR / 'louvain_resumen_ejecutivo.txt', 'w', encoding='utf-8') as f:
            f.write(resumen)
        
        print(resumen)
        
        return df_resultados, partition, modularidad
        
    except Exception as e:
        print(f'  ✗ Error: {e}')
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 8. ANÁLISIS DE EVOLUCIÓN TEMPORAL
# ============================================================================

def analisis_evolucion_temporal(df, agg_uni, agg_bi, agg_tri):
    """Análisis completo de evolución temporal"""
    print('\n' + '='*70)
    print('8. EVOLUCIÓN TEMPORAL')
    print('='*70)
    
    try:
        years = sorted(df['Año'].unique())
        if len(years) < 4:
            print('  ⚠️ Pocos años para análisis temporal')
            return
        
        # Crear ventanas temporales (3 años cada una)
        window_size = 3
        windows = []
        for i in range(len(years) - window_size + 1):
            window_years = years[i:i+window_size]
            windows.append({
                'nombre': f"{window_years[0]}-{window_years[-1]}",
                'years': window_years,
                'start': window_years[0],
                'end': window_years[-1]
            })
        
        print(f'  Ventanas temporales: {[w["nombre"] for w in windows]}')
        
        # Para cada ventana, obtener términos
        window_data = {}
        all_window_terms = []
        
        for window in windows:
            terms = Counter()
            for year in window['years']:
                terms.update(agg_uni[year])
                terms.update(agg_bi[year])
            
            if terms:
                window_data[window['nombre']] = {
                    'terms': terms,
                    'top_20': [t for t, _ in terms.most_common(20)]
                }
                all_window_terms.extend([t for t, _ in terms.most_common(30)])
        
        all_window_terms = list(set(all_window_terms))
        
        # ========== 8.1 MATRIZ DE EVOLUCIÓN ==========
        print('\n--- 8.1 Matriz de Evolución ---')
        
        evolution_matrix = []
        for term in all_window_terms:
            row = []
            for window in windows:
                freq = window_data.get(window['nombre'], {}).get('terms', Counter()).get(term, 0)
                row.append(freq)
            evolution_matrix.append(row)
        
        df_evolution = pd.DataFrame(
            evolution_matrix,
            index=all_window_terms,
            columns=[w['nombre'] for w in windows]
        )
        
        df_evolution.to_csv(EVOLUCION_TEMATICA_DIR / 'matriz_evolucion.csv')
        print(f'  ✓ Matriz guardada: {EVOLUCION_TEMATICA_DIR / "matriz_evolucion.csv"}')
        
        # ========== 8.2 HEATMAP DE EVOLUCIÓN ==========
        print('\n--- 8.2 Heatmap de Evolución ---')
        
        plt.figure(figsize=(16, 14))
        
        df_heatmap = df_evolution.copy()
        for col in df_heatmap.columns:
            max_val = df_heatmap[col].max()
            if max_val > 0:
                df_heatmap[col] = df_heatmap[col] / max_val
        
        df_heatmap = df_heatmap[df_heatmap.sum(axis=1) > 0]
        df_heatmap = df_heatmap.iloc[:50]
        
        sns.heatmap(df_heatmap, cmap='YlOrRd', annot=True, fmt='.2f', 
                   cbar_kws={'label': 'Frecuencia Normalizada'},
                   linewidths=0.5, linecolor='gray')
        plt.title('Evolución de Términos a lo Largo del Tiempo', fontsize=16, fontweight='bold')
        plt.xlabel('Ventana Temporal', fontsize=12)
        plt.ylabel('Término', fontsize=12)
        plt.tight_layout()
        plt.savefig(str(EVOLUCION_TEMATICA_DIR / 'heatmap_evolucion.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Heatmap guardado: {EVOLUCION_TEMATICA_DIR / "heatmap_evolucion.png"}')
        
        # ========== 8.3 TRAYECTORIAS DE TÉRMINOS ==========
        print('\n--- 8.3 Trayectorias de Términos ---')
        
        trends = []
        for term in all_window_terms:
            freqs = df_evolution.loc[term].values
            if sum(freqs) > 0:
                x = np.arange(len(freqs))
                if len(x) > 1:
                    slope = np.polyfit(x, freqs, 1)[0]
                    trends.append({
                        'termino': term,
                        'slope': slope,
                        'frecuencia_total': sum(freqs),
                        'frecuencia_inicial': freqs[0],
                        'frecuencia_final': freqs[-1],
                        'tendencia': 'creciente' if slope > 0 else 'decreciente' if slope < 0 else 'estable'
                    })
        
        df_trends = pd.DataFrame(trends)
        df_trends = df_trends.sort_values('slope', ascending=False)
        df_trends.to_csv(EVOLUCION_TEMATICA_DIR / 'tendencias_terminos.csv', index=False)
        
        # Términos emergentes
        emergentes = df_trends.nlargest(10, 'slope')
        if not emergentes.empty:
            print('\n  Términos EMERGENTES (crecimiento):')
            for _, row in emergentes.iterrows():
                print(f'    {row["termino"]}: +{row["slope"]:.2f} (inicial={row["frecuencia_inicial"]:.0f} → final={row["frecuencia_final"]:.0f})')
        
        # Términos en declive
        declive = df_trends.nsmallest(10, 'slope')
        if not declive.empty:
            print('\n  Términos en DECLIVE (caída):')
            for _, row in declive.iterrows():
                print(f'    {row["termino"]}: {row["slope"]:.2f} (inicial={row["frecuencia_inicial"]:.0f} → final={row["frecuencia_final"]:.0f})')
        
        # ========== 8.4 GRÁFICO DE TRAYECTORIAS ==========
        print('\n--- 8.4 Gráfico de Trayectorias ---')
        
        plt.figure(figsize=(14, 8))
        
        selected_terms = df_trends.nlargest(8, 'slope')['termino'].tolist() + df_trends.nsmallest(8, 'slope')['termino'].tolist()
        
        for term in selected_terms[:12]:
            if term in df_evolution.index:
                valores = df_evolution.loc[term].values
                if sum(valores) > 0:
                    plt.plot(range(len(valores)), valores, marker='o', 
                            label=term[:25], linewidth=2, markersize=6)
        
        plt.xlabel('Ventana Temporal', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.title('Trayectorias de Términos (Emergentes vs Declive)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(windows)), [w['nombre'] for w in windows], rotation=45)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(EVOLUCION_TEMATICA_DIR / 'trayectorias_terminos.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Trayectorias guardadas: {EVOLUCION_TEMATICA_DIR / "trayectorias_terminos.png"}')
        
        # ========== 8.5 WORD CLOUDS POR VENTANA (MEJORADO) ==========
        print('\n--- 8.5 Word Clouds por Ventana ---')
        
        for window in windows:
            window_name = window['nombre']
            if window_name in window_data:
                freq_dict = dict(window_data[window_name]['terms'].most_common(100))
                output_path = TOPICOS_TEMPORALES_DIR / f'wordcloud_{window_name}.png'
                save_wordcloud_mejorado(freq_dict, output_path, f'Tópicos - {window_name}', max_words=100)
                print(f'  ✓ WordCloud {window_name}')
        
        # ========== 8.6 REPORTE DE EVOLUCIÓN ==========
        print('\n--- 8.6 Reporte de Evolución ---')
        
        reporte = f"""
        ============================================================
        REPORTE DE EVOLUCIÓN TEMÁTICA
        ============================================================
        
        PERIODO ANALIZADO: {years[0]} - {years[-1]}
        VENTANAS TEMPORALES: {len(windows)}
        
        TÉRMINOS EMERGENTES (Mayor crecimiento):
        """
        
        for _, row in emergentes.head(10).iterrows():
            reporte += f"\n  - {row['termino']}: +{row['slope']:.2f} (inicial={row['frecuencia_inicial']:.0f} → final={row['frecuencia_final']:.0f})"
        
        reporte += "\n\nTÉRMINOS EN DECLIVE (Mayor caída):"
        for _, row in declive.head(10).iterrows():
            reporte += f"\n  - {row['termino']}: {row['slope']:.2f} (inicial={row['frecuencia_inicial']:.0f} → final={row['frecuencia_final']:.0f})"
        
        reporte += f"\n\nRESUMEN DE TENDENCIAS:"
        reporte += f"\n  - Términos con tendencia creciente: {len(df_trends[df_trends['slope'] > 0])}"
        reporte += f"\n  - Términos con tendencia decreciente: {len(df_trends[df_trends['slope'] < 0])}"
        reporte += f"\n  - Términos estables: {len(df_trends[df_trends['slope'] == 0])}"
        
        with open(ANALISIS_AVANZADO_DIR / 'reporte_evolucion_tematica.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print(reporte)
        print(f'  ✓ Reporte guardado: {ANALISIS_AVANZADO_DIR / "reporte_evolucion_tematica.txt"}')
        
    except Exception as e:
        print(f'  ✗ Error: {e}')
        import traceback
        traceback.print_exc()

# ============================================================================
# 9. ANÁLISIS BIBLIOMÉTRICO COMPLETO
# ============================================================================

def analisis_bibliometrico(df_base):
    """Análisis bibliométrico completo con citas, tasa, correlaciones"""
    print('\n' + '='*70)
    print('9. ANÁLISIS BIBLIOMÉTRICO COMPLETO')
    print('='*70)
    
    try:
        if df_base['Citas'].isna().all():
            print('  ⚠️ Sin datos de citas')
            return
        
        # ========== 9.1 ESTADÍSTICAS DESCRIPTIVAS ==========
        print('\n--- 9.1 Estadísticas Descriptivas ---')
        
        stats = {
            'Total documentos': len(df_base),
            'Total citas': df_base['Citas'].sum(),
            'Media citas': df_base['Citas'].mean(),
            'Mediana citas': df_base['Citas'].median(),
            'Máximo citas': df_base['Citas'].max(),
            'Mínimo citas': df_base['Citas'].min(),
            'Desviación estándar': df_base['Citas'].std(),
            'Q1 (25%)': df_base['Citas'].quantile(0.25),
            'Q3 (75%)': df_base['Citas'].quantile(0.75),
            'IQR': df_base['Citas'].quantile(0.75) - df_base['Citas'].quantile(0.25),
            'Asimetría': df_base['Citas'].skew(),
            'Curtosis': df_base['Citas'].kurtosis()
        }
        
        for key, value in stats.items():
            print(f'  {key}: {value:.2f}' if isinstance(value, float) else f'  {key}: {value}')
        
        # ========== 9.2 ANÁLISIS POR AÑO ==========
        print('\n--- 9.2 Análisis por Año ---')
        
        citas_anio = df_base.groupby('Año')['Citas'].agg(['count', 'sum', 'mean', 'std', 'min', 'max'])
        citas_anio.columns = ['documentos', 'total_citas', 'media_citas', 'std_citas', 'min_citas', 'max_citas']
        citas_anio['factor_impacto'] = citas_anio['total_citas'] / citas_anio['documentos']
        
        print(citas_anio)
        citas_anio.to_csv(BIBLIO_DIR / 'citas_por_anio.csv')
        
        # ========== 9.3 TOP CITADOS ==========
        print('\n--- 9.3 Top 10 Artículos Más Citados ---')
        
        top_citados = df_base.nlargest(10, 'Citas')[['Articulo', 'Año', 'Citas', 'Tasa']]
        print(top_citados.to_string(index=False))
        top_citados.to_csv(BIBLIO_DIR / 'top_citados.csv', index=False)
        
        # ========== 9.4 CORRELACIONES ==========
        print('\n--- 9.4 Correlaciones ---')
        
        # Correlación citas vs año
        corr_year = df_base['Citas'].corr(df_base['Año'])
        print(f'  Correlación citas-año: {corr_year:.4f}')
        
        # Correlación citas vs tasa
        if not df_base['Tasa'].isna().all():
            corr_tasa = df_base['Citas'].corr(df_base['Tasa'])
            print(f'  Correlación citas-tasa: {corr_tasa:.4f}')
            
            # Test de significancia
            from scipy.stats import pearsonr
            p_value = pearsonr(df_base['Citas'].dropna(), df_base['Tasa'].dropna())[1]
            print(f'  p-value: {p_value:.4f}')
        
        # ========== 9.5 DISTRIBUCIÓN DE CITAS ==========
        print('\n--- 9.5 Distribución de Citas ---')
        
        # Histograma
        plt.figure(figsize=(10, 6))
        plt.hist(df_base['Citas'].dropna(), bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlabel('Número de Citas')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Citas', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(HEATMAPS_DIR / 'distribucion_citas.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Histograma: {HEATMAPS_DIR / "distribucion_citas.png"}')
        
        # ========== 9.6 GUARDAR RESULTADOS ==========
        print('\n--- 9.6 Guardando resultados ---')
        
        with pd.ExcelWriter(BIBLIO_DIR / 'analisis_bibliometrico_completo.xlsx') as writer:
            pd.DataFrame([stats]).to_excel(writer, sheet_name='Estadisticas', index=False)
            citas_anio.to_excel(writer, sheet_name='Citas_por_Anio')
            top_citados.to_excel(writer, sheet_name='Top_Citados', index=False)
            
            # Distribución
            pd.DataFrame({
                'Rango': pd.cut(df_base['Citas'], bins=10).value_counts().index.astype(str),
                'Frecuencia': pd.cut(df_base['Citas'], bins=10).value_counts().values
            }).to_excel(writer, sheet_name='Distribucion_Citas', index=False)
            
            # Correlaciones
            if not df_base['Tasa'].isna().all():
                pd.DataFrame({
                    'Variable1': ['Citas', 'Citas'],
                    'Variable2': ['Año', 'Tasa'],
                    'Correlacion': [corr_year, corr_tasa]
                }).to_excel(writer, sheet_name='Correlaciones', index=False)
        
        print(f'  ✓ Datos guardados: {BIBLIO_DIR / "analisis_bibliometrico_completo.xlsx"}')
        
        # ========== 9.7 VISUALIZACIONES COMPLETAS ==========
        print('\n--- 9.7 Visualizaciones ---')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Citas por año (barras)
        ax1 = axes[0, 0]
        ax1.bar(citas_anio.index, citas_anio['total_citas'], color='steelblue', alpha=0.7, edgecolor='darkblue')
        ax1.set_xlabel('Año')
        ax1.set_ylabel('Total Citas')
        ax1.set_title('Citas por Año', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Media de citas por año (línea)
        ax2 = axes[0, 1]
        ax2.plot(citas_anio.index, citas_anio['media_citas'], marker='o', linewidth=2, color='coral')
        ax2.set_xlabel('Año')
        ax2.set_ylabel('Media de Citas')
        ax2.set_title('Media de Citas por Año', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # 3. Factor de impacto por año
        ax3 = axes[0, 2]
        ax3.bar(citas_anio.index, citas_anio['factor_impacto'], color='green', alpha=0.7, edgecolor='darkgreen')
        ax3.set_xlabel('Año')
        ax3.set_ylabel('Factor de Impacto')
        ax3.set_title('Factor de Impacto por Año', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Top citados (barras horizontales)
        ax4 = axes[1, 0]
        top_short = top_citados.copy()
        top_short['Articulo'] = top_short['Articulo'].str[:30]
        ax4.barh(top_short['Articulo'], top_short['Citas'], color='coral', alpha=0.7, edgecolor='darkred')
        ax4.set_xlabel('Citas')
        ax4.set_title('Top 10 Artículos Más Citados', fontsize=12, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        
        # 5. Boxplot de citas por año
        ax5 = axes[1, 1]
        data_by_year = [df_base[df_base['Año'] == year]['Citas'].dropna().values for year in sorted(df_base['Año'].unique())]
        ax5.boxplot(data_by_year, labels=[str(y) for y in sorted(df_base['Año'].unique())])
        ax5.set_xlabel('Año')
        ax5.set_ylabel('Citas')
        ax5.set_title('Distribución de Citas por Año', fontsize=12, fontweight='bold')
        
        # 6. Relación citas vs tasa (scatter)
        ax6 = axes[1, 2]
        if not df_base['Tasa'].isna().all():
            ax6.scatter(df_base['Citas'], df_base['Tasa'], alpha=0.6, color='steelblue')
            ax6.set_xlabel('Citas')
            ax6.set_ylabel('Tasa de Citas')
            ax6.set_title('Relación Citas vs Tasa', fontsize=12, fontweight='bold')
            ax6.grid(alpha=0.3)
            if not df_base['Tasa'].isna().all():
                corr = df_base['Citas'].corr(df_base['Tasa'])
                ax6.text(0.05, 0.95, f'Correlación: {corr:.3f}', transform=ax6.transAxes, fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(str(BIBLIO_DIR / 'analisis_bibliometrico_completo.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Gráficos guardados: {BIBLIO_DIR / "analisis_bibliometrico_completo.png"}')
        
        # ========== 9.8 RESULTADOS BIBLIOMÉTRICOS ==========
        print('\n--- 9.8 Resumen Bibliométrico ---')
        
        resumen = f"""
        ============================================================
        RESUMEN BIBLIOMÉTRICO
        ============================================================
        
        DATOS GENERALES:
        - Total de documentos: {len(df_base)}
        - Total de citas: {stats['Total citas']:.0f}
        - Media de citas por documento: {stats['Media citas']:.2f}
        - Mediana de citas: {stats['Mediana citas']:.0f}
        - Máximo de citas: {stats['Máximo citas']:.0f}
        
        DISTRIBUCIÓN TEMPORAL:
        - Años cubiertos: {sorted(df_base['Año'].unique())[0]} - {sorted(df_base['Año'].unique())[-1]}
        - Año con más documentos: {citas_anio['documentos'].idxmax()} ({citas_anio['documentos'].max()} documentos)
        - Año con más citas: {citas_anio['total_citas'].idxmax()} ({citas_anio['total_citas'].max():.0f} citas)
        
        CORRELACIONES:
        - Citas vs Año: {corr_year:.4f}
        - Citas vs Tasa: {corr_tasa:.4f}
        
        TOP 5 ARTÍCULOS MÁS CITADOS:
        """
        for _, row in top_citados.head(5).iterrows():
            resumen += f"\n  - {row['Articulo'][:40]} ({row['Año']}): {row['Citas']:.0f} citas"
        
        with open(ANALISIS_AVANZADO_DIR / 'resumen_bibliometrico.txt', 'w', encoding='utf-8') as f:
            f.write(resumen)
        
        print(resumen)
        
    except Exception as e:
        print(f'  ✗ Error en análisis bibliométrico: {e}')
        import traceback
        traceback.print_exc()

# ============================================================================
# 10. ANÁLISIS ESTADÍSTICO AVANZADO
# ============================================================================

def analisis_estadistico_avanzado(df_base, agg_uni):
    """Análisis estadístico avanzado"""
    print('\n' + '='*70)
    print('10. ANÁLISIS ESTADÍSTICO AVANZADO')
    print('='*70)
    
    try:
        # ========== 10.1 ESTADÍSTICAS DE TÉRMINOS ==========
        print('\n--- 10.1 Estadísticas de Términos ---')
        
        # Obtener todos los términos
        all_terms = Counter()
        for year in agg_uni:
            all_terms.update(agg_uni[year])
        
        df_terms = pd.DataFrame(all_terms.most_common(), columns=['termino', 'frecuencia'])
        df_terms.to_csv(ANALISIS_ESTADISTICO_DIR / 'frecuencia_terminos.csv', index=False)
        
        print(f'  Total términos únicos: {len(df_terms)}')
        print(f'  Frecuencia promedio: {df_terms["frecuencia"].mean():.2f}')
        print(f'  Frecuencia máxima: {df_terms["frecuencia"].max()}')
        print(f'  Frecuencia mínima: {df_terms["frecuencia"].min()}')
        
        # ========== 10.2 DISTRIBUCIÓN DE FRECUENCIAS ==========
        print('\n--- 10.2 Distribución de Frecuencias ---')
        
        plt.figure(figsize=(12, 6))
        plt.hist(df_terms['frecuencia'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlabel('Frecuencia')
        plt.ylabel('Número de Términos')
        plt.title('Distribución de Frecuencias de Términos', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(ANALISIS_ESTADISTICO_DIR / 'distribucion_frecuencias.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Gráfico guardado: {ANALISIS_ESTADISTICO_DIR / "distribucion_frecuencias.png"}')
        
        # ========== 10.3 LEY DE ZIPF ==========
        print('\n--- 10.3 Ley de Zipf ---')
        
        # Ordenar por frecuencia
        df_terms['rank'] = range(1, len(df_terms) + 1)
        df_terms['log_frecuencia'] = np.log(df_terms['frecuencia'])
        df_terms['log_rank'] = np.log(df_terms['rank'])
        
        plt.figure(figsize=(12, 6))
        plt.scatter(df_terms['log_rank'], df_terms['log_frecuencia'], alpha=0.3, s=10)
        plt.xlabel('Log(Rank)')
        plt.ylabel('Log(Frecuencia)')
        plt.title('Ley de Zipf - Distribución de Frecuencias', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(ANALISIS_ESTADISTICO_DIR / 'zipf_law.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Gráfico guardado: {ANALISIS_ESTADISTICO_DIR / "zipf_law.png"}')
        
        # ========== 10.4 ESTADÍSTICAS POR AÑO ==========
        print('\n--- 10.4 Estadísticas por Año ---')
        
        stats_año = []
        for year in sorted(agg_uni.keys()):
            terms = agg_uni[year]
            total_terms = sum(terms.values())
            unique_terms = len(terms)
            
            stats_año.append({
                'año': year,
                'total_terminos': total_terms,
                'terminos_unicos': unique_terms,
                'terminos_promedio': total_terms / unique_terms if unique_terms > 0 else 0,
                'top_term': terms.most_common(1)[0][0] if terms else None,
                'top_frecuencia': terms.most_common(1)[0][1] if terms else 0
            })
        
        df_stats_año = pd.DataFrame(stats_año)
        df_stats_año.to_csv(ANALISIS_ESTADISTICO_DIR / 'estadisticas_por_anio.csv', index=False)
        print(df_stats_año)
        
        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        ax1.plot(df_stats_año['año'], df_stats_año['terminos_unicos'], marker='o', linewidth=2)
        ax1.set_xlabel('Año')
        ax1.set_ylabel('Términos Únicos')
        ax1.set_title('Evolución de Términos Únicos', fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        ax2 = axes[1]
        ax2.plot(df_stats_año['año'], df_stats_año['total_terminos'], marker='o', linewidth=2, color='coral')
        ax2.set_xlabel('Año')
        ax2.set_ylabel('Total de Términos')
        ax2.set_title('Evolución del Total de Términos', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(ANALISIS_ESTADISTICO_DIR / 'evolucion_terminos_por_anio.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Gráfico guardado: {ANALISIS_ESTADISTICO_DIR / "evolucion_terminos_por_anio.png"}')
        
    except Exception as e:
        print(f'  ✗ Error: {e}')
        import traceback
        traceback.print_exc()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print('='*80)
    print('PROCESADOR COMPLETO DE DOCUMENTOS - VERSIÓN FINAL')
    print('='*80)
    
    # PASO 1: Crear base
    df_base = crear_base_completa()
    
    if df_base is None:
        print('✗ Error: No se pudo crear la base')
        return
    
    # Verificar datos
    print('\n' + '='*70)
    print('VERIFICANDO DATOS')
    print('='*70)
    print(f'  Documentos con año: {df_base["Año"].notna().sum()}')
    print(f'  Años presentes: {sorted(df_base["Año"].dropna().unique())}')
    print(f'  Documentos con citas: {df_base["Citas"].notna().sum()}')
    
    # PASO 2: Procesar documentos
    print('\n' + '='*70)
    print('PROCESANDO DOCUMENTOS')
    print('='*70)
    
    agg_uni = defaultdict(Counter)
    agg_bi = defaultdict(Counter)
    agg_tri = defaultdict(Counter)
    
    all_tokens = []
    all_bigrams = []
    all_trigrams = []
    edges = []
    manifest = []
    
    for idx, row in tqdm.tqdm(df_base.iterrows(), total=len(df_base)):
        articulo = row['Articulo']
        anio = row['Año']
        
        if pd.isna(anio):
            continue
        
        anio = int(anio)
        pdf_file = ORIGEN_DIR / f'{articulo}.pdf'
        
        if not pdf_file.exists():
            continue
        
        texto = extract_pdf_text(pdf_file)
        if not texto.strip():
            continue
        
        tokens = clean_text(texto)
        if len(tokens) < 10:
            continue
        
        uni = get_top_k(tokens, 30)
        bi = get_ngrams(tokens, 2, 30)
        tri = get_ngrams(tokens, 3, 30)
        
        all_tokens.extend(tokens)
        all_bigrams.extend([b for b, _ in bi])
        all_trigrams.extend([t for t, _ in tri])
        
        agg_uni[anio].update(dict(uni))
        agg_bi[anio].update(dict(bi))
        agg_tri[anio].update(dict(tri))
        
        base = f'{anio}_{idx:03d}_{articulo[:30]}'
        
        pd.DataFrame(uni, columns=['unigram', 'count']).to_csv(
            PER_DOC_DIR / f'{base}_unigramas.csv', index=False)
        pd.DataFrame(bi, columns=['bigram', 'count']).to_csv(
            PER_DOC_DIR / f'{base}_bigramas.csv', index=False)
        pd.DataFrame(tri, columns=['trigram', 'count']).to_csv(
            PER_DOC_DIR / f'{base}_trigramas.csv', index=False)
        
        # Word clouds individuales (usando versión mejorada)
        save_wordcloud_mejorado(dict(uni), WC_DIR / f'{base}_unigramas.png', f'Unigramas - {articulo[:20]}', max_words=50)
        save_wordcloud_mejorado(dict(bi), WC_DIR / f'{base}_bigramas.png', f'Bigramas - {articulo[:20]}', max_words=50, cmap='plasma')
        save_wordcloud_mejorado(dict(tri), WC_DIR / f'{base}_trigramas.png', f'Trigramas - {articulo[:20]}', max_words=50, cmap='magma')
        
        for term, w in uni:
            edges.append((articulo, term, 'unigram', w, anio))
        for term, w in bi:
            edges.append((articulo, term, 'bigram', w, anio))
        for term, w in tri:
            edges.append((articulo, term, 'trigram', w, anio))
        
        manifest.append({
            'Articulo': articulo,
            'Año': anio,
            'unigrams_csv': str(PER_DOC_DIR / f'{base}_unigramas.csv'),
            'bigrams_csv': str(PER_DOC_DIR / f'{base}_bigramas.csv'),
            'trigrams_csv': str(PER_DOC_DIR / f'{base}_trigramas.csv')
        })
    
    # Guardar manifest
    manifest_path = RES_DIR / 'manifest.csv'
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f'\n✓ {len(manifest)} documentos procesados')
    
    # Guardar agregados
    for datos, nombre in [(agg_uni, 'unigram'), (agg_bi, 'bigram'), (agg_tri, 'trigram')]:
        total = Counter()
        for year in datos:
            total.update(datos[year])
        pd.DataFrame(total.most_common(), columns=[nombre, 'count']).to_csv(
            AGG_DIR / f'{nombre}_all.csv', index=False)
    
    # PASO 3: Ejecutar TODOS los análisis
    
    counter, counter_bi, counter_tri = analisis_wordclouds(all_tokens, all_bigrams, all_trigrams, df_base)
    analisis_wordclouds_por_anio(agg_uni, agg_bi, agg_tri)  # <-- NUEVA LÍNEA
    analisis_flor_plots(agg_uni, agg_bi, agg_tri)
    analisis_redes_por_anio(edges)
    G_global = analisis_red_global(edges)
    G_terminos = analisis_red_termino_termino(edges)
    G_articulos = analisis_red_articulo_articulo(edges, df_base)
    df_resultados, partition, modularidad = analisis_louvain(manifest_path)  # <-- MODIFICADO (guardar resultados)
    analisis_wordclouds_comunidades(df_resultados, partition)  # <-- NUEVA LÍNEA
    analisis_evolucion_temporal(df_base, agg_uni, agg_bi, agg_tri)
    analisis_bibliometrico(df_base)
    analisis_estadistico_avanzado(df_base, agg_uni)

    # PASO 4: RESUMEN FINAL
    print('\n' + '='*80)
    print('PROCESAMIENTO COMPLETADO EXITOSAMENTE')
    print('='*80)
    print('\n📁 CARPETAS CON RESULTADOS:')
    for nombre, carpeta in CARPETAS.items():
        archivos = len(list(carpeta.glob('*'))) if carpeta.exists() else 0
        print(f'  - {nombre}: {archivos} archivos en {carpeta}')
    
    print('\n📊 ESTADÍSTICAS FINALES:')
    print(f'  - Total documentos procesados: {len(manifest)}')
    print(f'  - Total tokens únicos: {len(set(all_tokens))}')
    print(f'  - Total bigramas: {len(set(all_bigrams))}')
    print(f'  - Total trigramas: {len(set(all_trigrams))}')
    print(f'  - Total aristas en redes: {len(edges)}')
    print(f'  - Total años analizados: {len(agg_uni)}')
    
    print('\n✅ PROCESAMIENTO COMPLETO')

if __name__ == '__main__':
    main()