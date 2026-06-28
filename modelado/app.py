import os
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Configuración de carpetas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS_DIR = os.path.join(BASE_DIR, 'datos')
RESULTADOS_DIR = BASE_DIR

# Crear carpeta de resultados si no existe
os.makedirs(RESULTADOS_DIR, exist_ok=True)

def cargar_datasets():
    datasets = [
        {'file': 'dataset_procesado_tiktok_parte01.csv', 'origen': 'tiktok_01'},
        {'file': 'dataset_procesado_youtube_parte01.csv', 'origen': 'youtube_01'},
        {'file': 'dataset_procesado_youtube_parte02.csv', 'origen': 'youtube_02'}
    ]
    
    dfs = []
    for ds in datasets:
        file_path = os.path.join(DATOS_DIR, ds['file'])
        if os.path.exists(file_path):
            print(f"   Cargando {ds['file']}...")
            df = pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip')
            
            # Limpiar nombres de columnas (eliminar espacios)
            df.columns = df.columns.str.strip().str.lower()
            
            # Verificar que exista la columna emocion
            if 'emocion' not in df.columns:
                print(f"No se encontró la columna 'emocion' en {ds['file']}. Columnas: {df.columns}")
                
            df['origen'] = ds['origen']
            dfs.append(df)
        else:
            print(f"No se encontró {file_path}")
            
    if not dfs:
        raise ValueError("No se encontraron datasets para cargar.")
    
    # Concatenar los datasets
    df_final = pd.concat(dfs, ignore_index=True)
    
    # Limpieza de la columna 'emocion'
    if 'emocion' in df_final.columns:
        df_final['emocion'] = df_final['emocion'].astype(str).str.strip()
        # Estandarizar 'Alegria' a 'Alegría' (considerando posibles problemas de codificación previos)
        df_final['emocion'] = df_final['emocion'].replace({'Alegria': 'Alegría'})
        
        # Validar las emociones según Ekman
        emociones_validas = ['Tristeza', 'Alegría', 'Sorpresa', 'Miedo', 'Asco', 'Ira']
        df_final = df_final[df_final['emocion'].isin(emociones_validas)].copy()
        
    return df_final

def dividir_datos(df, columna_texto='lemas', columna_etiqueta='emocion',
                  test_size=0.20, val_size=0.30, random_state=42):
    """
    Divide el dataset en 3 conjuntos estratificados:
      - Entrenamiento efectivo: ~56% del total (70% del 80%)
      - Validación:             ~24% del total (30% del 80%)
      - Prueba y Evaluación:    20% del total

    Parámetros
    df             : DataFrame con los datos
    columna_texto  : Nombre de la columna con los textos/lemas
    columna_etiqueta: Nombre de la columna con las etiquetas de emoción
    test_size      : Proporción del total reservada para prueba (0.20)
    val_size       : Proporción del 80% restante para validación (0.30)
    random_state   : Semilla para reproducibilidad

    Retorna
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = df.dropna(subset=[columna_texto, columna_etiqueta]).copy()
    df[columna_texto] = df[columna_texto].astype(str)

    X = df[columna_texto]
    y = df[columna_etiqueta]

    # 1er corte: 80% temporal + 20% prueba final
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2do corte: 70% entrenamiento efectivo + 30% validación (sobre el 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def visualizar_datos(df):
    print("\nResumen del dataset combinado:")
    print(f"Total de registros: {len(df)}")
    if 'origen' in df.columns:
        print("\nRegistros por origen:")
        print(df['origen'].value_counts())
    
    if 'emocion' in df.columns:
        print("\nDistribución de emociones:")
        print(df['emocion'].value_counts())
        
        # Estilo de seaborn
        sns.set_theme(style="whitegrid")
        
        # Distribución general de emociones
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='emocion', order=df['emocion'].value_counts().index, palette='viridis')
        plt.title('Distribución General de Emociones')
        plt.xlabel('Emoción')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=45)
        plt.tight_layout()
        ruta_grafico1 = os.path.join(RESULTADOS_DIR, 'distribucion_emociones.png')
        plt.savefig(ruta_grafico1, dpi=300)
        print(f"\nGráfico guardado en: {ruta_grafico1}")
        plt.close()
        
        # Distribución de emociones por origen
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='emocion', hue='origen', order=df['emocion'].value_counts().index, palette='Set2')
        plt.title('Distribución de Emociones por Origen de Datos')
        plt.xlabel('Emoción')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=45)
        plt.legend(title='Origen', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        ruta_grafico2 = os.path.join(RESULTADOS_DIR, 'distribucion_emociones_por_origen.png')
        plt.savefig(ruta_grafico2, dpi=300)
        print(f"Gráfico guardado en: {ruta_grafico2}")
        plt.close()
    else:
        print("No se pudo generar las gráficas por la falta de la columna 'emocion'.")

def visualizar_distribucion_conjuntos(df, columna_texto='lemas', columna_etiqueta='emocion'):
    """
    Aplica la división 80/20 (con validación interna 70/30) y muestra:
      1. Tabla Markdown con la distribución de emociones por conjunto → distribucion_conjuntos.md
      2. Gráfica de barras comparativa de los 3 conjuntos              → distribucion_conjuntos.png
      3. Gráfica de pastel (pie chart) por cada conjunto               → distribucion_pie_conjuntos.png
    Todas las salidas se exportan en RESULTADOS_DIR (modelado/).
    """
    if columna_etiqueta not in df.columns:
        print(f"No se encontró la columna '{columna_etiqueta}'. No se puede visualizar la distribución.")
        return

    # Filtrar emociones válidas (las 4 clases del proyecto)
    emociones_validas = ['Alegría', 'Tristeza', 'Miedo', 'Sorpresa']
    df_filtrado = df[df[columna_etiqueta].isin(emociones_validas)].copy()

    if columna_texto not in df_filtrado.columns:
        print(f"Columna '{columna_texto}' no encontrada. Usando índice como texto temporal.")
        df_filtrado[columna_texto] = df_filtrado.index.astype(str)

    print(f"\nTotal de registros (con las 4 emociones): {len(df_filtrado)}")

    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(
        df_filtrado, columna_texto=columna_texto, columna_etiqueta=columna_etiqueta
    )

    total = len(df_filtrado)
    conjuntos = {
        f'Entrenamiento Efectivo\n(~{len(X_train)/total*100:.1f}% del total)': y_train,
        f'Validación\n(~{len(X_val)/total*100:.1f}% del total)':              y_val,
        f'Prueba y Evaluación\n(~{len(X_test)/total*100:.1f}% del total)':    y_test,
    }

    # Tabla en consola y exportación Markdown 
    print("\n" + "="*65)
    print("  DISTRIBUCIÓN DE EMOCIONES POR CONJUNTO")
    print("="*65)

    lineas_md = [
        "# Distribución de Datos por Conjunto\n",
        f"> Dataset total (4 emociones): **{total} registros**\n",
        "> División: **80% Entrenamiento** (70% efectivo + 30% validación) / **20% Prueba**\n\n",
    ]

    nombres_cortos = ['Entrenamiento Efectivo', 'Validación', 'Prueba y Evaluación']
    porcentajes    = [
        f'~{len(X_train)/total*100:.1f}% del total',
        f'~{len(X_val)/total*100:.1f}% del total',
        f'~{len(X_test)/total*100:.1f}% del total',
    ]
    series_list = [y_train, y_val, y_test]

    for nombre_corto, pct, serie in zip(nombres_cortos, porcentajes, series_list):
        conteo = serie.value_counts().reindex(emociones_validas, fill_value=0)
        print(f"\n  {nombre_corto} ({pct}) — {len(serie)} registros")
        print(f"  {'Emoción':<12} {'Registros':>10}")
        print(f"  {'-'*24}")
        for emocion, cantidad in conteo.items():
            print(f"  {emocion:<12} {cantidad:>10}")
        print(f"  {'TOTAL':<12} {len(serie):>10}")

        lineas_md.append(f"## {nombre_corto} ({pct})\n\n")
        lineas_md.append("| Emoción | Registros | Porcentaje |\n")
        lineas_md.append("|---------|-----------|------------|\n")
        for emocion, cantidad in conteo.items():
            pct_clase = cantidad / len(serie) * 100 if len(serie) > 0 else 0
            lineas_md.append(f"| {emocion} | {cantidad} | {pct_clase:.1f}% |\n")
        lineas_md.append(f"| **Total** | **{len(serie)}** | **100%** |\n\n")

    print("\n" + "="*65)

    # Exportar Markdown
    ruta_md = os.path.join(RESULTADOS_DIR, 'distribucion_conjuntos.md')
    with open(ruta_md, 'w', encoding='utf-8') as f:
        f.writelines(lineas_md)
    print(f"\nTabla de distribución exportada en: {ruta_md}")

    # Gráfica de barras agrupadas por conjunto
    sns.set_theme(style="whitegrid")
    PALETA = {'Alegría': '#4C72B0', 'Tristeza': '#DD8452', 'Miedo': '#55A868', 'Sorpresa': '#C44E52'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
    fig.suptitle('Distribución de Emociones por Conjunto de Datos', fontsize=15, fontweight='bold', y=1.02)

    titulos = ['Entrenamiento Efectivo', 'Validación', 'Prueba y Evaluación']
    subtitulos = [
        f'{len(X_train)} registros (~{len(X_train)/total*100:.1f}%)',
        f'{len(X_val)} registros (~{len(X_val)/total*100:.1f}%)',
        f'{len(X_test)} registros (~{len(X_test)/total*100:.1f}%)',
    ]

    for ax, titulo, subtitulo, serie in zip(axes, titulos, subtitulos, series_list):
        conteo = serie.value_counts().reindex(emociones_validas, fill_value=0)
        colores = [PALETA.get(e, '#999') for e in conteo.index]
        barras = ax.bar(conteo.index, conteo.values, color=colores, edgecolor='white', linewidth=0.8)
        ax.set_title(f'{titulo}\n{subtitulo}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Emoción', fontsize=10)
        ax.set_ylabel('Cantidad de Registros', fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        # Etiquetas sobre las barras
        for barra in barras:
            altura = barra.get_height()
            ax.text(barra.get_x() + barra.get_width() / 2., altura + 0.5,
                    f'{int(altura)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylim(0, conteo.max() * 1.20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    ruta_barras = os.path.join(RESULTADOS_DIR, 'distribucion_conjuntos.png')
    plt.savefig(ruta_barras, dpi=300, bbox_inches='tight')
    print(f"Gráfica de barras exportada en: {ruta_barras}")
    plt.close()

    # Gráficas de pastel por conjunto
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('Proporción de Emociones por Conjunto', fontsize=14, fontweight='bold', y=1.02)

    for ax, titulo, subtitulo, serie in zip(axes2, titulos, subtitulos, series_list):
        conteo = serie.value_counts().reindex(emociones_validas, fill_value=0)
        colores = [PALETA.get(e, '#999') for e in conteo.index]
        wedges, texts, autotexts = ax.pie(
            conteo.values,
            labels=conteo.index,
            autopct='%1.1f%%',
            colors=colores,
            startangle=90,
            pctdistance=0.80,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        for at in autotexts:
            at.set_fontsize(9)
        ax.set_title(f'{titulo}\n{subtitulo}', fontsize=10, fontweight='bold')

    plt.tight_layout()
    ruta_pie = os.path.join(RESULTADOS_DIR, 'distribucion_pie_conjuntos.png')
    plt.savefig(ruta_pie, dpi=300, bbox_inches='tight')
    print(f"Gráfica de pastel exportada en: {ruta_pie}")
    plt.close()

if __name__ == '__main__':
    print("Iniciando análisis y visualización de datos...")
    try:
        df = cargar_datasets()
        visualizar_datos(df)
        print("\nDistribución por Conjuntos de Entrenamiento/Validación/Prueba")
        visualizar_distribucion_conjuntos(df)
        print("\nProceso de visualización completado con éxito.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")
        raise
