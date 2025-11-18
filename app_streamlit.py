import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import ast
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import glob

st.set_page_config(page_title="Análisis de Confesiones", layout="wide", page_icon="💬")

@st.cache_data
def load_data():
    # Leer y unir todos los CSV de la carpeta splits_nlp5
    csv_files = glob.glob('../splits_nlp5/df_final_nlp5_part*.csv')
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    df_comentarios = pd.read_excel('comentarios_sentimientos.xlsx')
    return df, df_comentarios

df, df_comentarios = load_data()

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
if 'fecha' in df.columns:
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

fecha_col = 'date' if 'date' in df.columns else 'fecha' if 'fecha' in df.columns else None

st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a:", ["🔍 Buscador", "📊 Estadísticas"])

if page == "📊 Estadísticas":
    st.title("📊 Estadísticas y Análisis")
    st.markdown("---")
    
    st.subheader("📅 Confesiones por Fecha")
    
    if fecha_col:
        df_con_fecha = df[df[fecha_col].notna()].copy()
        
        confesiones_por_fecha = df_con_fecha.groupby(
            df_con_fecha[fecha_col].dt.date
        ).size().reset_index(name='cantidad')
        confesiones_por_fecha.columns = ['fecha', 'cantidad']
        
        fig1 = px.line(confesiones_por_fecha, x='fecha', y='cantidad',
                      title='Evolución de Confesiones en el Tiempo',
                      labels={'fecha': 'Fecha', 'cantidad': 'Número de Confesiones'})
        fig1.update_traces(mode='lines+markers')
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Confesiones", len(df))
        with col2:
            promedio_dia = confesiones_por_fecha['cantidad'].mean()
            st.metric("Promedio por día", f"{promedio_dia:.1f}")
        with col3:
            max_dia = confesiones_por_fecha['cantidad'].max()
            st.metric("Máximo en un día", int(max_dia))
    else:
        st.info("No hay información de fechas disponible")
    
    st.markdown("---")
    
    st.subheader("📝 Usuarios con más comentarios")
    
    user_col = None
    for col in ['users', 'user_name', 'usuario', 'alumno', 'nombre', 'author']:
        if col in df.columns:
            user_col = col
            break
    
    if user_col:
        user_confession_count = Counter()
        
        for idx, row in df.iterrows():
            users_data = row[user_col]
            
            if isinstance(users_data, str):
                try:
                    users_list = ast.literal_eval(users_data)
                except:
                    users_list = [users_data]
            elif isinstance(users_data, list):
                users_list = users_data
            else:
                users_list = []
            
            for user in users_list:
                if user and str(user).strip():
                    user_confession_count[str(user).strip()] += 1
        
        confesiones_por_usuario = pd.DataFrame(
            list(user_confession_count.items()), 
            columns=['usuario', 'num_confesiones']
        ).sort_values('num_confesiones', ascending=False).head(20)
        
        fig_confes = px.bar(confesiones_por_usuario, x='usuario', y='num_confesiones',
                     title='Top 20 Usuarios con más Confesiones',
                     labels={'usuario': 'Usuario', 'num_confesiones': 'Número de Comentarios'},
                     color='num_confesiones')
        fig_confes.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_confes, use_container_width=True)
        
        st.subheader("📋 Detalle Top 10")
        st.dataframe(confesiones_por_usuario.head(10), use_container_width=True)
    else:
        st.warning("No se encontró columna de usuario/alumno en los datos")
    
    st.markdown("---")
    
    st.subheader("Distribución de Sentimientos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sentimientos en Confesiones**")
        if 'sent_confe_etiqueta' in df.columns:
            sent_confe_dist = df['sent_confe_etiqueta'].value_counts()
            fig3 = px.pie(values=sent_confe_dist.values, names=sent_confe_dist.index,
                         title='Distribución de Sentimientos en Confesiones',
                         color_discrete_map={'NEG': '#2ecc71', 'NEU': '#95a5a6', 'POS': '#e74c3c'})
            st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown("**Sentimientos en Comentarios**")
        sent_coment_dist = df_comentarios['sentimiento_etiqueta'].value_counts()
        fig4 = px.pie(values=sent_coment_dist.values, names=sent_coment_dist.index,
                     title='Distribución de Sentimientos en Comentarios',
                     color_discrete_map={'NEG': '#2ecc71', 'NEU': '#95a5a6', 'POS': '#e74c3c'})
        st.plotly_chart(fig4, use_container_width=True)

else:
    st.title("🔍 Buscador de Confesiones")
    st.markdown("---")

    st.sidebar.header("Filtros")

    if fecha_col:
        fecha_min = df[fecha_col].min()
        fecha_max = df[fecha_col].max()
        
        if pd.notna(fecha_min) and pd.notna(fecha_max):
            fecha_inicio, fecha_fin = st.sidebar.date_input(
                "Rango de fechas",
                value=[fecha_min, fecha_max],
                min_value=fecha_min,
                max_value=fecha_max
            )

    if 'sent_confe_etiqueta' in df.columns:
        sentimiento_options = ['Todos'] + list(df['sent_confe_etiqueta'].dropna().unique())
        sentimiento_filter = st.sidebar.selectbox("Sentimiento de confesión", sentimiento_options)
    else:
        sentimiento_filter = 'Todos'

    cluster_cols = [col for col in df.columns if 'cluster' in col.lower() and 'dbscan' in col.lower()]
    if cluster_cols:
        cluster_col = st.sidebar.selectbox("Columna de cluster", cluster_cols)
        cluster_options = ['Todos'] + sorted([x for x in df[cluster_col].unique() if pd.notna(x)])
        cluster_filter = st.sidebar.selectbox("Cluster DBSCAN", cluster_options)
    else:
        cluster_filter = 'Todos'

    st.subheader("Buscar confesión")
    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input("Buscar por palabras clave", placeholder="Escribe palabras para buscar...")

    with col2:
        search_button = st.button("🔍 Buscar", type="primary")

    df_filtered = df.copy()


    if fecha_col and pd.notna(fecha_min) and pd.notna(fecha_max):
        df_filtered = df_filtered[
            (df_filtered[fecha_col] >= pd.to_datetime(fecha_inicio)) &
            (df_filtered[fecha_col] <= pd.to_datetime(fecha_fin))
        ]

    if sentimiento_filter != 'Todos' and 'sent_confe_etiqueta' in df.columns:
        df_filtered = df_filtered[df_filtered['sent_confe_etiqueta'] == sentimiento_filter]

    if cluster_cols and cluster_filter != 'Todos':
        df_filtered = df_filtered[df_filtered[cluster_col] == cluster_filter]

    if search_query:
        text_cols = ['confe_limpio_sin_stem', 'confe_limpio', 'confesion']
        text_col = None
        for col in text_cols:
            if col in df_filtered.columns:
                text_col = col
                break
        
        if text_col:
            mask = df_filtered[text_col].astype(str).str.contains(search_query, case=False, na=False)
            df_filtered = df_filtered[mask]

    st.markdown("---")
    st.subheader(f"📊 Resultados: {len(df_filtered)} confesiones encontradas")

    if len(df_filtered) > 0:
        df_filtered_reset = df_filtered.reset_index()
        
        def format_option(idx):
            row = df_filtered_reset.loc[idx]
            original_idx = row['index'] if 'index' in row else idx
            fecha_str = row[fecha_col].strftime('%Y-%m-%d') if fecha_col and pd.notna(row.get(fecha_col)) else 'Sin fecha'
            return f"Confesión #{original_idx} - {fecha_str}"
        
        selected_idx = st.selectbox(
            "Selecciona una confesión para ver detalles",
            range(len(df_filtered_reset)),
            format_func=format_option
        )
        
        if selected_idx is not None:
            row = df_filtered_reset.iloc[selected_idx]
            confesion_idx = row['index'] if 'index' in row else selected_idx
            confesion_id = confesion_idx
            
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                # Información de la confesión
                st.markdown("### 💬 Confesión")
                
                if 'texts' in df.columns and pd.notna(row.get('texts')):
                    st.markdown("**📝 Texto Original:**")
                    st.info(row['texts'])
                
                if 'comments' in df.columns and pd.notna(row.get('comments')):
                    st.markdown("**💬 Comentarios Originales (de esta confesión):**")
                    try:
                        if isinstance(row['comments'], str):
                            comentarios_orig = ast.literal_eval(row['comments'])
                        else:
                            comentarios_orig = row['comments']
                        
                        if isinstance(comentarios_orig, list) and len(comentarios_orig) > 0:
                            st.caption(f"Total: {len(comentarios_orig)} comentarios")
                            for i, coment in enumerate(comentarios_orig[:10], 1):
                                st.text(f"{i}. {coment}")
                            if len(comentarios_orig) > 10:
                                st.caption(f"... y {len(comentarios_orig) - 10} comentarios más")
                        else:
                            st.caption("Sin comentarios originales")
                    except:
                        st.text(str(row['comments'])[:500])
                
                st.markdown("---")
                
                text_cols = ['confe_limpio_sin_stem', 'confe_limpio', 'confesion']
                text_col = None
                for col in text_cols:
                    if col in df.columns and pd.notna(row.get(col)):
                        text_col = col
                        break
                
                if text_col:
                    st.markdown(f"**🔄 Texto Procesado ({text_col}):**")
                    with st.expander("Ver texto procesado"):
                        st.text(row[text_col])
                
                st.markdown("**Información:**")
                info_cols = st.columns(3)
                
                with info_cols[0]:
                    if fecha_col and pd.notna(row.get(fecha_col)):
                        st.metric("Fecha", row[fecha_col].strftime('%Y-%m-%d'))
                
                with info_cols[1]:
                    if 'sent_confe_etiqueta' in df.columns:
                        st.metric("Sentimiento", row['sent_confe_etiqueta'])
                
                with info_cols[2]:
                    if cluster_cols:
                        st.metric("Cluster", row[cluster_col])
                
                st.markdown("### 💭 Comentarios con Análisis de Sentimiento")
                
                comentarios_originales_lista = []
                if 'comments' in df.columns and pd.notna(row.get('comments')):
                    try:
                        if isinstance(row['comments'], str):
                            comentarios_originales_lista = ast.literal_eval(row['comments'])
                        else:
                            comentarios_originales_lista = row['comments']
                    except:
                        comentarios_originales_lista = []
                
                comentarios_confe = df_comentarios[df_comentarios['confesion_id'] == confesion_id]
                
                st.caption(f"Comentarios con análisis: {len(comentarios_confe)} | Comentarios originales: {len(comentarios_originales_lista)}")
                
                if len(comentarios_originales_lista) > 0:
                    for i, comentario_orig in enumerate(comentarios_originales_lista):
                        if i < len(comentarios_confe):
                            com = comentarios_confe.iloc[i]
                            sentiment_emoji = {"POS": "😊", "NEU": "😐", "NEG": "😞"}
                            emoji = sentiment_emoji.get(com['sentimiento_etiqueta'], "")
                            etiqueta = com['sentimiento_etiqueta']
                            puntaje = com['sentimiento_puntaje']
                        else:
                            emoji = "❓"
                            etiqueta = "Sin análisis"
                            puntaje = None
                        
                        preview = comentario_orig[:50] if len(comentario_orig) > 50 else comentario_orig
                        
                        with st.expander(f"{emoji} {etiqueta} - {preview}..."):
                            st.write(comentario_orig)
                            if puntaje is not None:
                                st.caption(f"Sentimiento: {etiqueta} (Puntaje: {puntaje})")
                            else:
                                st.caption("Sin análisis de sentimiento disponible")
                else:
                    st.info("Esta confesión no tiene comentarios")
            
            with col_right:
                st.markdown("### 📈 Estadísticas")
                
                if len(comentarios_confe) > 0:
                    st.metric("Total comentarios", len(comentarios_confe))
                    
                    sent_dist = comentarios_confe['sentimiento_etiqueta'].value_counts()
                    st.markdown("**Distribución de sentimientos:**")
                    for sent, count in sent_dist.items():
                        st.write(f"- {sent}: {count}")
                    
                    avg_sent = comentarios_confe['sentimiento_puntaje'].mean()
                    st.metric("Sentimiento promedio", f"{avg_sent:.2f}")
                
                st.markdown("### 🔗 Confesiones similares")
                embed_cols = [col for col in df.columns if 'embed' in col.lower()]
                if not embed_cols:
                    embed_cols = [col for col in df.columns if 'beto' in col.lower() or 'embed' in col.lower() or 'tfidf' in col.lower()]
                
                if embed_cols:
                    embed_col = embed_cols[0]
                    st.caption(f"Usando embeddings de: {embed_col}")
                    
                    try:
                        if pd.notna(row.get(embed_col)):
                            if isinstance(row[embed_col], str):
                                embed_actual = np.array(ast.literal_eval(row[embed_col]))
                            else:
                                embed_actual = np.array(row[embed_col])
                            
                            similitudes = []
                            for idx_other in df_filtered_reset.index:
                                if idx_other != selected_idx:
                                    row_other = df_filtered_reset.iloc[idx_other]
                                    original_idx_other = row_other['index'] if 'index' in row_other else idx_other
                                    
                                    embed_other = row_other[embed_col]
                                    if pd.notna(embed_other):
                                        if isinstance(embed_other, str):
                                            embed_other = np.array(ast.literal_eval(embed_other))
                                        else:
                                            embed_other = np.array(embed_other)
                                        
                                        sim = np.dot(embed_actual, embed_other) / (
                                            np.linalg.norm(embed_actual) * np.linalg.norm(embed_other) + 1e-10
                                        )
                                        similitudes.append((original_idx_other, sim, row_other))
                            
                            similitudes.sort(key=lambda x: x[1], reverse=True)
                            for original_idx_sim, sim, row_sim in similitudes[:5]:
                                text_sim = None
                                if 'texts' in df.columns and pd.notna(row_sim.get('texts')):
                                    text_sim = row_sim['texts']
                                else:
                                    for col in ['confe_limpio_sin_stem', 'confe_limpio', 'confesion']:
                                        if col in df.columns and pd.notna(row_sim.get(col)):
                                            text_sim = row_sim[col]
                                            break
                                
                                if text_sim:
                                    preview_text = str(text_sim)[:200] + "..."
                                    with st.expander(f"Similitud: {sim:.3f} - Confesión #{original_idx_sim}"):
                                        st.write(preview_text)
                                        if 'sent_confe_etiqueta' in df.columns:
                                            st.caption(f"Sentimiento: {row_sim.get('sent_confe_etiqueta', 'N/A')}")
                    except Exception as e:
                        st.warning(f"No se pudieron calcular confesiones similares: {str(e)}")
                else:
                    st.info("No hay embeddings disponibles")

    else:
        st.warning("No se encontraron confesiones con los filtros seleccionados")

    st.markdown("---")
    st.caption("Análisis de Confesiones con NLP | Desarrollado con Streamlit")
