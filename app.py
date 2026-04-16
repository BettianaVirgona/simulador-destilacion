import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

# --- Configuración de la página ---
st.set_page_config(page_title="Simulador McCabe-Thiele", page_icon="⚗️", layout="wide")

st.title("⚗️ Simulador de Destilación - Método McCabe-Thiele")
st.markdown("Bienvenido al simulador. Ingresá tus datos de equilibrio en el menú lateral y ajustá las variables operativas para generar el gráfico.")

# --- Panel Lateral (Sidebar) para los controles ---
st.sidebar.header("1. Datos de Equilibrio")
st.sidebar.markdown("Copiá y pegá desde Excel (columnas $x$ e $y$):")

datos_ejemplo = """0.00    0.00
0.10    0.25
0.20    0.45
0.30    0.60
0.40    0.70
0.50    0.78
0.60    0.85
0.70    0.90
0.80    0.94
0.90    0.98
1.00    1.00"""

text_datos = st.sidebar.text_area("", value=datos_ejemplo, height=200)

st.sidebar.header("2. Parámetros Operativos")
xd = st.sidebar.slider("Fracción de Destilado (xD)", 0.50, 0.99, 0.95, 0.01)
xf = st.sidebar.slider("Fracción de Alimentación (xF)", 0.10, 0.80, 0.50, 0.01)
xb = st.sidebar.slider("Fracción de Fondos (xB)", 0.01, 0.50, 0.05, 0.01)
R = st.sidebar.slider("Relación de Reflujo (R)", 0.5, 10.0, 2.0, 0.1)
q = st.sidebar.slider("Condición Térmica (q)", 0.0, 1.5, 1.0, 0.1)

boton_simular = st.sidebar.button("🚀 Graficar Simulación", type="primary", use_container_width=True)

# --- Lógica de Simulación (Se ejecuta al apretar el botón) ---
if boton_simular:
    if xd <= xf or xf <= xb:
        st.error("Error lógico: Verificá que xD > xF > xB para que la columna funcione.")
    else:
        # Procesar los datos ingresados
        x_lista, y_lista = [], []
        lineas = text_datos.strip().split('\n')
        try:
            for linea in lineas:
                linea_limpia = linea.replace(',', '.') 
                partes = linea_limpia.split()
                if len(partes) >= 2:
                    x_lista.append(float(partes[0]))
                    y_lista.append(float(partes[1]))
            
            x_tabla = np.array(x_lista)
            y_tabla = np.array(y_lista)
            equilibrio = interp1d(x_tabla, y_tabla, kind='cubic', bounds_error=False, fill_value="extrapolate")
            
            # Funciones
            def lrs(x): return (R / (R + 1)) * x + (xd / (R + 1))
            def recta_q(x): return None if q == 1 else (q / (q - 1)) * x - (xf / (q - 1))

            # Intersección
            if q == 1:
                xi, yi = xf, lrs(xf)
            else:
                func = lambda x : lrs(x) - recta_q(x)
                xi = fsolve(func, xf)[0]
                yi = lrs(xi)

            # Cálculo de etapas
            steps_x, steps_y = [xd], [xd]
            curr_x, n_etapas = xd, 0
            max_etapas = 50 

            while curr_x > xb and n_etapas < max_etapas:
                curr_y = steps_y[-1]
                func_eq = lambda x : equilibrio(x) - curr_y
                next_x = fsolve(func_eq, curr_x)[0]
                steps_x.extend([next_x, next_x])
                
                if next_x > xi:
                    next_y = lrs(next_x)
                else:
                    if xi == xb: next_y = xb
                    else:
                        m_lra = (yi - xb) / (xi - xb)
                        next_y = m_lra * (next_x - xb) + xb
                    
                steps_y.extend([curr_y, next_y])
                curr_x = next_x
                n_etapas += 1

            # --- Graficación ---
            fig, ax = plt.subplots(figsize=(8, 8))
            x_vals = np.linspace(0, 1, 100)
            ax.plot(x_vals, equilibrio(x_vals), label='Curva de Equilibrio', color='blue', linewidth=2)
            ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)

            ax.plot([xi, xd], [yi, xd], 'g-', label='LRS (Rectificación)', linewidth=2)
            ax.plot([xb, xi], [xb, yi], 'm-', label='LRA (Agotamiento)', linewidth=2)

            if q == 1: ax.axvline(xf, ymin=xf, ymax=yi, color='orange', label='Línea q', linewidth=2)
            else: ax.plot([xf, xi], [xf, yi], '-', color='orange', label='Línea q', linewidth=2)

            ax.plot(steps_x, steps_y, 'r', label=f'Etapas: {n_etapas}', linewidth=1.5)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f'Simulación Completada - {n_etapas} Etapas Teóricas', fontsize=16)
            ax.set_xlabel('Fracción molar en líquido (x)', fontsize=12)
            ax.set_ylabel('Fracción molar en vapor (y)', fontsize=12)
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.7)

            # Mostrar el gráfico y el resultado en Streamlit
            st.success(f"Cálculo exitoso: Se requieren **{n_etapas} etapas teóricas**.")
            st.pyplot(fig)

        except Exception as e:
            st.error("❌ Ocurrió un error con los datos. Asegurate de pegar dos columnas de números válidos.")
