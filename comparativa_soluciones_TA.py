# -*- coding: utf-8 -*-

import plotly
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import codecs
from plotly.subplots import make_subplots
import argparse

def ventas_solucion(sol,df_ventas):
    pesos = [0.5, 0.5]
    ventas_prod = df_ventas['VTAS_PRODUCTO'].tolist()
    ventas_mdo =df_ventas['VTAS_MERCADO'].tolist()
    ventas_prod = [i * pesos[0] for i in ventas_prod]
    ventas_mdo = [i * pesos[1] for i in ventas_mdo]
    ventas = [x + y for x, y in zip(ventas_prod, ventas_mdo)]
    v = []
    for row in sol:
        ventas_delegado = [a * b for a, b in zip(ventas, row)]
        v.append(sum(ventas_delegado))
    return v
 
def get_max_distances(sol, m_dsit):
    distancias=[]
    for i in range(0,44):
        dist_del = 0
        for j in range(0,265):
            if sol[i][j] == 1:
                if m_dsit[i][j] > dist_del:
                    dist_del = m_dsit[i][j]
        distancias.append(dist_del)
    return distancias

if __name__ == '__main__':
    
    #define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('optima_file', 
                        type = str, 
                        help = "file's path with matrix of new solution (.csv)")
    parser.add_argument('optima_delim', 
                        type = str,
                        help = "optima_file (.cvs) delimiter (; or ,)")
    parser.add_argument('optima_decim', 
                        type = str,
                        help = "optima_file (.cvs) decimal (, or .)")   
    parser.add_argument('mundi_file', 
                        type = str, 
                        help = "file's path with matrix of initial solution (.csv)")
    parser.add_argument('mundi_delim', 
                        type = str,
                        help = "mundi_file (.cvs) delimiter (; or ,)")
    parser.add_argument('mundi_decim', 
                        type = str,
                        help = "mundi_file (.cvs) decimal (, or .)") 
    parser.add_argument('ventas_file', 
                        type = str, 
                        help = "file's path of Ventas_Producto_y_Mdo_Hospital_Referencia.csv")
    parser.add_argument('distancias_file', 
                        type = str, 
                        help = "file's path of Matriz_distancias_delegado_centroides_hospitales_referencia.csv'")
    parser.add_argument('comparativa_html', 
                        type = str, 
                        help = "file's name to save html plots")
    args = parser.parse_args()  
    
    ###start program 
    optima = pd.read_csv(args.optima_file,
                         delimiter = args.optima_delim,
                         decimal = args.optima_decim)
    
    inicial = pd.read_csv(args.mundi_file,
                         delimiter = args.mundi_delim,
                         decimal =args.mundi_decim)
    
    df_ventas = pd.read_csv(args.ventas_file,
                          delimiter = ';',
                          decimal = ',')
    
    matrix_dist = pd.read_csv(args.distancias_file,
                          delimiter= ';',
                          decimal = ',')
    
    delegados = optima['DELEGADO'].values
    matrix_dist.drop(['DELEGADO'],axis=1,inplace=True)
    m_dist= matrix_dist.values

    dels = []
    for d in delegados:
        dels.append(d[:4])
        
    optima.drop(['DELEGADO'],axis=1,inplace=True)
    inicial.drop(['DELEGADO'],axis=1,inplace=True)
    
    hospi = optima.columns.values
    hospitales = []
    for h in hospi:
        hospitales.append(h[:30])
        
    sol_opt = optima.values
    sol_ini = inicial.values

    v_ini = ventas_solucion(sol_ini,df_ventas)
    v_opt = ventas_solucion(sol_opt,df_ventas)


    # Define color sets of pie charts
    ventas_colors = ['rgb(100,145,255)', 'rgb(193,211,255)']
    hospitales_colors = ['rgb(238,86,86)', 'rgb(248,187,187)']
    vendedores_colors = ['rgb(248,196,103)', 'rgb(252,237,209)']
    
    #calculate variables of initial solution
    n_ventas_ini=sum(v_ini)
    n_hosp_ini = np.array(sol_ini).sum()
    n_vend_ini=len((np.array(sol_ini).sum(axis=1)))-(np.array(sol_ini).sum(axis=1) ==0).sum()
    distancias_ini = np.array(get_max_distances(sol_ini, m_dist))
    h_ini = np.sum(sol_ini, axis=1)
    
    #calculate variables of new solution
    n_ventas_opt=sum(v_opt)
    n_hosp_opt = np.array(sol_opt).sum()
    n_vend_opt=len((np.array(sol_opt).sum(axis=1)))-(np.array(sol_opt).sum(axis=1) ==0).sum()
    h_opt = np.sum(sol_opt, axis=1)
    distancias_opt =np.array(get_max_distances(sol_opt, m_dist))

    ###generate traces of the plot
    ##pies
    trace1 = go.Pie(labels=['% ventas alcanzadas','% ventas perdido'], 
                    values=[n_ventas_ini, 44-n_ventas_ini], 
                    marker_colors=ventas_colors)
    trace2 = go.Pie(labels=['Nº hospitales visitados','Nº hospitales no visitados'],
                    values=[n_hosp_ini, 265-n_hosp_ini], 
                    marker_colors=hospitales_colors)
    trace3 = go.Pie(labels=['Nº vendedores utilizados','Nº vendedores no utilizados'], 
                    values=[n_vend_ini,44-n_vend_ini],
                    marker_colors=vendedores_colors)
    trace4 = go.Pie(labels=['% ventas alcanzadas','% ventas perdido'], 
                    values=[n_ventas_opt, 44-n_ventas_opt], 
                    marker_colors=ventas_colors)
    trace5 = go.Pie(labels=['Nº hospitales visitados','Nº hospitales no visitados'], 
                    values=[n_hosp_opt, 265-n_hosp_opt], 
                    marker_colors=hospitales_colors)
    ##ventas
    trace6 = go.Pie(labels=['Nº vendedores utilizados','Nº vendedores no utilizados'], 
                    values=[n_vend_opt,44-n_vend_opt],
                    marker_colors=vendedores_colors)
    trace7 = go.Bar(x=dels,
                    y=v_ini,
                    name='Solucion MundiPharma',
                    marker_color='#6491FF')
    trace8 = go.Bar(x=dels,
                    y=v_opt,
                    name='Solucion optimizada',
                    marker_color='#FFC740')
    trace9 = go.Scatter(x=dels,
                        y=[1]*44,
                        mode='lines',
                        line_color='#EE5656',
                        name='Objetivo de ventas')
    ##hospitales
    trace10 = go.Bar(x=dels,
                    y=h_ini,
                    name=f'Solucion MundiPharma - Media:{ round(h_ini.mean(),3)}',
                    marker_color='#6491FF')
    trace11= go.Bar(x=dels,
                    y=h_opt,
                    name=f'Solucion optimizada - Media:{ round(h_opt.mean(),3)}',
                    marker_color='#FFC740')
    ##distancias
    trace12 = go.Bar(x=dels,
                    y=distancias_ini,
                    name=f'Solucion MundiPharma - Media:{ round(distancias_ini.mean(),3)}',
                    marker_color='#6491FF')
    trace13= go.Bar(x=dels,
                    y=distancias_opt,
                    name=f'Solucion optimizada - Media:{ round(distancias_opt.mean(),3)}',
                    marker_color='#FFC740')
        
    subplot_titles= ['<b>MundiPharma<b> <br> Total ventas<br>', 
                    '<b>MundiPharma<b> <br> Total hospitales<br>', 
                    '<b>MundiPharma<b> <br> Total vendedores<br> ',
                     '<b>Optimizada<b> <br> Total ventas<br>', 
                    '<b>Optimizada<b> <br> Total hospitales<br>', 
                    '<b>Optimizada<b> <br> Total vendedores<br> ',
                     '% Ventas alcanzado por cada delegado<br>',
                     'Nº hospitales asignado a cada delegado<br>',
                     'Distancia máxima asignada a cada delegado<br>']
    
    fig = make_subplots(rows=8, 
                        cols=3,
                        specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}],
                                [{'type':'domain'}, {'type':'domain'}, {'type':'domain'}],
                                [{"rowspan": 2, "colspan": 3}, None,None],
                                [None, None,None],
                                [{"rowspan": 2, "colspan": 3}, None,None],
                                [None, None,None],
                                [{"rowspan": 2, "colspan": 3}, None,None],
                                [None, None,None]],
                        subplot_titles=subplot_titles)
    #pie charts
    fig.append_trace(trace1, 1,1)
    fig.append_trace(trace2, 1,2)
    fig.append_trace(trace3,1,3)
    fig.append_trace(trace4,2,1)
    fig.append_trace(trace5, 2,2)
    fig.append_trace(trace6,2,3)
    #ventas
    fig.append_trace(trace7,3,1)
    fig.append_trace(trace8, 3,1)
    fig.append_trace(trace9,3,1)
    #hospitales
    fig.append_trace(trace10, 5,1)
    fig.append_trace(trace11,5,1)
    #distancias
    fig.append_trace(trace12, 7,1)
    fig.append_trace(trace13,7,1)
    
    fig.update_layout(title_text="<b>Comparativa soluciones: MundiPharma vs. Solución Optimizada<b><br><br>",
                      height=1800,
                      width=1500)
    #save plot
    plotly.offline.plot(fig, filename= args.comparativa_html+'.html')
    f=codecs.open(args.comparativa_html+'.html', 'r')
    print(f'Plots html saved at {args.comparativa_html}.html')
    
    
    ####heatmap
    dif = sol_ini+3*sol_opt
    dif/4
    colorscale = [[0, 'white'],
                  [0.25, 'rgb(100,145,255)'],
                  [0.5, 'white'],
                  [0.75, 'rgb(255,199,64)'],
                  [1, 'rgb(96,204,51)']]
    my_layout = go.Layout(
                  yaxis = dict(
                        title = 'Delegados'
                        ),
                  xaxis = dict(
                      title = 'Hospitales'
                      ),
                      font=dict(size=5))
    fig = go.Figure(data=go.Heatmap(
                            z=dif,
                            x=hospitales,
                            y=dels,
                            hoverinfo='text',
                            colorscale = colorscale),
                    layout = my_layout)
    fig.update_xaxes(side="top",
                     tickangle = -90)
    fig.update_layout(xaxis_nticks=265,
                      yaxis_nticks=44)
    fig.update_traces(showscale=False)
    #save plot
    plotly.offline.plot(fig, filename=args.comparativa_html+'_heatmap.html')
    f=codecs.open(args.comparativa_html+'_heatmap.html', 'r')
    print(f'Heatmap html saved at {args.comparativa_html}_heatmap.html')