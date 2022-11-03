#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas as gpd
import folium
import pandas as pd
import cbsodata
import plotly.express as px
import numpy as np
from statsmodels.formula.api import ols
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)


# # Bruto Binnenlands Product

# In[2]:


bbp_capita = pd.read_csv('bbp per inwoner.csv', decimal=",")
bbp_capita.columns = ['Jaar', 'Bulgarije','Estland','Europese unie','Ierland','Letland', 'Litouwen','Luxemburg','Nederland']


# In[3]:


bbp_capita_long = bbp_capita.melt(id_vars='Jaar', value_name='BBP', var_name='Land')


# In[4]:


fig1 = go.Figure([
    go.Scatter(name='Bulgarije', x=bbp_capita['Jaar'],y= bbp_capita['Bulgarije'], mode = 'lines'
    ),
    go.Scatter(name='Estland', x=bbp_capita['Jaar'],y= bbp_capita['Estland'], mode = 'lines'
    ),
    go.Scatter(name='Nederland', x=bbp_capita['Jaar'],y= bbp_capita['Nederland'], mode = 'lines'
    ),
    go.Scatter(name='Ierland', x=bbp_capita['Jaar'],y= bbp_capita['Ierland'], mode = 'lines'
    ),
    go.Scatter(name='Letland', x=bbp_capita['Jaar'],y= bbp_capita['Letland'], mode = 'lines'
    ),
    go.Scatter(name='Litouwen', x=bbp_capita['Jaar'],y= bbp_capita['Litouwen'], mode = 'lines'
    ),
    go.Scatter(name='Luxemburg', x=bbp_capita['Jaar'],y= bbp_capita['Luxemburg'], mode = 'lines'
    ),
    go.Scatter(name='Europese unie', x=bbp_capita['Jaar'],y= bbp_capita['Europese unie'], mode = 'lines')])


dropdown_buttons = [
{'label': "Alle landen", 'method': "update", 'args': [{"visible": [True,True,True,True,True,True,True,True]}, {'title': "Bruto Binnenlands Product per inwoner per land"}]},
{'label': "Bulgarije", 'method': "update", 'args': [{"visible": [True,False,False,False,False,False,False,False]}, {'title': 'Bruto Binnenlands Product per inwoner Bulgarije'}]},
{'label': "Estland", 'method': "update", 'args': [{"visible": [False,True,False,False,False,False,False,False]}, {'title': "Bruto Binnenlands Product per inwoner Estland"}]},
{'label': "Nederland", 'method': "update", 'args': [{"visible": [False,False,True,False,False,False,False,False]}, {'title': "Bruto Binnenlands Product per inwoner Nederland"}]},
{'label': "Ierland", 'method': "update", 'args': [{"visible": [False,False,False,True,False,False,False,False]}, {'title': 'Bruto Binnenlands Product per inwoner Ierland'}]},
{'label': "Letland", 'method': "update", 'args': [{"visible": [False,False,False,False,True,False,False,False]}, {'title': "Bruto Binnenlands Product per inwoner Letland"}]},
{'label': "Litouwen", 'method': "update", 'args': [{"visible": [False,False,False,False,False,True,False,False]}, {'title': "Bruto Binnenlands Product per inwoner Litouwen"}]},
{'label': "Luxemburg", 'method': "update", 'args': [{"visible": [False,False,False,False,False,False,True,False]}, {'title': 'Bruto Binnenlands Product per inwoner Luxemburg'}]},
{'label': "Europese unie", 'method': "update", 'args': [{"visible": [False,False,False,False,False,False,False,True]}, {'title': "Bruto Binnenlands Product per inwoner Europese Unie"}]},
]

fig1.update_layout({
    'updatemenus':[{
            'type': 'dropdown',
            'x': 1.48, 'y': 1.1,
            'buttons': dropdown_buttons
            }]},
    title="Bruto Binnenlands Product per inwoner per land",
                   xaxis_title='Jaar',
                   yaxis_title='BBP per inwoner (x €1000)')

#fig1.update_layout(height=500, width=1000)


# In[5]:


welvaart = pd.read_csv('Welvaart_personen.csv', sep=";",  decimal=',')
welvaart['Jaar'] = welvaart['Jaar'].astype('int')


# In[6]:


welvaart_long = welvaart.melt(id_vars=['Kenmerken van personen', 'Jaar', 'Totaal personen met inkomen', 'Mannen met inkomen','Vrouwen met inkomen'], var_name='Groep', value_name='Gemiddeld_inkomen')


# In[7]:


fig2 = px.box(welvaart_long, x='Groep', y='Gemiddeld_inkomen', color='Groep', title='Gemiddeld inkomen per geslacht')
fig2.update_yaxes(title_text="Gemiddeld inkomen per inwoner (x €1000)")
fig2.update_layout(legend_title_text='Geslacht')
#fig2.update_layout(height=700, width=1000)


# In[8]:


fig3 = go.Figure([
    go.Histogram(name='Totaal', x=welvaart['Totaal gemiddeld persoonlijk inkomen']
    ),
    go.Histogram(name='Mannen', x=welvaart['Mannen gemiddeld persoonlijk inkomen']
    ),
    go.Histogram(name='vrouwen', x=welvaart['Vrouwen gemiddeld persoonlijk inkomen'])])


dropdown_buttons = [
{'label': "Alle variabelen", 'method': "update", 'args': [{"visible": [True,True,True]}, {'title': 'Totaal inkomen per groep'}]},
{'label': "Totaal", 'method': "update", 'args': [{"visible": [True,False,False]}, {'title': 'Totaal inkomen verdeling'}]},
{'label': "Mannen", 'method': "update", 'args': [{"visible": [False,True,False]}, {'title': "Inkomen verdeling Mannen"}]},
{'label': "Vrouwen", 'method': "update", 'args': [{"visible": [False,False,True]}, {'title': "Inkomen verdeling Vrouwen"}]},
]

fig3.update_layout({
    'updatemenus':[{
            'type': 'dropdown',
            'x': 1.48, 'y': 1.1,
            'buttons': dropdown_buttons
            }]},
    title="Inkomen verdeling op basis van geslacht",
                   xaxis_title='Gemiddelde inkomen (x €1000)',
                   yaxis_title='Aantal')

#fig3.update_layout(height=700, width=1000)
fig3.update_layout(legend_title_text='Groep')


# In[9]:


welvaart_Totaal = welvaart_long[welvaart_long['Kenmerken van personen'] == 'Totaal personen']
welvaart_Totaal = welvaart_Totaal[welvaart_long['Groep'] == 'Totaal gemiddeld persoonlijk inkomen']


# In[10]:


groep = ['Leeftijd: 0 tot 15 jaar', 'Leeftijd: 15 tot 25 jaar', 'Leeftijd: 25 tot 45 jaar', 'Leeftijd: 45 tot 65 jaar', 'Leeftijd: 65 jaar of ouder']

welvaart_leeftijd = welvaart_long[welvaart_long['Kenmerken van personen'].isin(groep)]


# In[11]:


groep = ['Migratieachtergrond: Nederland', 'Migratieachtergrond: westers', 'Migratieachtergrond: niet-westers']

welvaart_migratie = welvaart_long[welvaart_long['Kenmerken van personen'].isin(groep)]


# In[12]:


fig4 = px.box(welvaart_migratie, x='Kenmerken van personen', y='Gemiddeld_inkomen', color='Kenmerken van personen', title='Inkomen op basis van migratieachtegrond')
fig4.update_yaxes(title_text="Gemiddeld inkomen per inwoner (x €1000)")
fig4.update_xaxes(title_text="")
fig4.update_layout(legend_title_text='Migratieachtergrond')
#fig4.update_layout(height=500, width=1000)


# In[13]:


fig5 = px.histogram(welvaart_migratie, x='Gemiddeld_inkomen', color='Kenmerken van personen', title='Inkomen verdeling op basis van migratieachtergrond')
fig5.update_xaxes(title_text="Gemiddeld inkomen per inwoner (x €1000)")
fig5.update_yaxes(title_text="Aantal")
fig5.update_layout(legend_title_text='Migratieachtergrond')
#fig5.update_layout(height=700, width=1000)


# In[14]:


fig6 = px.box(welvaart_leeftijd, x='Kenmerken van personen', y='Gemiddeld_inkomen', color='Kenmerken van personen', title='Gemiddeld inkomen op basis van leeftijd')
fig6.update_yaxes(title_text="Gemiddeld inkomen per inwoner (x €1000)")
fig6.update_xaxes(title_text="")
fig6.update_layout(legend_title_text='Leeftijd')
#fig6.update_layout(height=700, width=1000)


# In[15]:


fig7 = px.histogram(welvaart_leeftijd, x='Gemiddeld_inkomen', color='Kenmerken van personen', title='Inkomen verdeling op basis van leeftijd')
fig7.update_xaxes(title_text="Gemiddeld inkomen per inwoner (x €1000)")
fig7.update_yaxes(title_text="Aantal")
fig7.update_layout(legend_title_text='Leeftijd')
#fig7.update_layout(height=700, width=1000)


# In[16]:


bbp_capita_long_nl = bbp_capita_long.loc[bbp_capita_long.Land == 'Nederland']


# In[17]:


samen = bbp_capita_long_nl.merge(welvaart_Totaal, how='left', on='Jaar')
samen = samen.dropna()


# In[18]:


fig8 = px.scatter(samen, y='BBP', x='Gemiddeld_inkomen', title='Spreiding gemiddelde inkomen tegen het BBP per inwoner')
fig8.update_xaxes(title_text="Gemiddeld inkomen per inwoner (x €1000)")
fig8.update_yaxes(title_text="BBP per inwoner (x €1000)")
#fig8.update_layout(height=700, width=1000)


# In[19]:


rho1 = np.corrcoef(samen['BBP'], samen['Gemiddeld_inkomen'])
print(rho1)


# In[20]:


fig_c, ax = plt.subplots()
fig = sns.heatmap(rho1, ax = ax, annot = True)


# In[21]:


model = ols("BBP ~ Gemiddeld_inkomen", data = samen).fit()
model = model.summary()
print(model)


# # Arbeidsmarkt

# In[22]:


werkloosheid = pd.read_csv('werkloosheid.csv', sep=";",  decimal=',')
werkloosheid['Perioden'] = pd.to_datetime(werkloosheid['Perioden'], format = '%d-%m-%Y')
werkloosheid = werkloosheid.melt(id_vars = ['Perioden'], value_name = 'Werkloosheidspercentage', var_name = 'Leeftijdsgroep')


# In[23]:


participatie = pd.read_csv('Nettoarbeidsparticipatie, seizoengecorrigeerd (% van beroepsbevolking).csv', sep=";",  decimal=',')
participatie['Perioden'] = pd.to_datetime(participatie['Perioden'], format = '%d-%m-%Y')
participatie = participatie.melt(id_vars = ['Perioden'], value_name = 'Arbeidsparticipatie', var_name = 'Leeftijdsgroep')


# In[24]:


df_arbeidsmarkt = pd.merge(werkloosheid, participatie, on=['Perioden','Leeftijdsgroep'], how='outer')


# In[25]:


fig10 = px.line(df_arbeidsmarkt, x='Perioden', y = 'Werkloosheidspercentage', color='Leeftijdsgroep', title='Werkloosheid door de jaren')
fig10.update_xaxes(title_text="Jaar")
fig10.update_yaxes(title_text="Werkloosheidspercentage")
#fig10.update_layout(height=700, width=1000)


# In[26]:


fig11 = px.line(df_arbeidsmarkt, x='Perioden', y = 'Arbeidsparticipatie', color='Leeftijdsgroep', title= 'Arbeidsparticipatie door de jaren')
fig11.update_xaxes(title_text="Jaar")
fig11.update_yaxes(title_text="Arbeidsparticipatie")
#fig11.update_layout(height=700, width=1000)


# In[27]:


provinciegrenzen = gpd.read_file('provinciegrenzen.json')


# In[28]:


df = pd.read_csv('Openstaande vacatures.csv',sep=";",  decimal=',')
df['Kwartaal'] = pd.to_datetime(df['Kwartaal'], format = '%d-%m-%Y')


# In[29]:


df2 = df.dropna()

df2['Jaar'] = df2['Jaar'].astype('int')


# In[30]:


df2 = df2.melt(id_vars = ['Jaar', 'Kwartaal'], value_name = 'Aantal vacatures', var_name = 'Provincie')
df2.replace('Fryslân', 'Friesland', inplace = True)
df3 = df2.loc[df2.Jaar == 2022]
df3 = df2.loc[df2.Kwartaal == '2022-01-01']


# In[31]:


df5 = df3.merge(provinciegrenzen, how = 'left', left_on='Provincie', right_on='PROV_NAAM')

df5 = df5[['Jaar', 'Provincie','Aantal vacatures','geometry']]


# In[32]:


geo_df_crs = {'init': 'epsg:28992'}
geo_df = gpd.GeoDataFrame(df5, crs= geo_df_crs, geometry = df5.geometry)


# In[33]:


m = folium.Map(location= [52.371807, 4.896029], zoom_start = 7)

m.choropleth(
    geo_data=geo_df,
    name="geometry",
    data=geo_df,
    columns= ["Provincie","Aantal vacatures"],
    key_on="feature.properties.Provincie",
    fill_color="RdYlGn_r",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Aantal vacatures x1000")

folium.features.GeoJson(geo_df,  
                        name='Labels',
                        style_function=lambda x: {'color':'transparent','fillColor':'transparent','weight':0},
                        tooltip=folium.features.GeoJsonTooltip(fields=['Provincie',
                                                                       'Aantal vacatures'],
                                                                aliases = ['Provincie: ',
                                                                           'Aantal vacatures (x 1000): '],
                                                                labels=True,
                                                                sticky=False
                                                                            )
                       ).add_to(m)




# In[34]:


fig12 = px.bar(df2, x='Provincie', y= 'Aantal vacatures', color='Provincie', animation_frame="Jaar",  animation_group="Provincie", title='Aantal vacatures per jaar')
#fig12.update_layout(height=700, width=1000)


# In[35]:


fig13 = px.line(df2, x='Kwartaal', y= 'Aantal vacatures', color='Provincie', title = 'Aantal vacatures per provincie')
fig13.update_xaxes(title_text="Jaar")
fig13.update_yaxes(title_text="Aantal vacatures")
#fig13.update_layout(height=700, width=1000)


# # Inflatie

# In[36]:


df_inflatie2 = pd.read_csv('Inflatie_2.csv',sep=";",  decimal=',')
df_inflatie2['Perioden'] = pd.to_datetime(df_inflatie2['Perioden'], format = '%d-%m-%Y')


# In[37]:


fig14 = go.Figure()

fig14.add_trace(
    go.Scatter(x=df_inflatie2['Perioden'], y=df_inflatie2['CPI']))

# Set title
fig14.update_layout(
    title_text="Inflatie in de afgelopen 7 jaar"
)

# Add range slider
fig14.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

fig14.add_annotation(showarrow=True,
                   arrowhead=1,
                   align = 'right',
                   x='2020-04-01',
                   y=1.4,
                   text="Corona en lockdown",
                   opacity=0.7)

fig14.add_annotation(showarrow=True,
                   arrowhead=1,
                   align = 'right',
                   x='2021-09-01',
                   y=2.7,
                   text="Olieprijzen",
                   opacity=0.7)
fig14.add_annotation(showarrow=True,
                   arrowhead=1,
                   align = 'right',
                   x='2022-02-01',
                   y=6.2,
                   text="Oekraine",
                   opacity=0.7
                   )
fig14.add_annotation(showarrow=True,
                   arrowhead=1,
                   align = 'right',
                   x='2022-06-01',
                   y=8.6,
                   text="Gas en stroom",
                   opacity=0.7
                   )

fig14.update_xaxes(title_text="Jaar")
fig14.update_yaxes(title_text="Inflatie in %")
#fig14.update_layout( height=700, width=1000)



# In[38]:


inflatie_jaren = pd.read_csv('inflatie_jaren.csv', sep=";",  decimal=',')
inflatie_jaren['Jaar'] = inflatie_jaren['Jaar'].astype('str')
inflatie_jaren['inflatie'] = inflatie_jaren['inflatie'].astype('float')


# # Bedrijven

# In[39]:


#bedrijven_bedrijfstak_dirty = pd.DataFrame(cbsodata.get_data('81589NED'))
#bedrijven_bedrijfstak_dirty = bedrijven_bedrijfstak_dirty[bedrijven_bedrijfstak_dirty['BedrijfstakkenBranchesSBI2008'] == 'A-U Alle economische activiteiten']
#bedrijven_bedrijfstak_dirty = bedrijven_bedrijfstak_dirty.drop(['ID','BedrijfstakkenBranchesSBI2008','k_0Tot50WerkzamePersonen_10','k_0Tot250WerkzamePersonen_11'], axis=1)
#bedrijven_bedrijfstak_dirty.columns = ['Perioden','Totaal aantal bedrijven','1 werkzaam persoon','2 werkzame personen','3 tot 5 werkzame personen','5 tot 10 werkzame personen','10 tot 20 werkzame personen','20 tot 50 werkzame personen','50 tot 100 werkzame personen','100 of meer werkzame personen','Natuurlijke personen','Rechtspersonen']

#bedrijven_bedrijfstak = bedrijven_bedrijfstak_dirty
#bedrijven_bedrijfstak.to_csv('bedrijven_bedrijfstak.csv')


# Voor de snelheid in streamlit gebruiken we een csv
bedrijven_bedrijfstak = pd.read_csv('bedrijven_bedrijfstak.csv')


# In[59]:


fig19 = go.Figure([
    go.Scatter(name = 'Totaal aantal bedrijven',       x = bedrijven_bedrijfstak['Perioden'],y = bedrijven_bedrijfstak['Totaal aantal bedrijven']      , mode = 'lines'),
    go.Scatter(name = '1 werkzaam persoon'           , x = bedrijven_bedrijfstak['Perioden'],y = bedrijven_bedrijfstak['1 werkzaam persoon']           , mode = 'lines'),
    go.Scatter(name = '3 tot 5 werkzame personen'    , x = bedrijven_bedrijfstak['Perioden'],y = bedrijven_bedrijfstak['3 tot 5 werkzame personen']    , mode = 'lines'),
    go.Scatter(name = '5 tot 10 werkzame personen'   , x = bedrijven_bedrijfstak['Perioden'],y = bedrijven_bedrijfstak['5 tot 10 werkzame personen']   , mode = 'lines'),
    go.Scatter(name = '10 tot 20 werkzame personen'  , x = bedrijven_bedrijfstak['Perioden'],y = bedrijven_bedrijfstak['10 tot 20 werkzame personen']  , mode = 'lines'),
    go.Scatter(name = '20 tot 50 werkzame personen'  , x = bedrijven_bedrijfstak['Perioden'],y = bedrijven_bedrijfstak['20 tot 50 werkzame personen']  , mode = 'lines'),
    go.Scatter(name = '50 tot 100 werkzame personen' , x = bedrijven_bedrijfstak['Perioden'],y = bedrijven_bedrijfstak['50 tot 100 werkzame personen'] , mode = 'lines'),
    go.Scatter(name = '100 of meer werkzame personen', x = bedrijven_bedrijfstak['Perioden'],y = bedrijven_bedrijfstak['100 of meer werkzame personen'], mode = 'lines'),
    go.Scatter(name = 'Natuurlijke personen'         , x = bedrijven_bedrijfstak['Perioden'],y = bedrijven_bedrijfstak['Natuurlijke personen']         , mode = 'lines'),
    go.Scatter(name = 'Rechtspersonen'               , x = bedrijven_bedrijfstak['Perioden'],y = bedrijven_bedrijfstak['Rechtspersonen']               , mode = 'lines')])

dropdown_buttons = [
{'label': "All"                          , 'method': "update", 'args': [{"visible": [True,True,True,True,True,True,True,True,True,True]}         , {'title': "Totaal aantal bedrijven"}]},
{'label': "Totaal aantal bedrijven"      , 'method': "update", 'args': [{"visible": [True,False,False,False,False,False,False,False,False,False]}, {'title': 'Totaal aantal bedrijven'}]},
{'label': "1 werkzaam persoon"           , 'method': "update", 'args': [{"visible": [False,True,False,False,False,False,False,False,False,False]}, {'title': " Totaal aantal bedrijven met 1 werkzaam persoon"}]},
{'label': "3 tot 5 werkzame personen"    , 'method': "update", 'args': [{"visible": [False,False,True,False,False,False,False,False,False,False]}, {'title': "Totaal aantal bedrijven met 3 tot 5 werkzame personen"}]},
{'label': "5 tot 10 werkzame personen"   , 'method': "update", 'args': [{"visible": [False,False,False,True,False,False,False,False,False,False]}, {'title': 'Totaal aantal bedrijven met 5 tot 10 werkzame personen'}]},
{'label': "10 tot 20 werkzame personen"  , 'method': "update", 'args': [{"visible": [False,False,False,False,True,False,False,False,False,False]}, {'title': "Totaal aantal bedrijven met 10 tot 20 werkzame personen"}]},
{'label': "20 tot 50 werkzame personen"  , 'method': "update", 'args': [{"visible": [False,False,False,False,False,True,False,False,False,False]}, {'title': "Totaal aantal bedrijven met 20 tot 50 werkzame personen"}]},
{'label': "50 tot 100 werkzame personen" , 'method': "update", 'args': [{"visible": [False,False,False,False,False,False,True,False,False,False]}, {'title': 'Totaal aantal bedrijven met 50 tot 100 werkzame personen'}]},
{'label': "100 of meer werkzame personen", 'method': "update", 'args': [{"visible": [False,False,False,False,False,False,False,True,False,False]}, {'title': "Totaal aantal bedrijven met 100 of meer werkzame personen"}]},
{'label': "Natuurlijke personen"         , 'method': "update", 'args': [{"visible": [False,False,False,False,False,False,False,False,True,False]}, {'title': "Natuurlijke personen"}]},
{'label': "Rechtspersonen"               , 'method': "update", 'args': [{"visible": [False,False,False,False,False,False,False,False,False,True]}, {'title': "Rechtspersonen"}]},]

fig19.update_layout({
    'updatemenus':[{
            'type': 'dropdown',
            'x': 1.568,
            'y': 1.115,
            'buttons': dropdown_buttons}]},
    
    title       ='Totaal aantal bedrijven',
    xaxis_title ='Periode',
    yaxis_title ='Aantal bedrijven')

#fig19.update_layout(height = 600, width = 1000)


# In[41]:


#failliet_bedrijven_regio = pd.DataFrame(cbsodata.get_data('82522NED'))
#provinciegrenzen = gpd.read_file('provinciegrenzen.json')

#provincies = ['Drenthe', 'Friesland', 'Flevoland', 'Gelderland', 'Groningen',
 #             'Limburg', 'Noord-Brabant', 'Noord-Holland', 'Overijssel',
  #            'Utrecht', 'Zeeland', 'Zuid-Holland']

#failliet_bedrijven_regio['RegioS'] = failliet_bedrijven_regio['RegioS'].replace(['Drenthe (PV)',
#                                                                                 'Fryslân (PV)',
#                                                                                 'Flevoland (PV)',
#                                                                                 'Gelderland (PV)',
#                                                                                 'Groningen (PV)',
#                                                                                 'Limburg (PV)',
#                                                                                 'Noord-Brabant (PV)',
#                                                                                 'Noord-Holland (PV)',
#                                                                                 'Overijssel (PV)'
#                                                                                 'Utrecht (PV)',
#                                                                                 'Zeeland (PV)',
#                                                                                 'Zuid-Holland (PV)'],
#                                                                                ['Drenthe',
#                                                                                 'Friesland',
#                                                                                 'Flevoland',
#                                                                                 'Gelderland',
#                                                                                 'Groningen',
#                                                                                 'Limburg',
#                                                                                 'Noord-Brabant',
#                                                                                 'Noord-Holland',
#                                                                                 'Overijssel'
#                                                                                 'Utrecht',
#                                                                                 'Zeeland',
#                                                                                 'Zuid-Holland'])

#failliet_bedrijven_regio1 = failliet_bedrijven_regio[failliet_bedrijven_regio['RegioS'].isin(provincies)]
#failliet_bedrijven_regio1.to_csv('failliet_bedrijven_regio1.csv')


# In[42]:


failliet_bedrijven_regio1 = pd.read_csv('failliet_bedrijven_regio1.csv')

failliet_bedrijven_regio1 = failliet_bedrijven_regio1.loc[~failliet_bedrijven_regio1['Perioden'].str.contains("kwartaal", case = False)]
failliet_bedrijven_regio1['Jaar'] = failliet_bedrijven_regio1['Perioden'].str[:4]
failliet_bedrijven_regio1['Maand'] = failliet_bedrijven_regio1['Perioden'].str[5:]

failliet_bedrijven_regio1['Maand'] = failliet_bedrijven_regio1['Maand'].replace(['januari',
                                                                                'februari',
                                                                                'maart',
                                                                                'april',
                                                                                'mei',
                                                                                'juni',
                                                                                'juli',
                                                                                'augustus',
                                                                                'september', 
                                                                                'oktober',
                                                                                'november',
                                                                                'december'],
                                                                                [1,
                                                                                2,
                                                                                3,
                                                                                4,
                                                                                5,
                                                                                6,
                                                                                7,
                                                                                8,
                                                                                9,
                                                                                10,
                                                                                11,
                                                                                12])

failliet_bedrijven_regio1['Jaar_Maand'] = failliet_bedrijven_regio1['Jaar'].astype('str') + '-' + failliet_bedrijven_regio1['Maand'].astype('str')
pd.to_datetime(failliet_bedrijven_regio1['Jaar_Maand'])


failliet_bedrijven_regio1 = failliet_bedrijven_regio1.merge(provinciegrenzen, how = 'left', left_on='RegioS', right_on='PROV_NAAM')
failliet_bedrijven_regio1 = failliet_bedrijven_regio1[['Jaar_Maand', 'Jaar', 'RegioS', 'TypeGefailleerde', 'UitgesprokenFaillissementen_1', 'geometry']]



# In[43]:


gefilterd = failliet_bedrijven_regio1[failliet_bedrijven_regio1['Jaar_Maand'] == '2020-']
gefilterd = gefilterd[gefilterd['TypeGefailleerde'] == 'Bedrijven, instellingen en eenmanszaken']

geo_df_crs = {'init': 'epsg:28992'}
gefilterd = gpd.GeoDataFrame(gefilterd, crs= geo_df_crs, geometry = gefilterd.geometry)


# In[44]:


m_y = folium.Map(location= [52.371807, 4.896029], zoom_start = 7)

m_y.choropleth(
    geo_data=gefilterd,
    name="geometry",
    data=gefilterd,
    columns= ["RegioS","UitgesprokenFaillissementen_1"],
    key_on="feature.properties.RegioS",
    fill_color="RdYlGn_r",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Aantal faillissementen 2020")

folium.features.GeoJson(gefilterd,  
                        name='Labels',
                        style_function=lambda x: {'color':'transparent','fillColor':'transparent','weight':0},
                        tooltip=folium.features.GeoJsonTooltip(fields=['UitgesprokenFaillissementen_1'],
                                                                aliases = ['Aantal faillissementen 2020'],
                                                                labels=True,
                                                                sticky=False
                                                                            )
                       ).add_to(m_y)



# # Woningmarkt

# In[45]:


woning = pd.read_csv('woningen.csv', sep=";",  decimal=',')
woning = woning.melt(id_vars = ["Regio's", 'Onderwerp'], value_name = 'Prijs', var_name = 'Jaar')
woning['Jaar'] = woning['Jaar'].astype('str')


# In[46]:


fig15 = px.line(woning, x='Jaar', y='Prijs', animation_frame="Regio's",  animation_group="Jaar", range_y=[0,600000], title='Huizenprijzen per provincie')
fig15.update_xaxes(title_text="Jaar")
fig15.update_yaxes(title_text="Huizenprijzen in €")
#fig15.update_layout(height=700, width=1000)


# In[47]:


woning_samen = woning.merge(inflatie_jaren, on='Jaar', how='left')


# In[48]:


rho2 = np.corrcoef(woning_samen.Prijs, woning_samen.inflatie)
#print(rho)

#correlatie van niks dus hier maar geen regressie mee maken


# In[49]:


fig16 = make_subplots(rows=4, cols=1)

fig16.append_trace(go.Bar(
    x=woning_samen['Jaar'],
    y=woning_samen['Prijs'],
    name='Huizenprijzen'
), row=1, col=1)

fig16.append_trace(go.Scatter(
    x=woning_samen['Jaar'],
    y=woning_samen['inflatie'],
    mode='lines',
    name='Inflatie'
), row=2, col=1)

fig16.append_trace(go.Scatter(
    x=bbp_capita_long_nl['Jaar'],
    y=bbp_capita_long_nl['BBP'],
    mode='lines',
    name='BBP'
), row=3, col=1)

fig16.append_trace(go.Scatter(
    x=welvaart_Totaal['Jaar'],
    y=welvaart_Totaal['Gemiddeld_inkomen'],
    mode='lines',
    name='Inkomen'
), row=4, col=1)


fig16.update_xaxes(title_text="Jaar", row=1, col=1)
fig16.update_xaxes(title_text="Jaar", row=2, col=1)
fig16.update_xaxes(title_text="Jaar", row=3, col=1)
fig16.update_xaxes(title_text="Jaar", row=4, col=1)

fig16.update_yaxes(title_text="Woningprijs", row=1, col=1)
fig16.update_yaxes(title_text="Inflatie", row=2, col=1)
fig16.update_yaxes(title_text="BBP per inwoner", row=3, col=1)
fig16.update_yaxes(title_text="Gemiddeld inkomen", row=4, col=1)

fig16.update_layout(height=1000, width=1000, title_text="Vergelijking tussen aspecten")


# In[50]:


fig17 = px.histogram(woning, x='Prijs', nbins = 10, title='Verdeling huizenprijzen')
fig17.update_yaxes(title_text="Aantal")
fig17.update_xaxes(title_text="Huizenprijzen in €")
#fig17.update_layout(height=700, width=1000)


# In[51]:


fig18 = px.box(woning, x="Regio's", y = 'Prijs', color = "Regio's", title='Huizenprijzen per provincie')
fig18.update_xaxes(title_text="Provincie")
fig18.update_yaxes(title_text="Huizenprijzen in €")
#fig18.update_layout(height=700, width=1000)


# In[52]:


geodata_url = 'https://geodata.nationaalgeoregister.nl/cbsgebiedsindelingen/wfs?request=GetFeature&service=WFS&version=2.0.0&typeName=cbs_gemeente_2017_gegeneraliseerd&outputFormat=json'
gemeentegrenzen = gpd.read_file(geodata_url)


# In[53]:


woning2 = pd.read_csv('Woningprijzen.csv', sep=",",  decimal=',')
woning2.columns = ['statnaam','prijs2021', 'prijs2020']


# In[54]:


woning2.replace('Laren (NH.)', 'Laren', inplace = True)
woning2.replace('Utrecht (gemeente)', 'Utrecht', inplace = True)
woning2.replace('s-Gravenhage (gemeente)', 's-Gravenhage', inplace = True)
woning2.replace('Rijswijk (ZH.)', 'Rijswijk', inplace = True)
woning2.replace('Groningen (gemeente)', 'Groningen', inplace = True)
woning2.replace('Middelburg (Z.)', 'Middelburg', inplace = True)
woning2.replace('Beek (L.)', 'Beek', inplace = True)
woning2.replace('Hengelo (O.)', 'Hengelo', inplace = True)
woning2.replace('Stein (L.)', 'Stein', inplace = True)
woning2 = woning2.merge(gemeentegrenzen, how='left', on='statnaam')
woning2 = woning2[['statnaam','prijs2021', 'prijs2020','geometry' ]]

woning2 = woning2.dropna()


# In[55]:


geo_df_crs = {'init': 'epsg:4326'}
geo_df2 = gpd.GeoDataFrame(woning2, crs= geo_df_crs, geometry = woning2.geometry)
geo_df.dropna()


# In[56]:


m1 = folium.Map(location= [52.371807, 4.896029], zoom_start = 7, tiles="cartodbpositron")



m1.choropleth(
    geo_data=geo_df2,
    name="2021",
    data=geo_df2,
    columns= ["statnaam","prijs2021"],
    key_on="feature.properties.statnaam",
    fill_color="YlGnBu",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="2021 Huizenprijzen (x 1000)")

folium.features.GeoJson(geo_df2,  
                        name='Huizenprijzen per gemeente',
                        style_function=lambda x: {'color':'transparent','fillColor':'transparent','weight':0},
                        tooltip=folium.features.GeoJsonTooltip(fields=['statnaam',
                                                                      'prijs2021'],
                                                                aliases = ['Gemeente: ',
                                                                           'Gemiddelde woningprijs: '],
                                                                labels=True,
                                                                sticky=False
                                                                            )
                       ).add_to(m1)



# In[57]:


m2 = folium.Map(location= [52.371807, 4.896029], zoom_start = 7, tiles="cartodbpositron")

m2.choropleth(
    geo_data=geo_df2,
    name="2020",
    data=geo_df2,
    columns= ["statnaam","prijs2020"],
    key_on="feature.properties.statnaam",
    fill_color="YlGnBu",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="2020 Huizenprijzen (x 1000)")

folium.features.GeoJson(geo_df2,  
                        name='Huizenprijzen per gemeente',
                        style_function=lambda x: {'color':'transparent','fillColor':'transparent','weight':0},
                        tooltip=folium.features.GeoJsonTooltip(fields=['statnaam',
                                                                      'prijs2020'],
                                                                aliases = ['Gemeente: ',
                                                                           'Gemiddelde woningprijs: '],
                                                                labels=True,
                                                                sticky=False
                                                                            )
                       ).add_to(m2)




# # Streamlit opbouw

# In[58]:


st.title('Dashboard Nederlandse Economie')
st.markdown('In dit dashboard hebben we verschillende aspecten van de Nederlandse Economie geanalyseerd en in kaart gebracht. De inhoud bestaat uit:')
st.markdown('1. Algemene indicatoren van economische groei')
st.markdown('2. Arbeidsmarkt')
st.markdown('3. Bedrijven')
st.markdown('4. Woningmarkt')
st.markdown('5. Conclusie')
st.markdown(' ')


st.header("1. Algemene indicatoren van economische groei")
st.markdown('Binnen dit onderwerp hebben we gekeken naar het Bruto Binnenlands Product (BBP) per inwoner, de inflatie door de jaren heen en het gemiddelde inkomen per inwoner. **Bron: CBS**')
st.subheader('Bruto Binnenlands Product per inwoner')
st.plotly_chart(fig1)
st.markdown("Door de coronacrisis kromp de Nederlandse economie in volume met 3,8 procent ten opzichte van een jaar eerder.") 
st.subheader('Inflatie door de jaren heen')
st.plotly_chart(fig14)
st.markdown("De inflatie is in september gestegen naar een recordhoogte van 14,5%. Binnen deze grafiek speelde de coronacrisis, energieprijzen en de oorlog in Oekraine een belangrijke rol.")
st.subheader('Gemiddelde inkomen per inwoner')
st.plotly_chart(fig2)
st.plotly_chart(fig3)
st.plotly_chart(fig4)
st.plotly_chart(fig5)
st.plotly_chart(fig6)
st.plotly_chart(fig7)

st.subheader('Lineaire regressie')
show_trendline = st.checkbox('Show trendline')
if show_trendline:
    fig8 = px.scatter(samen, y='BBP', x='Gemiddeld_inkomen', title='Spreiding gemiddelde inkomen tegen het BBP per inwoner', trendline='ols')
    fig8.update_xaxes(title_text="Gemiddeld inkomen per inwoner (x €1000)")
    fig8.update_yaxes(title_text="BBP per inwoner (x €1000)")
st.plotly_chart(fig8)
st.markdown('Zoals in de scatter plot te zien is, is er een sterke positieve correlatie tussen het BBP en Inkomen per inwoner. Middels een Correlatie matrix bekijken we de correlatie')
st.write(fig_c)
st.markdown('Om dit verband verder aan te tonen voeren we een linaire regressie uit met als x (onafhankelijke) variabele het Gemiddelde inkomen per inwoner per jaar en als y (afhankelijke) variabele het BBP per inwoner. ')
st.subheader('Resultaten Lineaire Regressie')
st.write(model)
st.markdown('')
st.markdown('**Uit het model vallen de volgende punten op:**')
st.markdown("**R-squared**: 0,948. In dit model verklaard 94,8% van de variabele Gemiddeld inkomen het BBP per inwoner. ")
st.markdown("**Durban-Watson**: 1,772. De uitkomst is in de buurt van 2, dus de residuals zijn niet aan elkaar gecorreleerd.")
st.markdown("**p>|t|**: p-value < 0,05. Het model is statistisch significant. De variabale heeft een significant invloed op de uitkomst van het BBP per inwoner.")
st.markdown("**Intercept**: -6,5947")
st.markdown("**Coefficient**: 1,5657 ")
st.markdown("We kunnen hier dus de volgende formule mee maken: **BBP per inwoner = 1,5657*Gemiddeld inkomen - 6,5947**")


st.header("2. Arbeidsmarkt")
st.markdown('Binnen dit onderwerp hebben we gekeken naar de werkloosheid en arbeidsparticipatie en het aantal vacatures in Nederland. **Bron: CBS** en **Geoportaal Overijssel**')

st.plotly_chart(fig10)
st.plotly_chart(fig11)
st.markdown('Opvallend is dat het werkloosheidpercentage toenam, maar de arbeidsparticipatie redelijk stabiel bleef. Een verklaring hiervoor kan zijn dat er nieuwe mensen zijn toegetreden tot de arbeidsmarkt die eerder niet tot de beroepsbevolking behoorden.')
st.plotly_chart(fig12)
st.subheader('Aantal openstaande vacatures per provincie')
folium_static(m)
st.markdown("Opvallend uit de plots is dat het aantal vacatures vanaf 2021 weer hard steeg (met name in de randstad), maar het werkloosheidspercentage sinds 2022 weer aan het toenemen is. ")

st.header("3. Bedrijven")
st.markdown('Binnen dit aspect hebben we gekeken naar de karakteristieken van bedrijven. **Bron: CBS** ')
st.plotly_chart(fig19)
st.markdown("In deze grafiek staat hoeveel bedrijven er zijn door de loop van de jaren zijn. Duidelijk wordt dat het aantal bedrijven in de afgelopen 15 jaar verdubbeld is. Wanneer beter naar de dataset gekeken wordt, wordt er duidelijk dat dat vooral komt door het toenemende aantal zzp'ers. Bij bedrijven met meer dan 1 werkzaam persoon is de vele malen minder of is het aantal zelfs gelijk gebleven.")
st.subheader('Aantal faillisementen per provincie')
folium_static(m_y)


st.header("4. Woningmarkt")
st.markdown('Binnen dit aspect hebben we gekeken hoe de huizenprijzen per provincie en gemeente zijn veranderd door de jaren heen. **Bron: CBS** en **Nationaal Georegister**.') 
st.markdown('De geodata wordt via de API van het Nationaal Georegister van PDOK gedownload en vervolgens ingelezen met read_file uit geopandas.')
st.sidebar.title("Dashboard Nederlandse Economie")
st.sidebar.subheader("Woningmarkt kaart")
location_select = st.sidebar.selectbox('Welk jaar wil je zien?',("2020", "2021"))

if location_select == '2020': 
    m_s = m2
elif location_select =='2021':
    m_s = m1
st.subheader('Gemiddelde woningprijzen per gemeente')    
folium_static(m_s)
st.plotly_chart(fig15)
st.plotly_chart(fig17)
show_datapoints = st.checkbox('Show data points')
if show_datapoints:
    fig18 = px.box(woning, x="Regio's", y = 'Prijs', color = "Regio's", points='all', title='Huizenprijzen per provincie')
    fig18.update_xaxes(title_text="Provincie")
    fig18.update_yaxes(title_text="Huizenprijzen in €")
st.plotly_chart(fig18)
st.markdown("Uit de plots is op te maken dat in elke provincie de gemiddelde huizenprijzen duurder zijn geworden. De stijging ten op zichte van 2020 is vooral groot in de randstad.")

st.header("5. Conclusie")
st.plotly_chart(fig16)


# In[ ]:




