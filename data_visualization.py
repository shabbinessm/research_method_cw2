import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('C:/Users/Mike/Desktop/Results_21Mar2022.csv')

# define variable grouping
ghg_related = ['mean_ghgs', 'mean_ghgs_ch4', 'mean_ghgs_n2o']
land_water = ['mean_land', 'mean_watuse']
eco_harm = ['mean_bio', 'mean_acid', 'mean_eut', 'mean_watscar']
ehi_columns = ghg_related + land_water + eco_harm

# standardize the data
scaler = MinMaxScaler()
normalized = pd.DataFrame(scaler.fit_transform(df[ehi_columns]), columns=ehi_columns)

# weights for EHI calculations  
weights = {
    **{col: 0.2 / len(ghg_related) for col in ghg_related},
    **{col: 0.2 / len(land_water) for col in land_water},
    **{col: 0.6 / len(eco_harm) for col in eco_harm},
}

weights_2 = {    
    **{col: 0.6 / len(ghg_related) for col in ghg_related},
    **{col: 0.2 / len(land_water) for col in land_water},
    **{col: 0.2 / len(eco_harm) for col in eco_harm},
}

# Calculate EHI
df['EHI_eco_harm'] = sum(normalized[col] * weights[col] for col in ehi_columns)
df['EHI_green_house_harm'] = sum(normalized[col] * weights_2[col] for col in ehi_columns)

# Configuring Treemap
fig = px.treemap(
    df,
    path=['diet_group', 'sex', 'age_group'],
    values='EHI_green_house_harm',       # Size = greenhouse gas harm
    color='EHI_eco_harm',              # Color = eco system harm
    color_continuous_scale='RdBu_r',
    title='Treemap: Total Environmental Harm by Group (Area = EHI_greenhouse_harm, Color = EHI_eco_harm)',
)

fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
fig.show()