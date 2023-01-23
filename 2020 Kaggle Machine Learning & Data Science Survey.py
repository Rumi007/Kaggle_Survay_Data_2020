#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt #likley won't be used much as i'm experimenting with plotly 
import plotly.graph_objects as go #you will be learning how go and px work with me! 
import plotly.express as px 

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# Input data files are available in the read-only "../input/" directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


#load data 
df = pd.read_csv('/Users/jalilkhan/Documents/Kaggle_Survay_Data_2020/kaggle_survey_2020_responses.csv')
df.shape


# In[3]:


df.describe()
# describe the data


# In[4]:


df.head()
# look into the individual data 


# In[5]:


#remove the top row 
# where we have the questions

df_fin = df.iloc[1:,:]


# In[6]:


# now look at the head again
df_fin.head()


# In[7]:


# get percent of null values in question

df_fin.isnull().sum() / df.shape[0]


# Part of EDA is finding a way to make the data useful to you. I wanted to make it easy to run analysis on individual questions if I wanted to. The most practical way I found was to put all the questions in a dictionary. Each key in the dictionary is the Question number and each value is a dataframe with the parts to the question. I could now easily pull data for individual questions rather than filtering every time. This is particularly important for questions with multiple parts.

# In[8]:


#create a dictionary for questions 
Questions = {}

#create list of questions 
#not very efficient, but keeps things ordered
qnums = list(dict.fromkeys([i.split('_')[0] for i in df_fin.columns]))
qnums


# In[9]:


#add data for each question to key value pairs in dictionary
for i in qnums:
    if i in ['Q1','Q2','Q3']: #since we are using .startswith() below this prevents all questions that start with 
        Questions[i] = df_fin[i] #[1,2,3] from going in the key value pair (Example in vid)
    else:
        Questions[i] = df_fin[[q for q in df_fin.columns if q.startswith(i)]]


# Q1 & Q7 Examples to explain px vs go
# plotly express (px) --> takes the data frame in as a parameter and you use other paramaters to mainipulate the columns. I think this is better  that allows you to work with a full dataframe.
# 
# plotly graph objects (go) --> Takes in just the data as parameters. In this case you manipulate the data before passing it in. This is a bit more flexbile for questions like Q7 where there are columns for each answer type.
# 

# In[10]:


df_fin.Q1


# In[11]:


#q1 histogram using px 

fig = px.histogram(df_fin, x = 'Q1')
fig.show()


# In[12]:


# heatmap using px for q1 & q6

fig = px.density_heatmap(df_fin, x='Q1', y='Q6', category_orders={'Q1':['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70+'],'Q6':['I have never written code','< 1 years','1-2 years','3-5 years','5-10 years','10-20 years','20+ years']})
fig.show()


# In[13]:


Questions['Q7']


# In[14]:


# Q7 example for go use. We aggregate the data beforehand with .value_counts()
Questions['Q7'].columns = list(Questions['Q7'].mode().iloc[0,:])
q7 = Questions['Q7'].count().reset_index()
q7.columns = ['language','Count']
q7 = q7.sort_values('Count', ascending = False)
fig = go.Figure([go.Bar(x = q7.language, y = q7.Count)])
fig.show()


# The main thing I wanted to understand through this analysis was position by roles. I used a similar process as above to create a dictionary where they roles were the keys and the dataframes filtered by role were the value pairs. This might not have been the most efficient approach, but with a relatively small dataset like this, I valued ease of use over compute time.

# In[15]:


#Create dictionary with role / data key value pairs
Roles = {}
for i in df_fin.Q5.unique():
    Roles[i] = df_fin[df_fin.Q5 == i]


# In[16]:


Roles.keys()


# dict_keys(['Student', 'Data Engineer', 'Software Engineer', 'Data Scientist', 'Data Analyst', 'Research Scientist', 'Other', 'Currently not employed', 'Statistician', 'Product/Project Manager', 'Machine Learning Engineer', nan, 'Business Analyst', 'DBA/Database Engineer'])

# In[17]:


Roles['Student']


# #first subquestion --> How does education level vary by role 

# In[18]:


#all education graph
edu = df_fin.Q4.value_counts()
edu


# In[19]:


#education across whole survey sample 
fig = go.Figure([go.Bar(x=edu.index, y=edu.values)])
fig.show()


# In[20]:


#education for just data scientists 
ds_edu = Roles['Data Scientist'].Q4.value_counts()
fig = go.Figure([go.Bar(x= ds_edu.index, y=ds_edu.values)])
fig.show()


# ## Building an Advanced Graph

# I wanted to try to compare education levels between different career tracks. A great thing about plotly is that it is interactive. I wanted to explore these features to build a graph that uses a dropdown to compare different roles. The below graphs are the iterations of how I came to the final graph. 

# In[21]:


#########################################
# First Iteration - Basic dropdown 
#########################################

#https://stackoverflow.com/questions/59406167/plotly-how-to-filter-a-pandas-dataframe-using-a-dropdown-menu
#https://plotly.com/python/dropdowns/

fig = go.Figure()
fig.add_trace(go.Bar(x= edu.index, y=edu.values))

#buttons are the things you see in the dropdown 
buttons = []

#for each graph we want to show, we need a button for it
#you can do a lot with dropdowns, not just replace data 
buttons.append(dict(method='restyle',
                    label='Data Scientist',
                    visible=True,
                    args=[{'y':[Roles['Data Scientist'].Q4.value_counts().values],
                           'x':[Roles['Data Scientist'].Q4.value_counts().index],
                           'type':'bar'}, [0]],
                    )
              )
buttons.append(dict(method='restyle',
                    label='Student',
                    visible=True,
                    args=[{'y':[Roles['Student'].Q4.value_counts().values],
                           'x':[Roles['Student'].Q4.value_counts().index],
                           'type':'bar'}, [0]],
                    )
              )
buttons.append(dict(method='restyle',
                    label='Data Analyst',
                    visible=True,
                    args=[{'y':[Roles['Data Analyst'].Q4.value_counts().values],
                           'x':[Roles['Data Analyst'].Q4.value_counts().index],
                           'type':'bar'}, [0]],
                    )
              )

#to get a menu to show, you need to create an updatemenu. 
#at this point I had no clue how it worked, I just was trying to get something to run

updatemenu = []
your_menu = {}
updatemenu.append(your_menu)

updatemenu[0]['buttons'] = buttons
updatemenu[0]['direction'] = 'down'
updatemenu[0]['showactive'] = True

# add dropdown menus to the figure
fig.update_layout(showlegend=False, updatemenus=updatemenu)
fig.show()


# In[22]:


#########################################
# Second Iteration - Comparison Chart vs Baseline 
#########################################

#Added title to the figure 
fig = go.Figure(layout=go.Layout(title= go.layout.Title(text="Comparing Education by Position")))

#change to percent of group rather than raw numbers
fig.add_trace(go.Bar(name= 'Role Selection', x= edu.index, y=(edu.values/ edu.values.sum())))

#added another trace, this is the second series of bars 
fig.add_trace(go.Bar(name= 'All Data',x= edu.index, y=(edu.values/ edu.values.sum())))

#updatemenu = []
buttons = []
              
#add all roles with a loop, in previous we added them individually.
for i in list(Roles.keys())[1:]:
    buttons.append(dict(method='restyle',
                        label= i,
                        visible=True,
                        args=[{'y':[Roles[i].Q4.value_counts().values/Roles[i].Q4.value_counts().values.sum()],
                               'x':[Roles[i].Q4.value_counts().index],
                               'type':'bar'}, [0]],
                        )
                  )


#at this point I still didn't understand how this worked, I just knew it didn't add a dropdown without it 
updatemenu = []
your_menu = {}
updatemenu.append(your_menu)

updatemenu[0]['buttons'] = buttons
updatemenu[0]['direction'] = 'down'
updatemenu[0]['showactive'] = True

# add dropdown menus to the figure
fig.update_layout( updatemenus=updatemenu)

#order axes https://plotly.com/python/categorical-axes/
fig.update_xaxes(categoryorder= 'array', categoryarray= ["Doctoral degree",'Master’s degree','Bachelor’s degree','Some college/university study without earning a bachelor’s degree',"Professional degree","No formal education past high school","I prefer not to answer"])
fig.show()


# In[23]:


#########################################
# Third Iteration - Two Drop Down Comparison 
#########################################

fig = go.Figure(layout=go.Layout(title= go.layout.Title(text="Comparing Education by Position")))
fig.add_trace(go.Bar(name= 'Role Selection', x= edu.index, y=(edu.values/ edu.values.sum())))

buttons = []
# add buttons for first series of bars  
for i in list(Roles.keys())[1:]:
    buttons.append(dict(method='restyle',
                        label= i,
                        visible=True,
                        args=[{'y':[Roles[i].Q4.value_counts().values/Roles[i].Q4.value_counts().values.sum()],
                               'x':[Roles[i].Q4.value_counts().index],
                               'type':'bar'}, [0]], # the [0] at the end lets us know they are for the first trace
                        )
                  )

fig.add_trace(go.Bar(name= 'All Data',x= edu.index, y=(edu.values/ edu.values.sum())))

buttons2 = []
# add buttons for second series of bars               
for i in list(Roles.keys())[1:]:
    buttons2.append(dict(method='restyle',
                        label= i,
                        visible=True,
                        args=[{'y':[Roles[i].Q4.value_counts().values/Roles[i].Q4.value_counts().values.sum()],
                               'x':[Roles[i].Q4.value_counts().index],
                               'type':'bar'}, [1]], # the [1] at the end lets us know they are for the first trace
                        )                        #literally figured that out by just experimenting 
                  )
# adjusted dropdown placement 
#found out updatemenus take a dictionary of buttons and allow you to format how the dropdowns look etc.
# https://plotly.com/python/dropdowns/
button_layer_1_height = 1.23
updatemenus = list([
    dict(buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top"),
    dict(buttons=buttons2,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.5,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top")])
    
fig.update_layout( updatemenus=updatemenus)
fig.update_xaxes(categoryorder= 'array', categoryarray= ["Doctoral degree",'Master’s degree','Bachelor’s degree','Some college/university study without earning a bachelor’s degree',"Professional degree","No formal education past high school","I prefer not to answer"])
fig.show()

#add topline to each for all types
# add seleciton 1 and selection 2


# In[24]:


#########################################
# Final Iteration - Touch-ups
#########################################
fig = go.Figure(layout=go.Layout(title= go.layout.Title(text="Comparing Education by Position")))
#changed from role selection to selection 1
fig.add_trace(go.Bar(name= 'Selection 1', x= edu.index, y=(edu.values/ edu.values.sum())))

buttons = []

#added button for all data comparison
buttons.append(dict(method='restyle',
                        label= 'All Samples',
                        visible=True,
                        args=[{'y':[df_fin.Q4.value_counts().values/df_fin.Q4.value_counts().values.sum()],
                               'x':[df_fin.Q4.value_counts().index],
                               'type':'bar'}, [0]], # the [0] at the end lets us know they are for the first trace
                        )
                  )

for i in list(Roles.keys())[1:]:
    buttons.append(dict(method='restyle',
                        label= i,
                        visible=True,
                        args=[{'y':[Roles[i].Q4.value_counts().values/Roles[i].Q4.value_counts().values.sum()],
                               'x':[Roles[i].Q4.value_counts().index],
                               'type':'bar'}, [0]], # the [0] at the end lets us know they are for the first trace
                        )
                  )

fig.add_trace(go.Bar(name= 'Selection 2',x= edu.index, y=(edu.values/ edu.values.sum())))

buttons2 = []
#added button for all data comparison
buttons2.append(dict(method='restyle',
                        label= 'All Samples',
                        visible=True,
                        args=[{'y':[df_fin.Q4.value_counts().values/df_fin.Q4.value_counts().values.sum()],
                               'x':[df_fin.Q4.value_counts().index],
                               'type':'bar'}, [1]], # the [0] at the end lets us know they are for the first trace
                        )
                  )

for i in list(Roles.keys())[1:]:
    buttons2.append(dict(method='restyle',
                        label= i,
                        visible=True,
                        args=[{'y':[Roles[i].Q4.value_counts().values/Roles[i].Q4.value_counts().values.sum()],
                               'x':[Roles[i].Q4.value_counts().index],
                               'type':'bar'}, [1]], # the [1] at the end lets us know they are for the first trace
                        )                        #literally figured that out by just experimenting 
                  )
# adjusted dropdown placement 
#found out updatemenus take a dictionary of buttons and allow you to format how the dropdowns look etc.
# https://plotly.com/python/dropdowns/
button_layer_1_height = 1.23
updatemenus = list([
    dict(buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.11,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top"),
    dict(buttons=buttons2,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.71,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top")])
    
fig.update_layout( updatemenus=updatemenus)
#added annotations next to dropdowns 
fig.update_layout(
    annotations=[
        dict(text="Selection 1", x=0, xref="paper", y=1.15, yref="paper",
                             align="left", showarrow=False),
        dict(text="Selection 2", x=0.65, xref="paper", y=1.15,
                             yref="paper", showarrow=False)
    ])
fig.update_xaxes(categoryorder= 'array', categoryarray= ["Doctoral degree",'Master’s degree','Bachelor’s degree','Some college/university study without earning a bachelor’s degree',"Professional degree","No formal education past high school","I prefer not to answer"])
fig.show()


# Part 3  (Building Advanced Graphs)
# 
# Create more advanced graphs comparing programming languages, IDE's, etc. by role
# Create a function to easily graph results for other comparisons
# Separate notebook for comparing gender differences linked here:

# In[25]:


#########################################
# Same Format But Coding Languages Q7
#########################################
Questions['Q7']['Roles'] = df_fin.Q5

fig = go.Figure(layout=go.Layout(title= go.layout.Title(text="Comparing Coding Languages by Position")))
#changed from role selection to selection 1
fig.add_trace(go.Bar(name= 'Selection 1', x= q7.language, y=(q7.Count/ q7.Count.sum())))

def filter_bars(role, data):
    df = data[data['Roles'] == role]
    q7 = df.drop('Roles', axis= 1).count().reset_index()
    q7.columns = ['language','Count']
    return (q7.language, q7.Count/q7.Count.sum())

buttons = []

#added button for all data comparison
buttons.append(dict(method='restyle',
                        label= 'All Samples',
                        visible=True,
                        args=[{'y':[(q7.Count/ q7.Count.sum())],
                               'x':[q7.language],
                               'type':'bar'}, [0]], # the [0] at the end lets us know they are for the first trace
                        )
                  )

for i in list(Roles.keys())[1:]:
    buttons.append(dict(method='restyle',
                        label= i,
                        visible=True,
                        args=[{'y':[filter_bars(i,Questions['Q7'])[1].values],
                               'x':[filter_bars(i,Questions['Q7'])[0].values],
                               'type':'bar'}, [0]], # the [0] at the end lets us know they are for the first trace
                        )
                  )

fig.add_trace(go.Bar(name= 'Selection 2', x= q7.language, y=(q7.Count/ q7.Count.sum())))

buttons2 = []
#added button for all data comparison
buttons2.append(dict(method='restyle',
                        label= 'All Samples',
                        visible=True,
                        args=[{'y':[(q7.Count/ q7.Count.sum())],
                               'x':[q7.language],
                               'type':'bar'}, [1]], # the [0] at the end lets us know they are for the first trace
                        )
                  )

for j in list(Roles.keys())[1:]:
    buttons2.append(dict(method='restyle',
                        label= j,
                        visible=True,
                        args=[{'y':[filter_bars(j,Questions['Q7'])[1].values],
                               'x':[filter_bars(j,Questions['Q7'])[0].values],
                               'type':'bar'}, [1]], # the [1] at the end lets us know they are for the first trace
                        )                        #literally figured that out by just experimenting 
                  )
# adjusted dropdown placement 
#found out updatemenus take a dictionary of buttons and allow you to format how the dropdowns look etc.
# https://plotly.com/python/dropdowns/
button_layer_1_height = 1.15
updatemenus = list([
    dict(buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top"),
    dict(buttons=buttons2,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.50,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top")])
    
fig.update_layout( updatemenus=updatemenus)
#added annotations next to dropdowns 
fig.update_layout(
    annotations=[
        dict(text="Selection 1", x=0, xref="paper", y=1.1, yref="paper",
                             align="left", showarrow=False),
        dict(text="Selection 2", x=0.45, xref="paper", y=1.1,
                             yref="paper", showarrow=False)
    ])
fig.update_xaxes(categoryorder= 'array', categoryarray= q7.language)
fig.show()


# In[26]:


#########################################
# Same Format But for IDE's Q9
#########################################

# Q7 example for go use. We aggregate the data beforehand with .value_counts()
Questions['Q9'].columns = list(Questions['Q9'].mode().iloc[0,:])
q9 = Questions['Q9'].count().reset_index()
q9.columns = ['language','Count']
q9 = q9.sort_values('Count', ascending = False)

Questions['Q9']['Roles'] = df_fin.Q5

fig = go.Figure(layout=go.Layout(title= go.layout.Title(text="Comparing IDE's by Position")))
#changed from role selection to selection 1
fig.add_trace(go.Bar(name= 'Selection 1', x= q9.language, y=(q9.Count/ q9.Count.sum())))

buttons = []

#added button for all data comparison
buttons.append(dict(method='restyle',
                        label= 'All Samples',
                        visible=True,
                        args=[{'y':[(q9.Count/ q9.Count.sum())],
                               'x':[q9.language],
                               'type':'bar'}, [0]], # the [0] at the end lets us know they are for the first trace
                        )
                  )

for i in list(Roles.keys())[1:]:
    buttons.append(dict(method='restyle',
                        label= i,
                        visible=True,
                        args=[{'y':[filter_bars(i,Questions['Q9'])[1].values],
                               'x':[filter_bars(i,Questions['Q9'])[0].values],
                               'type':'bar'}, [0]], # the [0] at the end lets us know they are for the first trace
                        )
                  )

fig.add_trace(go.Bar(name= 'Selection 2', x= q9.language, y=(q9.Count/ q9.Count.sum())))

buttons2 = []
#added button for all data comparison
buttons2.append(dict(method='restyle',
                        label= 'All Samples',
                        visible=True,
                        args=[{'y':[(q9.Count/ q9.Count.sum())],
                               'x':[q9.language],
                               'type':'bar'}, [1]], # the [0] at the end lets us know they are for the first trace
                        )
                  )

for j in list(Roles.keys())[1:]:
    buttons2.append(dict(method='restyle',
                        label= j,
                        visible=True,
                        args=[{'y':[filter_bars(j,Questions['Q9'])[1].values],
                               'x':[filter_bars(j,Questions['Q9'])[0].values],
                               'type':'bar'}, [1]], # the [1] at the end lets us know they are for the first trace
                        )                        #literally figured that out by just experimenting 
                  )
# adjusted dropdown placement 
#found out updatemenus take a dictionary of buttons and allow you to format how the dropdowns look etc.
# https://plotly.com/python/dropdowns/
button_layer_1_height = 1.15
updatemenus = list([
    dict(buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top"),
    dict(buttons=buttons2,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.50,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top")])
    
fig.update_layout( updatemenus=updatemenus)
#added annotations next to dropdowns 
fig.update_layout(
    annotations=[
        dict(text="Selection 1", x=0, xref="paper", y=1.1, yref="paper",
                             align="left", showarrow=False),
        dict(text="Selection 2", x=0.45, xref="paper", y=1.1,
                             yref="paper", showarrow=False)
    ])
fig.update_xaxes(categoryorder= 'array', categoryarray= q9.language)
fig.show()


# In[27]:


#########################################
# Question 8 -- What would they recommend
#########################################
edu2 = df_fin.Q8.value_counts()
fig = go.Figure(layout=go.Layout(title= go.layout.Title(text="Recommended Coding Languages by Position")))
#changed from role selection to selection 1
fig.add_trace(go.Bar(name= 'Selection 1', x= edu2.index, y=(edu2.values/ edu2.values.sum())))

buttons = []

#added button for all data comparison
buttons.append(dict(method='restyle',
                        label= 'All Samples',
                        visible=True,
                        args=[{'y':[df_fin.Q8.value_counts().values/df_fin.Q8.value_counts().values.sum()],
                               'x':[df_fin.Q8.value_counts().index],
                               'type':'bar'}, [0]], # the [0] at the end lets us know they are for the first trace
                        )
                  )

for i in list(Roles.keys())[1:]:
    buttons.append(dict(method='restyle',
                        label= i,
                        visible=True,
                        args=[{'y':[Roles[i].Q8.value_counts().values/Roles[i].Q8.value_counts().values.sum()],
                               'x':[Roles[i].Q8.value_counts().index],
                               'type':'bar'}, [0]], # the [0] at the end lets us know they are for the first trace
                        )
                  )

fig.add_trace(go.Bar(name= 'Selection 2',x= edu2.index, y=(edu2.values/ edu2.values.sum())))

buttons2 = []
#added button for all data comparison
buttons2.append(dict(method='restyle',
                        label= 'All Samples',
                        visible=True,
                        args=[{'y':[df_fin.Q8.value_counts().values/df_fin.Q8.value_counts().values.sum()],
                               'x':[df_fin.Q8.value_counts().index],
                               'type':'bar'}, [1]], # the [0] at the end lets us know they are for the first trace
                        )
                  )

for i in list(Roles.keys())[1:]:
    buttons2.append(dict(method='restyle',
                        label= i,
                        visible=True,
                        args=[{'y':[Roles[i].Q8.value_counts().values/Roles[i].Q8.value_counts().values.sum()],
                               'x':[Roles[i].Q8.value_counts().index],
                               'type':'bar'}, [1]], # the [1] at the end lets us know they are for the first trace
                        )                        #literally figured that out by just experimenting 
                  )
# adjusted dropdown placement 
#found out updatemenus take a dictionary of buttons and allow you to format how the dropdowns look etc.
# https://plotly.com/python/dropdowns/
button_layer_1_height = 1.15
updatemenus = list([
    dict(buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top"),
    dict(buttons=buttons2,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.50,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top")])
    
fig.update_layout( updatemenus=updatemenus)
#added annotations next to dropdowns 
fig.update_layout(
    annotations=[
        dict(text="Selection 1", x=0, xref="paper", y=1.1, yref="paper",
                             align="left", showarrow=False),
        dict(text="Selection 2", x=0.45, xref="paper", y=1.1,
                             yref="paper", showarrow=False)
    ])
#fig.update_xaxes(categoryorder= 'array', categoryarray= ["Doctoral degree",'Master’s degree','Bachelor’s degree','Some college/university study without earning a bachelor’s degree',"Professional degree","No formal education past high school","I prefer not to answer"])
fig.show()


# In[28]:


#########################################
# Design Function 
#########################################

def filter_bars(role, data):
    df = data[data['Roles'] == role]
    q = df.drop('Roles', axis= 1).count().reset_index()
    q.columns = ['language','Count']
    return (q.language, q.Count/q.Count.sum())

def build_graph(q_number, Roles, Title):
    """Create dropdown visual with question data"""
    if isinstance(q_number, pd.DataFrame):
        qnumber = q_number.copy()
        qnumber.columns = list(qnumber.mode().iloc[0,:])
        qcnt = qnumber.count().reset_index()
        qcnt.columns = ['feature','cnt']
        qcnt = qcnt.sort_values('cnt', ascending = False)
        qnumber['Roles'] = df_fin.Q5
        

        fig = go.Figure(layout=go.Layout(title= go.layout.Title(text=Title)))
        #changed from role selection to selection 1
        fig.add_trace(go.Bar(name= 'Selection 1', x= qcnt.feature, y=(qcnt.cnt/ qcnt.cnt.sum())))

        buttons = []

        #added button for all data comparison
        buttons.append(dict(method='restyle',
                                label= 'All Samples',
                                visible=True,
                                args=[{'y':[(qcnt.cnt/ qcnt.cnt.sum())],
                                       'x':[qcnt.feature],
                                       'type':'bar'}, [0]], # the [0] at the end lets us know they are for the first trace
                                )
                          )

        for i in list(Roles.keys())[1:]:
            buttons.append(dict(method='restyle',
                                label= i,
                                visible=True,
                                args=[{'y':[filter_bars(i,qnumber)[1].values],
                                       'x':[filter_bars(i,qnumber)[0].values],
                                       'type':'bar'}, [0]], # the [0] at the end lets us know they are for the first trace
                                )
                          )

        fig.add_trace(go.Bar(name= 'Selection 2', x= qcnt.feature, y=(qcnt.cnt/ qcnt.cnt.sum())))

        buttons2 = []
        #added button for all data comparison
        buttons2.append(dict(method='restyle',
                                label= 'All Samples',
                                visible=True,
                                args=[{'y':[(qcnt.cnt/ qcnt.cnt.sum())],
                                       'x':[qcnt.feature],
                                       'type':'bar'}, [1]], 
                                )
                          )

        for i in list(Roles.keys())[1:]:
            buttons2.append(dict(method='restyle',
                                label= i,
                                visible=True,
                                args=[{'y':[filter_bars(i,qnumber)[1].values],
                                       'x':[filter_bars(i,qnumber)[0].values],
                                       'type':'bar'}, [1]],
                                )
                          )

        # adjusted dropdown placement 
        #found out updatemenus take a dictionary of buttons and allow you to format how the dropdowns look etc.
        # https://plotly.com/python/dropdowns/
        button_layer_1_height = 1.15
        updatemenus = list([
            dict(buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=button_layer_1_height,
                    yanchor="top"),
            dict(buttons=buttons2,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.50,
                    xanchor="left",
                    y=button_layer_1_height,
                    yanchor="top")])

        fig.update_layout( updatemenus=updatemenus)
        #added annotations next to dropdowns 
        fig.update_layout(
            annotations=[
                dict(text="Selection 1", x=0, xref="paper", y=1.1, yref="paper",
                                     align="left", showarrow=False),
                dict(text="Selection 2", x=0.45, xref="paper", y=1.1,
                                     yref="paper", showarrow=False)
            ])
        fig.update_xaxes(categoryorder= 'array', categoryarray= qcnt.feature)
        fig.show()
        
        
    else:
        qnumber= q_number.copy()
        vcnts = qnumber.value_counts()
        qnumber = pd.concat([qnumber,df_fin.Q5], axis =1)
        qnumber.columns = ['feature','Roles']

        fig = go.Figure(layout=go.Layout(title= go.layout.Title(text=Title)))
        #changed from role selection to selection 1
        fig.add_trace(go.Bar(name= 'Selection 1', x= vcnts.index, y=(vcnts.values/ vcnts.values.sum())))

        buttons = []

        #added button for all data comparison
        buttons.append(dict(method='restyle',
                                label= 'All Samples',
                                visible=True,
                                args=[{'y':[vcnts.values/ vcnts.values.sum()],
                                       'x':[vcnts.index],
                                       'type':'bar'}, [0]], # the [0] at the end lets us know they are for the first trace
                                )
                          )

        for i in list(Roles.keys())[1:]:
            qrole = qnumber[qnumber['Roles']==i].feature.value_counts()
            buttons.append(dict(method='restyle',
                                label= i,
                                visible=True,
                                args=[{'y':[qrole.values/qrole.values.sum()],
                                       'x':[qrole.index],
                                       'type':'bar'}, [0]], # the [0] at the end lets us know they are for the first trace
                                )
                          )

        fig.add_trace(go.Bar(name= 'Selection 2',x= vcnts.index, y=(vcnts.values/ vcnts.values.sum())))

        buttons2 = []
                #added button for all data comparison
        buttons2.append(dict(method='restyle',
                                label= 'All Samples',
                                visible=True,
                                args=[{'y':[(vcnts.values/ vcnts.values.sum())],
                                       'x':[vcnts.index],
                                       'type':'bar'}, [1]], # the [0] at the end lets us know they are for the first trace
                                )
                          )

        for i in list(Roles.keys())[1:]:
            qrole = qnumber[qnumber['Roles']==i].feature.value_counts()
            buttons2.append(dict(method='restyle',
                                label= i,
                                visible=True,
                                args=[{'y':[qrole.values/qrole.values.sum()],
                                       'x':[qrole.index],
                                       'type':'bar'}, [1]], # the [0] at the end lets us know they are for the first trace
                                )
                          )
        # adjusted dropdown placement 
        #found out updatemenus take a dictionary of buttons and allow you to format how the dropdowns look etc.
        # https://plotly.com/python/dropdowns/
        button_layer_1_height = 1.15
        updatemenus = list([
            dict(buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=button_layer_1_height,
                    yanchor="top"),
            dict(buttons=buttons2,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.50,
                    xanchor="left",
                    y=button_layer_1_height,
                    yanchor="top")])

        fig.update_layout( updatemenus=updatemenus)
        #added annotations next to dropdowns 
        fig.update_layout(
            annotations=[
                dict(text="Selection 1", x=0, xref="paper", y=1.1, yref="paper",
                                     align="left", showarrow=False),
                dict(text="Selection 2", x=0.45, xref="paper", y=1.1,
                                     yref="paper", showarrow=False)
            ])
        fig.update_xaxes(categoryorder= 'array', categoryarray= vcnts.index)
        fig.show()
        
    return


# In[29]:


build_graph(Questions['Q1'],Roles,'Age by Position')


# In[30]:


build_graph(Questions['Q12'],Roles,'Hardware by position')


# In[ ]:




