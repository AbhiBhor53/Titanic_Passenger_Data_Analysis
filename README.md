<h1>Titanic_Passenger_Analysis</h1>

<h2>Importing required resources</h2>

```python

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("Titanic-Dataset.csv")
df

```

<img width="1278" height="520" alt="image" src="https://github.com/user-attachments/assets/759a15da-629d-4655-87c4-32f57ddec95c" />


<h2>Data Cleaning Process</h2>

Age values are crucial for analysis and many values as Null

<img width="1324" height="527" alt="image" src="https://github.com/user-attachments/assets/601282af-8faa-44a6-ae25-64735b8ba81d" />

<h2>Exploratory Data Analysis</h2>

What percentage of passengers survived?

```python

Percentage_of_people_survived=(((df['Survived']==1).sum())/(df['Survived'].count()))*100
print(Percentage_of_people_survived)

```
Output : 38.38383838383838

```python

X_label=['Survived', 'Not Survived']
Y_label=[ Percentage_of_people_survived,100-Percentage_of_people_survived]
bars= plt.barh(X_label,Y_label)
plt.title('Percentage of people survived and not survived')
plt.bar_label(bars)
plt.show()

```

<img width="809" height="558" alt="image" src="https://github.com/user-attachments/assets/ce7c3ac0-2c60-44a6-9bef-847ddfedb891" />


Did gender affect survival chances?

```python

df.groupby(['Sex', 'Survived']).size()

```

<img width="248" height="149" alt="image" src="https://github.com/user-attachments/assets/c32375f1-876d-4b88-bbae-b194edaa6e15" />

```python

#Survival percentage of males

X_label=['Survived','Not Survived']
Y_label=[109/(4.68+1.09), 100 - 109/(4.68+1.09)]
bars= plt.barh(X_label,Y_label)
plt.title('Percentage of Males survived and not survived')
plt.bar_label(bars)
plt.show()

```

<img width="816" height="549" alt="image" src="https://github.com/user-attachments/assets/a6ece38b-09f1-4306-8002-72167299fde7" />


```python

#Survival percentage of females

X_label=['Survived','Not Survived']
Y_label=[233/(2.33+0.81),100- 233/(2.33+0.81)]
bars= plt.barh(X_label,Y_label)
plt.title('Percentage of Females survived and not survived')
plt.bar_label(bars)
plt.show()

```

<img width="829" height="552" alt="image" src="https://github.com/user-attachments/assets/ee518f1c-0229-4f49-b0b6-15f80e137c94" />

EDA revealed a striking gender bias: approximately 74% of female passengers survived compared to only 19% of males, reflecting historical evacuation policies rather than randomness

Which passenger class had the highest survival rate?

```python

df_new=(df.groupby(['Pclass','Survived']).size())
df_new

```

<img width="244" height="191" alt="image" src="https://github.com/user-attachments/assets/d927b501-7d6e-4f84-9550-5de39d430be8" />


```python

def percentage_survival_for_class(x) :
      percent= (df_new[x][1]/(df_new[x][0]+df_new[x][1]))*100
      return percent
Class_wise_survival_list=[]
for i in range(1,4) :
       Class_wise_survival_list.append(percentage_survival_for_class(i))

x_label=['Class 1', 'Class 2', 'Class 3']
bars=plt.barh(x_label,Class_wise_survival_list)
plt.bar_label(bars)
plt.show()

```

<img width="752" height="516" alt="image" src="https://github.com/user-attachments/assets/d3ce0591-2b82-402d-8440-a09c4c84480c" />


“First-class passengers had significantly higher survival due to proximity to lifeboats, fewer physical barriers, better access to information, and socio-economic privilege.”

How does age affect survival?

```python

df.boxplot(column='Age', by='Survived')
plt.title('Age vs Survival')
plt.suptitle('')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.show()

```

<img width="748" height="557" alt="image" src="https://github.com/user-attachments/assets/edd72d10-4507-495a-be38-4186b49d122b" />

```python

import numpy as np

plt.figure()
plt.scatter(
    df['Age'],
    df['Survived'] + np.random.normal(0, 0.02, size=len(df)),
    alpha=0.4
)
plt.xlabel('Age')
plt.ylabel('Survived')
plt.title('Age vs Survival (with jitter)')
plt.show()

```

<img width="721" height="553" alt="image" src="https://github.com/user-attachments/assets/e7e4b41f-cbb2-47c3-9f18-43cc7d017a5c" />

```python

bins = [0, 17, 58, 100]
labels = ['Children', 'Adults', 'Old Agers']

df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

bins = [0, 17, 58, 100]
labels = ['Children', 'Adults', 'Old Agers']

df['Age_Group'].value_counts()

```

<img width="304" height="119" alt="image" src="https://github.com/user-attachments/assets/68fa579e-2e3f-49cc-88e9-69cdc29fc1c4" />


```python
age_percentage = df['Age_Group'].value_counts(normalize=True) * 100
age_percentage.round(2)
```

<img width="312" height="114" alt="image" src="https://github.com/user-attachments/assets/628a0ecd-b97c-41af-991d-9b2372647226" />

```python

age_percentage.plot(kind='bar')
plt.xlabel('Age Group')
plt.ylabel('Percentage (%)')
plt.title('Passenger Distribution by Age Group')
plt.show()

```

<img width="718" height="626" alt="image" src="https://github.com/user-attachments/assets/5c7a187b-978a-476e-acc2-fb5353a92ca5" />

```python
survival_by_age_group = df.groupby('Age_Group',observed=False)['Survived'].mean() * 100

survival_by_age_group 
```

<img width="287" height="114" alt="image" src="https://github.com/user-attachments/assets/f4079d9f-a315-4dcc-bd0b-c97d25a4c945" />

```python

plt.figure()
ax = survival_by_age_group.plot(kind='bar')

plt.xlabel('Age Group')
plt.ylabel('Survival Percentage (%)')
plt.title('Survival Rate by Age Group')

# Add percentage labels on bars
for p in ax.patches:
    ax.annotate(
        f'{p.get_height():.1f}%',
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha='center',
        va='bottom'
    )

plt.show()

```

<img width="732" height="628" alt="image" src="https://github.com/user-attachments/assets/7d1753fa-6077-47e2-936b-1e4ef8a6e7d2" />


This reveals that Children have highest survival rate and old agers have lowest survival rate. It seems survival rate is decreasing with age

Which fare groups (low, mid, high) had better survival?

```python

bins = [0, 50, 150, df['Fare'].max()]
labels = ['Low', 'Mid', 'High']

df['Fare_Category'] = pd.cut(df['Fare'], bins=bins, labels=labels)

df['Fare_Category'].value_counts()

(df['Fare_Category'].value_counts(normalize=True) * 100).round(2)

fare_pct = df['Fare_Category'].value_counts(normalize=True) * 100

plt.figure()
fare_pct.plot(kind='bar')
plt.xlabel('Fare Category')
plt.ylabel('Percentage (%)')
plt.title('Passenger Distribution by Fare Category')
plt.show()

```

<img width="724" height="590" alt="image" src="https://github.com/user-attachments/assets/8fea952f-99bb-4fab-993a-2284f3dc7c29" />


```python

survival_by_fare = df.groupby('Fare_Category',observed=False)['Survived'].mean() * 100
survival_by_fare

plt.figure()
ax = survival_by_fare.plot(kind='bar')

plt.xlabel('Fare Category')
plt.ylabel('Survival Percentage (%)')
plt.title('Survival Rate by Fare Category')

# Add percentage labels on bars
for p in ax.patches:
    ax.annotate(
        f'{p.get_height():.1f}%',
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha='center',
        va='bottom'
    )

plt.show()

```

<img width="707" height="591" alt="image" src="https://github.com/user-attachments/assets/46ca8dc0-1b36-4b50-8076-efb24793ac58" />



“Fare category shows a strong positive relationship with survival. Passengers paying higher fares had significantly better survival chances due to proximity to lifeboats and class privilege.”

Conclusions
Following are the insights from tragic incident of titanic drowning:

1] Only 38.38 percent passengers were saved. From the background studies it is found that available life boats could only accommodate only 50% of total passengers. this is sheer negligence to safety protocols. For ferry managers it is of utmost importance to have enough number of life boats to accommodate all passengers in order to prevent loss of lives.

2] Analysis reveals 74% of females survived and only 18% of males survived. This shows male prioritize their women and children over their lives, In the endeavour of securing their women and children they put their lives under risks and therefore such a low survival percent.

3] Class 1 has highest survival rate due to its proximity with life boats and minimum obstructions to reach them.

4] The analysis clearly shows that survival rate of children was highest 54% then adult's survival rate 36% and old passenger’s survival rate was lowest 25%. Amidst catastrophe due to old age passengers couldn't show agility to get out. on the other hand children were saved due women children first policy and agility.

5] High fare group passengers were in class 1 which was in proximity with deck and life boats, As titanic had life boats with only 50% capacity majority of these boats were occupied by high fare group passengers. moreover, Class 1 had more families sailing containing relatively larger number of women and children therefore this class was also benefited by demographics. Also cabin crew coordinated well during evacuating passengers of class because that was initial phase of catastrophe.
