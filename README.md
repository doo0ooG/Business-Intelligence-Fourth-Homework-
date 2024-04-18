# 商务智能第四次作业 关联分析apriori实战
# 数据集来源：https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv

### 2108080217 余睿



# 代码


```python
import pandas as pd
import json
import gc
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
```


```python
pd.options.display.max_columns=100
```

## 1.读取数据


```python
# 读入元数据
movies_metadata = pd.read_csv("../data/movies_metadata.csv")
```

    d:\OTHER\software\Anaconda3\envs\doog\lib\site-packages\IPython\core\interactiveshell.py:3258: DtypeWarning: Columns (10) have mixed types.Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    


```python
# 只要 id 标题 题材（原始数据）
movies = movies_metadata[{'id', 'title', 'genres'}]

# 回收metadata
del movies_metadata
gc.collect()

movies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genres</th>
      <th>id</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'id': 16, 'name': 'Animation'}, {'id': 35, '...</td>
      <td>862</td>
      <td>Toy Story</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[{'id': 12, 'name': 'Adventure'}, {'id': 14, '...</td>
      <td>8844</td>
      <td>Jumanji</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[{'id': 10749, 'name': 'Romance'}, {'id': 35, ...</td>
      <td>15602</td>
      <td>Grumpier Old Men</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'nam...</td>
      <td>31357</td>
      <td>Waiting to Exhale</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[{'id': 35, 'name': 'Comedy'}]</td>
      <td>11862</td>
      <td>Father of the Bride Part II</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45461</th>
      <td>[{'id': 18, 'name': 'Drama'}, {'id': 10751, 'n...</td>
      <td>439050</td>
      <td>Subdue</td>
    </tr>
    <tr>
      <th>45462</th>
      <td>[{'id': 18, 'name': 'Drama'}]</td>
      <td>111109</td>
      <td>Century of Birthing</td>
    </tr>
    <tr>
      <th>45463</th>
      <td>[{'id': 28, 'name': 'Action'}, {'id': 18, 'nam...</td>
      <td>67758</td>
      <td>Betrayal</td>
    </tr>
    <tr>
      <th>45464</th>
      <td>[]</td>
      <td>227506</td>
      <td>Satan Triumphant</td>
    </tr>
    <tr>
      <th>45465</th>
      <td>[]</td>
      <td>461257</td>
      <td>Queerama</td>
    </tr>
  </tbody>
</table>
<p>45466 rows × 3 columns</p>
</div>



## 制作数据集


```python
# gpt-4编写的字符串处理函数
# 转换体裁

def genres2genre(str):
    # Since the input string uses single quotes, we need to replace them with double quotes for valid JSON format
    json_string = str.replace("'", '"')

    # Load the string as a JSON object (list of dictionaries)
    data = json.loads(json_string)

    # Extract the 'name' key from each dictionary and join them with '|'
    result = '|'.join(d['name'] for d in data)
    return result
```


```python
# 将genres转换成容易处理的形式

movies['genre'] = movies['genres'].apply(genres2genre)
movies.drop(columns='genres', inplace=True)
movies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>862</td>
      <td>Toy Story</td>
      <td>Animation|Comedy|Family</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8844</td>
      <td>Jumanji</td>
      <td>Adventure|Fantasy|Family</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15602</td>
      <td>Grumpier Old Men</td>
      <td>Romance|Comedy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31357</td>
      <td>Waiting to Exhale</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11862</td>
      <td>Father of the Bride Part II</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45461</th>
      <td>439050</td>
      <td>Subdue</td>
      <td>Drama|Family</td>
    </tr>
    <tr>
      <th>45462</th>
      <td>111109</td>
      <td>Century of Birthing</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>45463</th>
      <td>67758</td>
      <td>Betrayal</td>
      <td>Action|Drama|Thriller</td>
    </tr>
    <tr>
      <th>45464</th>
      <td>227506</td>
      <td>Satan Triumphant</td>
      <td></td>
    </tr>
    <tr>
      <th>45465</th>
      <td>461257</td>
      <td>Queerama</td>
      <td></td>
    </tr>
  </tbody>
</table>
<p>45466 rows × 3 columns</p>
</div>




```python
# 队电影题材进行ont-hot编码
movies = movies.join(movies.genre.str.get_dummies())
movies.drop(columns='genre', inplace=True)
movies
```

    C:\Users\64292\AppData\Roaming\Python\Python37\site-packages\pandas\compat\_optional.py:117: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
      if distutils.version.LooseVersion(version) < minimum_version:
    d:\OTHER\software\Anaconda3\envs\doog\lib\site-packages\setuptools\_distutils\version.py:345: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
      other = LooseVersion(other)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Aniplex</th>
      <th>BROSTA TV</th>
      <th>Carousel Productions</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>Foreign</th>
      <th>GoHands</th>
      <th>History</th>
      <th>Horror</th>
      <th>Mardock Scramble Production Committee</th>
      <th>Music</th>
      <th>Mystery</th>
      <th>Odyssey Media</th>
      <th>Pulser Productions</th>
      <th>Rogue State</th>
      <th>Romance</th>
      <th>Science Fiction</th>
      <th>Sentai Filmworks</th>
      <th>TV Movie</th>
      <th>Telescene Film Group Productions</th>
      <th>The Cartel</th>
      <th>Thriller</th>
      <th>Vision View Entertainment</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>862</td>
      <td>Toy Story</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8844</td>
      <td>Jumanji</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15602</td>
      <td>Grumpier Old Men</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31357</td>
      <td>Waiting to Exhale</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11862</td>
      <td>Father of the Bride Part II</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45461</th>
      <td>439050</td>
      <td>Subdue</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45462</th>
      <td>111109</td>
      <td>Century of Birthing</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45463</th>
      <td>67758</td>
      <td>Betrayal</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45464</th>
      <td>227506</td>
      <td>Satan Triumphant</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45465</th>
      <td>461257</td>
      <td>Queerama</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>45466 rows × 34 columns</p>
</div>



## 关联分析


```python
# 获取频繁项集
frequent_itemsets_movies = apriori(movies.drop(columns={'title', 'id'}), use_colnames=True, min_support=0.01)
```

    d:\OTHER\software\Anaconda3\envs\doog\lib\site-packages\mlxtend\frequent_patterns\fpcommon.py:113: DeprecationWarning: DataFrames with non-bool types result in worse computationalperformance and their support might be discontinued in the future.Please use a DataFrame with bool type
      DeprecationWarning,
    


```python
frequent_itemsets_movies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.145075</td>
      <td>(Action)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.076893</td>
      <td>(Adventure)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.042559</td>
      <td>(Animation)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.289931</td>
      <td>(Comedy)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.094730</td>
      <td>(Crime)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.016870</td>
      <td>(Crime, Action, Thriller)</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.019157</td>
      <td>(Drama, Action, Thriller)</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.030836</td>
      <td>(Comedy, Drama, Romance)</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.025821</td>
      <td>(Crime, Drama, Thriller)</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.015594</td>
      <td>(Mystery, Drama, Thriller)</td>
    </tr>
  </tbody>
</table>
<p>75 rows × 2 columns</p>
</div>




```python
# 获取规则
rules_movies = association_rules(frequent_itemsets_movies, metric='lift', min_threshold=1.25)
```


```python
rules_movies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
      <th>zhangs_metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Adventure)</td>
      <td>(Action)</td>
      <td>0.076893</td>
      <td>0.145075</td>
      <td>0.038116</td>
      <td>0.495709</td>
      <td>3.416908</td>
      <td>0.026961</td>
      <td>1.695301</td>
      <td>0.766257</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Action)</td>
      <td>(Adventure)</td>
      <td>0.145075</td>
      <td>0.076893</td>
      <td>0.038116</td>
      <td>0.262735</td>
      <td>3.416908</td>
      <td>0.026961</td>
      <td>1.252070</td>
      <td>0.827369</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Action)</td>
      <td>(Crime)</td>
      <td>0.145075</td>
      <td>0.094730</td>
      <td>0.030088</td>
      <td>0.207398</td>
      <td>2.189361</td>
      <td>0.016345</td>
      <td>1.142150</td>
      <td>0.635431</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Crime)</td>
      <td>(Action)</td>
      <td>0.094730</td>
      <td>0.145075</td>
      <td>0.030088</td>
      <td>0.317622</td>
      <td>2.189361</td>
      <td>0.016345</td>
      <td>1.252862</td>
      <td>0.600093</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Fantasy)</td>
      <td>(Action)</td>
      <td>0.050873</td>
      <td>0.145075</td>
      <td>0.011019</td>
      <td>0.216602</td>
      <td>1.493029</td>
      <td>0.003639</td>
      <td>1.091303</td>
      <td>0.347920</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>77</th>
      <td>(Thriller)</td>
      <td>(Drama, Crime)</td>
      <td>0.167686</td>
      <td>0.055536</td>
      <td>0.025821</td>
      <td>0.153987</td>
      <td>2.772749</td>
      <td>0.016509</td>
      <td>1.116371</td>
      <td>0.768156</td>
    </tr>
    <tr>
      <th>78</th>
      <td>(Mystery, Drama)</td>
      <td>(Thriller)</td>
      <td>0.025887</td>
      <td>0.167686</td>
      <td>0.015594</td>
      <td>0.602379</td>
      <td>3.592309</td>
      <td>0.011253</td>
      <td>2.093235</td>
      <td>0.740805</td>
    </tr>
    <tr>
      <th>79</th>
      <td>(Drama, Thriller)</td>
      <td>(Mystery)</td>
      <td>0.075375</td>
      <td>0.054260</td>
      <td>0.015594</td>
      <td>0.206886</td>
      <td>3.812850</td>
      <td>0.011504</td>
      <td>1.192439</td>
      <td>0.797868</td>
    </tr>
    <tr>
      <th>80</th>
      <td>(Mystery)</td>
      <td>(Drama, Thriller)</td>
      <td>0.054260</td>
      <td>0.075375</td>
      <td>0.015594</td>
      <td>0.287394</td>
      <td>3.812850</td>
      <td>0.011504</td>
      <td>1.297526</td>
      <td>0.780055</td>
    </tr>
    <tr>
      <th>81</th>
      <td>(Thriller)</td>
      <td>(Mystery, Drama)</td>
      <td>0.167686</td>
      <td>0.025887</td>
      <td>0.015594</td>
      <td>0.092996</td>
      <td>3.592309</td>
      <td>0.011253</td>
      <td>1.073989</td>
      <td>0.867013</td>
    </tr>
  </tbody>
</table>
<p>82 rows × 10 columns</p>
</div>




```python
# 选取提升都大于3的电影
rules_movies_lift3 = rules_movies[rules_movies['lift'] > 3].sort_values('lift', ascending=False)
rules_movies_lift3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
      <th>zhangs_metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>(Family)</td>
      <td>(Animation)</td>
      <td>0.060925</td>
      <td>0.042559</td>
      <td>0.018849</td>
      <td>0.309386</td>
      <td>7.269538</td>
      <td>0.016256</td>
      <td>1.386362</td>
      <td>0.918392</td>
    </tr>
    <tr>
      <th>18</th>
      <td>(Animation)</td>
      <td>(Family)</td>
      <td>0.042559</td>
      <td>0.060925</td>
      <td>0.018849</td>
      <td>0.442894</td>
      <td>7.269538</td>
      <td>0.016256</td>
      <td>1.685632</td>
      <td>0.900776</td>
    </tr>
    <tr>
      <th>38</th>
      <td>(Fantasy)</td>
      <td>(Family)</td>
      <td>0.050873</td>
      <td>0.060925</td>
      <td>0.013483</td>
      <td>0.265024</td>
      <td>4.350026</td>
      <td>0.010383</td>
      <td>1.277695</td>
      <td>0.811395</td>
    </tr>
    <tr>
      <th>39</th>
      <td>(Family)</td>
      <td>(Fantasy)</td>
      <td>0.060925</td>
      <td>0.050873</td>
      <td>0.013483</td>
      <td>0.221300</td>
      <td>4.350026</td>
      <td>0.010383</td>
      <td>1.218860</td>
      <td>0.820079</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(Fantasy)</td>
      <td>(Adventure)</td>
      <td>0.050873</td>
      <td>0.076893</td>
      <td>0.015000</td>
      <td>0.294855</td>
      <td>3.834635</td>
      <td>0.011088</td>
      <td>1.309103</td>
      <td>0.778841</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(Adventure)</td>
      <td>(Fantasy)</td>
      <td>0.076893</td>
      <td>0.050873</td>
      <td>0.015000</td>
      <td>0.195080</td>
      <td>3.834635</td>
      <td>0.011088</td>
      <td>1.179157</td>
      <td>0.800794</td>
    </tr>
    <tr>
      <th>80</th>
      <td>(Mystery)</td>
      <td>(Drama, Thriller)</td>
      <td>0.054260</td>
      <td>0.075375</td>
      <td>0.015594</td>
      <td>0.287394</td>
      <td>3.812850</td>
      <td>0.011504</td>
      <td>1.297526</td>
      <td>0.780055</td>
    </tr>
    <tr>
      <th>79</th>
      <td>(Drama, Thriller)</td>
      <td>(Mystery)</td>
      <td>0.075375</td>
      <td>0.054260</td>
      <td>0.015594</td>
      <td>0.206886</td>
      <td>3.812850</td>
      <td>0.011504</td>
      <td>1.192439</td>
      <td>0.797868</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(Adventure)</td>
      <td>(Family)</td>
      <td>0.076893</td>
      <td>0.060925</td>
      <td>0.017244</td>
      <td>0.224256</td>
      <td>3.680880</td>
      <td>0.012559</td>
      <td>1.210548</td>
      <td>0.788994</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(Family)</td>
      <td>(Adventure)</td>
      <td>0.060925</td>
      <td>0.076893</td>
      <td>0.017244</td>
      <td>0.283032</td>
      <td>3.680880</td>
      <td>0.012559</td>
      <td>1.287516</td>
      <td>0.775578</td>
    </tr>
    <tr>
      <th>74</th>
      <td>(Drama, Thriller)</td>
      <td>(Crime)</td>
      <td>0.075375</td>
      <td>0.094730</td>
      <td>0.025821</td>
      <td>0.342574</td>
      <td>3.616312</td>
      <td>0.018681</td>
      <td>1.376991</td>
      <td>0.782453</td>
    </tr>
    <tr>
      <th>75</th>
      <td>(Crime)</td>
      <td>(Drama, Thriller)</td>
      <td>0.094730</td>
      <td>0.075375</td>
      <td>0.025821</td>
      <td>0.272580</td>
      <td>3.616312</td>
      <td>0.018681</td>
      <td>1.271101</td>
      <td>0.799182</td>
    </tr>
    <tr>
      <th>49</th>
      <td>(Thriller)</td>
      <td>(Mystery)</td>
      <td>0.167686</td>
      <td>0.054260</td>
      <td>0.032882</td>
      <td>0.196091</td>
      <td>3.613898</td>
      <td>0.023783</td>
      <td>1.176427</td>
      <td>0.869011</td>
    </tr>
    <tr>
      <th>48</th>
      <td>(Mystery)</td>
      <td>(Thriller)</td>
      <td>0.054260</td>
      <td>0.167686</td>
      <td>0.032882</td>
      <td>0.605999</td>
      <td>3.613898</td>
      <td>0.023783</td>
      <td>2.112468</td>
      <td>0.764788</td>
    </tr>
    <tr>
      <th>78</th>
      <td>(Mystery, Drama)</td>
      <td>(Thriller)</td>
      <td>0.025887</td>
      <td>0.167686</td>
      <td>0.015594</td>
      <td>0.602379</td>
      <td>3.592309</td>
      <td>0.011253</td>
      <td>2.093235</td>
      <td>0.740805</td>
    </tr>
    <tr>
      <th>81</th>
      <td>(Thriller)</td>
      <td>(Mystery, Drama)</td>
      <td>0.167686</td>
      <td>0.025887</td>
      <td>0.015594</td>
      <td>0.092996</td>
      <td>3.592309</td>
      <td>0.011253</td>
      <td>1.073989</td>
      <td>0.867013</td>
    </tr>
    <tr>
      <th>52</th>
      <td>(Adventure, Drama)</td>
      <td>(Action)</td>
      <td>0.022940</td>
      <td>0.145075</td>
      <td>0.011481</td>
      <td>0.500479</td>
      <td>3.449787</td>
      <td>0.008153</td>
      <td>1.711490</td>
      <td>0.726800</td>
    </tr>
    <tr>
      <th>55</th>
      <td>(Action)</td>
      <td>(Adventure, Drama)</td>
      <td>0.145075</td>
      <td>0.022940</td>
      <td>0.011481</td>
      <td>0.079139</td>
      <td>3.449787</td>
      <td>0.008153</td>
      <td>1.061028</td>
      <td>0.830631</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Action)</td>
      <td>(Adventure)</td>
      <td>0.145075</td>
      <td>0.076893</td>
      <td>0.038116</td>
      <td>0.262735</td>
      <td>3.416908</td>
      <td>0.026961</td>
      <td>1.252070</td>
      <td>0.827369</td>
    </tr>
    <tr>
      <th>0</th>
      <td>(Adventure)</td>
      <td>(Action)</td>
      <td>0.076893</td>
      <td>0.145075</td>
      <td>0.038116</td>
      <td>0.495709</td>
      <td>3.416908</td>
      <td>0.026961</td>
      <td>1.695301</td>
      <td>0.766257</td>
    </tr>
    <tr>
      <th>62</th>
      <td>(Action, Thriller)</td>
      <td>(Crime)</td>
      <td>0.052127</td>
      <td>0.094730</td>
      <td>0.016870</td>
      <td>0.323629</td>
      <td>3.416323</td>
      <td>0.011932</td>
      <td>1.338421</td>
      <td>0.746184</td>
    </tr>
    <tr>
      <th>63</th>
      <td>(Crime)</td>
      <td>(Action, Thriller)</td>
      <td>0.094730</td>
      <td>0.052127</td>
      <td>0.016870</td>
      <td>0.178082</td>
      <td>3.416323</td>
      <td>0.011932</td>
      <td>1.153246</td>
      <td>0.781300</td>
    </tr>
    <tr>
      <th>60</th>
      <td>(Action, Crime)</td>
      <td>(Thriller)</td>
      <td>0.030088</td>
      <td>0.167686</td>
      <td>0.016870</td>
      <td>0.560673</td>
      <td>3.343591</td>
      <td>0.011824</td>
      <td>1.894519</td>
      <td>0.722664</td>
    </tr>
    <tr>
      <th>65</th>
      <td>(Thriller)</td>
      <td>(Action, Crime)</td>
      <td>0.167686</td>
      <td>0.030088</td>
      <td>0.016870</td>
      <td>0.100603</td>
      <td>3.343591</td>
      <td>0.011824</td>
      <td>1.078402</td>
      <td>0.842134</td>
    </tr>
    <tr>
      <th>41</th>
      <td>(Science Fiction)</td>
      <td>(Fantasy)</td>
      <td>0.067061</td>
      <td>0.050873</td>
      <td>0.011393</td>
      <td>0.169892</td>
      <td>3.339515</td>
      <td>0.007982</td>
      <td>1.143377</td>
      <td>0.750912</td>
    </tr>
    <tr>
      <th>40</th>
      <td>(Fantasy)</td>
      <td>(Science Fiction)</td>
      <td>0.050873</td>
      <td>0.067061</td>
      <td>0.011393</td>
      <td>0.223952</td>
      <td>3.339515</td>
      <td>0.007982</td>
      <td>1.202166</td>
      <td>0.738105</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(Adventure)</td>
      <td>(Animation)</td>
      <td>0.076893</td>
      <td>0.042559</td>
      <td>0.010755</td>
      <td>0.139874</td>
      <td>3.286572</td>
      <td>0.007483</td>
      <td>1.113140</td>
      <td>0.753684</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(Animation)</td>
      <td>(Adventure)</td>
      <td>0.042559</td>
      <td>0.076893</td>
      <td>0.010755</td>
      <td>0.252713</td>
      <td>3.286572</td>
      <td>0.007483</td>
      <td>1.235279</td>
      <td>0.726658</td>
    </tr>
  </tbody>
</table>
</div>




```python
rules_movies_lift3.shape
```




    (28, 10)



**总共得到28条强关联的数据**

## 保存数据


```python
frequent_itemsets_movies.to_csv('../data/frequent_itemsets_movies.csv', index=False)
rules_movies_lift3.to_csv('../data/rules_movies_lift3.csv', index=False)
```


```python

```
