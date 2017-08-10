import requests
import pandas as pd

param = {'cot_grupo': 'frutas', 'cot_data': '01/07/2017'}
url = 'http://www.ceagesp.gov.br/entrepostos/servicos/cotacoes/'


resp = requests.post(url, data = param)

df = pd.read_html(resp.text, match='table')
