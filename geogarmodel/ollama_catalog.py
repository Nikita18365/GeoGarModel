"""Ollama model names, installation commands, and catalog discovery."""
from html.parser import HTMLParser
import json
import re
from urllib.request import Request, ProxyHandler, build_opener, urlopen


DEFAULT_MODEL = 'qwen3:4b'
FEATURED = (
    DEFAULT_MODEL,'qwen3.5:4b','gemma4:12b','qwen3.8:27b',
    'glm-5.3-flash:cloud','kimi-k3.1-pro:cloud','deepseek-v4.1-flash:cloud',
)
LIBRARY = 'https://ollama.com/library'
MODEL_NAME = re.compile(r'[a-zA-Z0-9][\w.-]*(?::\d{1,5})?(?:/[\w.-]+)*(?::[\w.-]+)?',re.ASCII)
FAMILY = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*')


def valid_model(name):
    return bool(len(name)<=200 and MODEL_NAME.fullmatch(name) and
                all(part not in ('.','..') for part in name.split('/')))


def is_cloud(model):
    return model.rsplit(':',1)[-1]=='cloud' or model.endswith('-cloud')


def setup_commands(model):
    if not valid_model(model):
        return ''
    if is_cloud(model):
        return f'ollama signin\nollama run {model}'
    return f'ollama pull {model}'


def installed_models():
    with build_opener(ProxyHandler({})).open('http://127.0.0.1:11434/api/tags',timeout=3) as response:
        models=json.load(response)['models']
    names = (row.get('name') or row.get('model') for row in models)
    return sorted({name for name in names if isinstance(name,str) and valid_model(name)})


class CatalogLinks(HTMLParser):
    def __init__(self,family):
        super().__init__()
        self.family=family
        self.models=set()

    def handle_starttag(self,tag,attrs):
        if tag!='a':
            return
        href=dict(attrs).get('href') or ''
        if not href.startswith('/library/'):
            return
        entry=href[len('/library/'):].split('?',1)[0]
        if self.family:
            if entry.startswith(self.family+':') and valid_model(entry):
                self.models.add(entry)
        elif FAMILY.fullmatch(entry):
            self.models.add(entry)


def catalog_models(family=None):
    if family is not None and not FAMILY.fullmatch(family):
        raise ValueError('Неверное название семейства Ollama.')
    path=LIBRARY if family is None else f'{LIBRARY}/{family}/tags'
    with urlopen(Request(path,headers={'User-Agent':'GeoGarModel/2.3'}),timeout=10) as response:
        parser=CatalogLinks(family)
        parser.feed(response.read(6_000_000).decode('utf-8'))
    if not parser.models:
        raise ValueError('Каталог Ollama не вернул список. Откройте каталог в браузере или введите тег вручную.')
    return sorted(parser.models)
